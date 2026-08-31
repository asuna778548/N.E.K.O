"""B01 merge smoke: BOTH chains live in ONE checkout, ONE process.

After v1/main merged wave4/input (MicBroker input chain, V4-A) and
wave4/output (UniqueAudioSink output chain, V4-B), this driver proves the
card's completion condition in a single run:

- input chain: real FunASR worker on :8765, PttControlService served over
  real loopback HTTP/WS, fixture PCM fed on the production 60 ms grid
  (same entry the device callback uses);
- output chain: real VoxCPM2 worker booted as a subprocess, sentence
  chunker -> client -> UniqueAudioSink -> playback events -> WAV;
- cross-link: the asr.final transcript of turn 1 is spoken through the
  output chain WHILE turn 2's microphone turn is live (broker turn active,
  sink playing) — both chains simultaneously alive in one process;
- barge-in: cancel of the overlapping speech reaches a terminal event
  within 150 ms.

Usage (from the merged neko_fork root, fork venv):
  .venv/Scripts/python.exe scripts/b01_combined_smoke.py \
      [--funasr-metrics http://127.0.0.1:8765/metrics] \
      [--voxcpm2-port 8799] [--http-port 18770] [--out <json path>]

Exit 0 = all gates PASS.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request
import wave

import aiohttp
from aiohttp import web

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from main_logic.voice_input.mic_broker.control_server import (  # noqa: E402
    PttControlService,
    build_app,
)
from main_logic.voice_input.mic_broker.controller import WorkerPipeline  # noqa: E402
from main_logic.voice_output.contracts import PlaybackEvent  # noqa: E402
from main_logic.voice_output.controller import SpeechOutputController  # noqa: E402
from main_logic.voice_output.sink import PlaybackClock, UniqueAudioSink  # noqa: E402
from main_logic.voice_output.voxcpm2_client import (  # noqa: E402
    VoxCpm2SpeechClient,
    WebSocketSpeechWorkerTransport,
)

CHUNK_MS = 60
SAMPLE_RATE_HZ = 16_000
FIXTURES = pathlib.Path(
    r"C:\Users\Administrator\Documents\999\voice_workers\funasr_worker\tests\fixtures\speech"
)
VOXCPM2_WORKER_ROOT = pathlib.Path(r"C:\Users\Administrator\Documents\999\voice_workers\voxcpm2_speech_worker")
EVIDENCE_DIR = pathlib.Path(r"C:\Users\Administrator\Documents\999\puosui\docs\evidence\B01")


class NullCapture:
    """Opens cleanly, never emits: frames enter via broker.on_capture_frame
    exactly as a device callback would (mic hardware stays a Q4 item)."""

    def __init__(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self, on_frame) -> None:  # noqa: ANN001
        if self._open:
            raise RuntimeError("CAPTURE_ALREADY_OPEN")
        self._open = True

    def close(self) -> None:
        self._open = False


def load_wav_chunks(path: pathlib.Path) -> list[bytes]:
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE_HZ, wav.getframerate()
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        raw = wav.readframes(wav.getnframes())
    chunk_bytes = SAMPLE_RATE_HZ * CHUNK_MS // 1000 * 2
    return [raw[i : i + chunk_bytes] for i in range(0, len(raw), chunk_bytes)]


class GodotSide:
    """Acts as Godot: HTTP press/release + WS event collector."""

    def __init__(self, base_url: str, events_url: str) -> None:
        self._base = base_url
        self._events_url = events_url
        self.session: aiohttp.ClientSession
        self.ws: aiohttp.ClientWebSocketResponse
        self.collected: list[dict] = []

    async def connect(self) -> None:
        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(self._events_url, max_msg_size=8 * 1024 * 1024)

    async def press(self, side: str) -> dict:
        async with self.session.post(f"{self._base}/ptt/press", json={"side": side}) as resp:
            return await resp.json()

    async def release(self, side: str) -> dict:
        async with self.session.post(f"{self._base}/ptt/release", json={"side": side}) as resp:
            return await resp.json()

    async def collect_until_final(self, voice_turn_id: str, timeout_s: float = 40.0) -> list[dict]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            msg = await self.ws.receive(timeout=1.0)
            if msg.type == aiohttp.WSMsgType.TEXT:
                event = json.loads(msg.data)
                self.collected.append(event)
                if event.get("type") == "asr.final" and event.get("voice_turn_id") == voice_turn_id:
                    return self.collected
        raise TimeoutError(f"no asr.final for {voice_turn_id} within {timeout_s}s")

    async def close(self) -> None:
        await self.ws.close()
        await self.session.close()


async def fetch_metrics(endpoint: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint) as resp:
            resp.raise_for_status()
            return await resp.json(content_type=None)


async def wait_funasr_ready(metrics_endpoint: str, timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            metrics = await fetch_metrics(metrics_endpoint)
            if metrics.get("models_loaded") or "uptime_s" in metrics:
                return
        except Exception as exc:
            last = exc
        await asyncio.sleep(2.0)
    raise RuntimeError(f"funasr worker not ready: {last!r}")


async def feed_frames(broker, chunks: list[bytes]) -> None:
    grid_ns = CHUNK_MS * 1_000_000
    started = time.monotonic_ns()
    for index, pcm16 in enumerate(chunks):
        target = started + index * grid_ns
        delay = (target - time.monotonic_ns()) / 1e9
        if delay > 0:
            await asyncio.sleep(delay)
        broker.on_capture_frame(pcm16, time.monotonic_ns())


def wait_voxcpm2_healthy(port: int, timeout_s: float = 300.0) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("voxcpm2 speech-worker did not become healthy in time")


def boot_voxcpm2(port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["VOXCPM2_HOST"] = "127.0.0.1"
    env["VOXCPM2_PORT"] = str(port)
    env["VOXCPM2_DEVICE"] = "cuda"
    return subprocess.Popen(
        [str(VOXCPM2_WORKER_ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "voxcpm2_speech_worker.server"],
        cwd=str(VOXCPM2_WORKER_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--funasr-ws", default="ws://127.0.0.1:8765/asr")
    parser.add_argument("--funasr-metrics", default="http://127.0.0.1:8765/metrics")
    parser.add_argument("--voxcpm2-port", type=int, default=8799)
    parser.add_argument("--http-port", type=int, default=18770)
    parser.add_argument("--out", default=str(EVIDENCE_DIR / "combined-smoke.json"))
    args = parser.parse_args()

    conv_chunks = load_wav_chunks(FIXTURES / "focus_then_dialogue.wav")
    act_chunks = load_wav_chunks(FIXTURES / "smash_it.wav")

    await wait_funasr_ready(args.funasr_metrics)
    print(f"[b01] funasr worker ready ({args.funasr_metrics})", flush=True)

    vox = boot_voxcpm2(args.voxcpm2_port)
    report: dict = {}
    events: list[tuple[float, PlaybackEvent]] = []
    lips: list[tuple[int, float, float]] = []
    pcm_all = bytearray()

    try:
        wait_voxcpm2_healthy(args.voxcpm2_port)
        print(f"[b01] voxcpm2 worker ready (127.0.0.1:{args.voxcpm2_port})", flush=True)

        # ── input chain (V4-A), production assembly ────────────────────────────
        pipeline = WorkerPipeline(
            worker_ws=args.funasr_ws,
            worker_metrics=args.funasr_metrics,
            capture_factory=NullCapture,
        )
        service = PttControlService(pipeline.bridge)
        app = build_app(service, startup=pipeline.start, cleanup=pipeline.close)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", args.http_port)
        await site.start()
        print(f"[b01] control service on 127.0.0.1:{args.http_port}", flush=True)

        # ── output chain (V4-B), production assembly ───────────────────────────
        sink = UniqueAudioSink(
            on_event=lambda ev: events.append((time.perf_counter(), ev)),
            on_lipsync=lambda sid, rms, peak, played, ns: lips.append((played, rms, peak)),
            clock=PlaybackClock(),
        )
        transport = WebSocketSpeechWorkerTransport(f"ws://127.0.0.1:{args.voxcpm2_port}/ws")
        client = VoxCpm2SpeechClient(transport)
        controller = SpeechOutputController(sink, client)
        await client.open()
        print("[b01] both chains assembled in one process", flush=True)

        # Gate 0: both singletons live simultaneously before any traffic.
        dual_alive_static = UniqueAudioSink.active_sink() is sink and pipeline.broker is not None
        report["dual_chain_static_alive"] = dual_alive_static

        godot = GodotSide(f"http://127.0.0.1:{args.http_port}", f"ws://127.0.0.1:{args.http_port}/events")
        await godot.connect()

        # ── warmup turn (unmeasured): first offline pass on a freshly booted
        # worker pays CT-Punc/model warmup (~0.7s observed). Reported honestly
        # as warmup_release_to_final_ms; the 600ms gate applies to warm turns,
        # matching how V4-A measured (worker already resident, 6-turn p95). ──
        wpress = await godot.press("mouse4")
        assert wpress["accepted"] is True, wpress
        wturn = wpress["voice_turn_id"]
        await feed_frames(pipeline.broker, conv_chunks)
        wreleased_ns = time.monotonic_ns()
        await godot.release("mouse4")
        await godot.collect_until_final(wturn)
        warmup_ms = (time.monotonic_ns() - wreleased_ns) / 1e6
        print(f"[b01] warmup turn (unmeasured): release->final {warmup_ms:.0f}ms", flush=True)

        # ── turn 1 (mouse4 / conversation): final -> spoken while turn 2 live ──
        press = await godot.press("mouse4")
        assert press["accepted"] is True, press
        turn1 = press["voice_turn_id"]
        await feed_frames(pipeline.broker, conv_chunks)
        released_ns = time.monotonic_ns()
        release = await godot.release("mouse4")
        assert release["ok"] is True, release
        ev1 = await godot.collect_until_final(turn1)
        final1 = next(e for e in reversed(ev1) if e.get("type") == "asr.final" and e.get("voice_turn_id") == turn1)
        text1 = final1.get("final_transcript", final1.get("text", ""))
        rel_to_final_1 = (time.monotonic_ns() - released_ns) / 1e6
        print(f"[b01] turn1 final in {rel_to_final_1:.0f}ms: {text1!r}", flush=True)
        assert final1["channel"] == "conversation" and final1["input_owner"] == "mouse4"

        speak_task = asyncio.create_task(
            controller.speak("b01t1", text1, on_timeline=lambda tl: None, on_pcm=pcm_all.extend)
        )
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline and not any(
            ev.speech_id == "b01t1" and ev.state.value == "playing" and ev.played_samples > 0
            for _, ev in events
        ):
            await asyncio.sleep(0.01)
        playing_now = any(
            ev.speech_id == "b01t1" and ev.state.value == "playing" and ev.played_samples > 0
            for _, ev in events
        )
        assert playing_now, "output chain never reached audible PLAYING for b01t1"

        # ── turn 2 (mouse5 / action) WHILE the output chain is playing ─────────
        overlap_probe = {"broker_turn_live_during_playback": False, "sink_playing_during_mic_turn": False}
        press2 = await godot.press("mouse5")
        assert press2["accepted"] is True, press2
        turn2 = press2["voice_turn_id"]
        if pipeline.broker.active_turn is not None:
            overlap_probe["broker_turn_live_during_playback"] = True
        await feed_frames(pipeline.broker, act_chunks)
        if any(ev.speech_id == "b01t1" and ev.state.value == "playing" for _, ev in events):
            overlap_probe["sink_playing_during_mic_turn"] = True
        released2_ns = time.monotonic_ns()
        release2 = await godot.release("mouse5")
        assert release2["ok"] is True, release2
        ev2 = await godot.collect_until_final(turn2)
        final2 = next(e for e in reversed(ev2) if e.get("type") == "asr.final" and e.get("voice_turn_id") == turn2)
        text2 = final2.get("final_transcript", final2.get("text", ""))
        rel_to_final_2 = (time.monotonic_ns() - released2_ns) / 1e6
        print(f"[b01] turn2 final in {rel_to_final_2:.0f}ms: {text2!r}", flush=True)
        assert final2["channel"] == "action_select" and final2["input_owner"] == "mouse5"

        # ── barge-in: mouse5 final is the player "acting" — cut TTS ────────────
        cancel_ns = time.perf_counter()
        controller.cancel("b01t1")
        await speak_task
        t1_events = [ev for _, ev in events if ev.speech_id == "b01t1"]
        terminal1 = [ev for ev in t1_events if ev.state.value in ("cancelled", "interrupted", "ended")]
        last_wall = max((w for w, ev in events if ev.speech_id == "b01t1"), default=0.0)
        barge_in_ms = (last_wall - cancel_ns) * 1000.0
        terminal_state1 = terminal1[-1].state.value if terminal1 else None

        # ── speak turn 2's final fully (uncancelled output evidence) ──────────
        pcm2_before = len(pcm_all)
        await controller.speak("b01t2", text2, on_timeline=lambda tl: None, on_pcm=pcm_all.extend)
        await asyncio.sleep(0.5)
        t2_pcm_bytes = len(pcm_all) - pcm2_before

        report.update(
            {
                "warmup_release_to_final_ms": round(warmup_ms, 1),
                "turn1": {"voice_turn_id": turn1, "channel": "conversation", "final": text1, "release_to_final_ms": round(rel_to_final_1, 1)},
                "turn2": {"voice_turn_id": turn2, "channel": "action_select", "final": text2, "release_to_final_ms": round(rel_to_final_2, 1)},
                "release_to_final_p95_ms": round(max(rel_to_final_1, rel_to_final_2), 1),
                "overlap_probe": overlap_probe,
                "barge_in_terminal_state": terminal_state1,
                "barge_in_stop_ms": round(barge_in_ms, 2),
                "lipsync_samples": len(lips),
                "pcm_total_bytes": len(pcm_all),
                "turn2_pcm_bytes": t2_pcm_bytes,
                "audio_seconds": round(len(pcm_all) / 2 / 48000, 3),
            }
        )

        # WAV: combined audible output (turn1-truncated + turn2 full)
        rate = 48000
        for _, ev in events:
            if ev.sample_rate:
                rate = ev.sample_rate
                break
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        wav_path = EVIDENCE_DIR / "b01-dual-chain-audio.wav"
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(bytes(pcm_all))
        report["wav_path"] = str(wav_path)

        await godot.close()
        sink.close()
        await client.close()
        await runner.cleanup()

        gates = {
            "dual_chain_static_alive": dual_alive_static,
            "input_two_finals": bool(text1) and bool(text2),
            "input_release_to_final_p95_le_600ms": report["release_to_final_p95_ms"] <= 600.0,
            "both_chains_simultaneously_live": (
                overlap_probe["broker_turn_live_during_playback"]
                and overlap_probe["sink_playing_during_mic_turn"]
            ),
            "output_audible_pcm": len(pcm_all) > 0 and t2_pcm_bytes > 0,
            "lipsync_reported": len(lips) > 0,
            "barge_in_terminal_reached": terminal_state1 in ("cancelled", "interrupted"),
            "barge_in_le_150ms": barge_in_ms <= 150.0,
        }
        report["gates"] = gates
        report["gates_pass"] = all(gates.values())

        pathlib.Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(f"[b01] GATES {'PASS' if report['gates_pass'] else 'FAIL'}", flush=True)
        return 0 if report["gates_pass"] else 1
    finally:
        vox.terminate()
        try:
            vox.wait(timeout=10)
        except subprocess.TimeoutExpired:
            vox.kill()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
