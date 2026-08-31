"""V4-A control-path E2E: press/release/final over the real PttControlService wires.

Start the resident FunASR worker first (``uv run python -m funasr_worker.server``).
This driver:

- assembles the production pipeline (MicBroker + PttVoiceInputBridge +
  FunasrWorkerClient), served over real loopback HTTP/WS sockets;
- mimics the Godot side: HTTP ``POST /ptt/press|release`` + WS ``/events``;
  audio is fed through ``broker.on_capture_frame`` exactly as the device
  callback would, on the production 60 ms grid, from recorded PCM fixtures
  (real microphone hardware handoff stays a Q4 human-checks item);
- asserts broker first-press-wins over the wire, envelope fields on
  ``asr.final``, partials are received but never routed here, worker
  ``idle_inference_calls`` never grows while no PTT key is held.

Usage (from neko_fork_input, venv python):
  .venv/Scripts/python.exe scripts/v4a_control_e2e.py \
      --worker-ws ws://127.0.0.1:8765/asr --worker-metrics http://127.0.0.1:8765/metrics \
      --http-port 18766 --turns 3 --idle-seconds 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import wave
import pathlib
from typing import Any

import aiohttp
from aiohttp import web

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from main_logic.voice_input.mic_broker.control_server import PttControlService, build_app  # noqa: E402
from main_logic.voice_input.mic_broker.controller import WorkerPipeline  # noqa: E402

CHUNK_MS = 60
SAMPLE_RATE_HZ = 16_000


class NullCapture:
    """A capture port that opens cleanly but never emits. The driver feeds
    frames via broker.on_capture_frame (identical entry the device callback
    uses) so no microphone hardware is required."""

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


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
    return ordered[index]


class GodotSide:
    """Acts as Godot: HTTP press/release + WS event collector."""

    def __init__(self, base_url: str, events_url: str) -> None:
        self._base = base_url
        self._events_url = events_url
        self.session: aiohttp.ClientSession
        self.ws: aiohttp.ClientWebSocketResponse
        self.collected: list[dict[str, Any]] = []

    async def connect(self) -> None:
        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(self._events_url, max_msg_size=8 * 1024 * 1024)

    async def press(self, side: str) -> dict[str, Any]:
        async with self.session.post(f"{self._base}/ptt/press", json={"side": side}) as resp:
            return await resp.json()

    async def release(self, side: str) -> dict[str, Any]:
        snapshot = time.monotonic_ns()
        async with self.session.post(f"{self._base}/ptt/release", json={"side": side}) as resp:
            payload = await resp.json()
        return {"payload": payload, "released_ns": snapshot}

    async def collect_until_final(self, voice_turn_id: str, timeout_s: float = 40.0) -> list[dict[str, Any]]:
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
        if self.ws is not None:
            await self.ws.close()
        if self.session is not None:
            await self.session.close()


async def fetch_metrics(endpoint: str) -> dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint) as resp:
            resp.raise_for_status()
            return await resp.json(content_type=None)


async def wait_worker_ready(metrics_endpoint: str, timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            metrics = await fetch_metrics(metrics_endpoint)
            if metrics.get("models_loaded") or "uptime_s" in metrics:
                return
        except Exception as exc:  # worker still loading / not started
            last = exc
        await asyncio.sleep(2.0)
    raise RuntimeError(f"worker not ready: {last!r}")


async def feed_frames(broker, chunks: list[bytes]) -> float:
    """Push all chunks on the 60 ms grid; return time from press and first frame
    processing (press->capture latency anchor on the broker path)."""
    grid_ns = CHUNK_MS * 1_000_000
    started = time.monotonic_ns()
    first_ns = 0
    for index, pcm16 in enumerate(chunks):
        target = started + index * grid_ns
        delay = (target - time.monotonic_ns()) / 1e9
        if delay > 0:
            await asyncio.sleep(delay)
        broker.on_capture_frame(pcm16, time.monotonic_ns())
        if index == 0:
            first_ns = time.monotonic_ns()
    return (first_ns - started) / 1e6 if first_ns else 0.0


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-ws", default="ws://127.0.0.1:8765/asr")
    parser.add_argument("--worker-metrics", default="http://127.0.0.1:8765/metrics")
    parser.add_argument("--http-port", type=int, default=18766)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--idle-seconds", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    fixtures = pathlib.Path(
        r"C:\Users\Administrator\Documents\999\voice_workers\funasr_worker\tests\fixtures\speech"
    )
    conv_chunks = load_wav_chunks(fixtures / "focus_then_dialogue.wav")
    act_chunks = load_wav_chunks(fixtures / "smash_it.wav")

    await wait_worker_ready(args.worker_metrics)

    pipeline = WorkerPipeline(
        worker_ws=args.worker_ws,
        worker_metrics=args.worker_metrics,
        capture_factory=NullCapture,
    )
    service = PttControlService(pipeline.bridge)
    app = build_app(service, startup=pipeline.start, cleanup=pipeline.close)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", args.http_port)
    await site.start()
    print(f"[ctrl] control service on 127.0.0.1:{args.http_port}", flush=True)

    godot = GodotSide(f"http://127.0.0.1:{args.http_port}", f"ws://127.0.0.1:{args.http_port}/events")
    await godot.connect()
    print("[ctrl] godot-side WS connected", flush=True)

    release_to_final: list[float] = []
    press_to_first: list[float] = []
    partial_seen: dict[str, bool] = {}
    turn_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    try:
        # First-press-wins over the wire.
        first = await godot.press("mouse4")
        second = await godot.press("mouse5")
        assert first["accepted"] is True, first
        assert second["accepted"] is False, second
        voice_turn_id = first["voice_turn_id"]
        await godot.release("mouse4")

        for i in range(args.turns):
            press = await godot.press("mouse4")
            assert press["accepted"] is True, press
            turn_id = press["voice_turn_id"]
            press_ns = time.monotonic_ns()
            first_ms = await feed_frames(pipeline.broker, conv_chunks)
            press_to_first.append(first_ms)
            print(f"[ctrl] turn {i}: fed {len(conv_chunks)} chunks, broker frames_captured={pipeline.broker.metrics.frames_captured}", flush=True)
            released = await godot.release("mouse4")
            released_ns = released["released_ns"]
            events = await godot.collect_until_final(turn_id)
            final_event = next(e for e in reversed(events) if e.get("type") == "asr.final" and e.get("voice_turn_id") == turn_id)
            final_ms = (time.monotonic_ns() - released_ns) / 1e6
            release_to_final.append(final_ms)
            partial_seen[turn_id] = any(
                e.get("type") == "asr.partial" and e.get("voice_turn_id") == turn_id and bool(e.get("text"))
                for e in events
            )
            assert final_event["channel"] == "conversation"
            assert final_event["input_owner"] == "mouse4"
            assert final_event["asr_profile_id"] == "conversation.zh"
            self_final = next(
                (e for e in reversed(events) if e.get("type") == "asr.error"), None
            )
            if self_final is not None:
                errors.append(self_final)
            turn_results.append(
                {
                    "side": "mouse4",
                    "voice_turn_id": turn_id,
                    "final_transcript": final_event.get(
                        "final_transcript", final_event.get("text", "")
                    ),
                    "partial_seen_before_final": partial_seen[turn_id],
                    "release_to_final_ms": round(final_ms, 1),
                    "press_to_first_frame_ms": round(first_ms, 3),
                }
            )
            print(
                f"[ctrl] turn {i}: release->final {final_ms:.0f}ms "
                f"partial_seen={partial_seen[turn_id]} "
                f"final={final_event.get('final_transcript', final_event.get('text', ''))!r}",
                flush=True,
            )

        for i in range(args.turns):
            press = await godot.press("mouse5")
            assert press["accepted"] is True, press
            turn_id = press["voice_turn_id"]
            press_ns = time.monotonic_ns()
            first_ms = await feed_frames(pipeline.broker, act_chunks)
            press_to_first.append(first_ms)
            released = await godot.release("mouse5")
            released_ns = released["released_ns"]
            events = await godot.collect_until_final(turn_id)
            final_event = next(e for e in reversed(events) if e.get("type") == "asr.final" and e.get("voice_turn_id") == turn_id)
            final_ms = (time.monotonic_ns() - released_ns) / 1e6
            release_to_final.append(final_ms)
            partial_seen[turn_id] = any(
                e.get("type") == "asr.partial" and e.get("voice_turn_id") == turn_id and bool(e.get("text"))
                for e in events
            )
            assert final_event["channel"] == "action_select"
            assert final_event["input_owner"] == "mouse5"
            assert final_event["asr_profile_id"] == "action.zh"
            turn_results.append(
                {
                    "side": "mouse5",
                    "voice_turn_id": turn_id,
                    "final_transcript": final_event.get(
                        "final_transcript", final_event.get("text", "")
                    ),
                    "partial_seen_before_final": partial_seen[turn_id],
                    "release_to_final_ms": round(final_ms, 1),
                    "press_to_first_frame_ms": round(first_ms, 3),
                }
            )
            print(
                f"[ctrl] turn {i}: release->final {final_ms:.0f}ms "
                f"partial_seen={partial_seen[turn_id]} "
                f"final={final_event.get('text')!r}",
                flush=True,
            )

        worker_metrics = await fetch_metrics(args.worker_metrics)
        idle_result: dict[str, Any] | None = None
        if args.idle_seconds > 0:
            keys = ("online_inference_calls", "offline_inference_calls", "idle_inference_calls", "turns_started", "frames_received")
            before = {k: worker_metrics[k] for k in keys}
            print(f"[ctrl] idle window {args.idle_seconds}s (no PTT key, WS still open)", flush=True)
            await asyncio.sleep(args.idle_seconds)
            after = {k: (await fetch_metrics(args.worker_metrics))[k] for k in keys}
            idle_result = {
                "idle_seconds": args.idle_seconds,
                "counters_before": before,
                "counters_after": after,
                "zero_growth_ok": before == after,
            }
            print(f"[ctrl] idle gate: {idle_result['zero_growth_ok']}", flush=True)

        result = {
            "turn_results": turn_results,
            "partial_seen": partial_seen,
            "release_to_final_p95_ms": round(percentile(release_to_final, 0.95), 1),
            "press_to_first_frame_p95_ms": round(percentile(press_to_first, 0.95), 3),
            "worker_release_to_final": worker_metrics.get("release_to_final_latency"),
            "worker_first_partial": worker_metrics.get("first_partial_latency"),
            "worker_realtime_factor": worker_metrics.get("realtime_factor"),
            "error_events": errors,
            "idle_gate": idle_result,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.out:
            pathlib.Path(args.out).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        ok = (
            not errors
            and all(partial_seen.get(t["voice_turn_id"]) for t in turn_results)
            and result["release_to_final_p95_ms"] <= 600.0
            and (idle_result is None or idle_result["zero_growth_ok"])
        )
        print(f"[ctrl] GATES {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    finally:
        await godot.close()
        await runner.cleanup()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))