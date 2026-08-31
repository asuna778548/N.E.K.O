"""V4-A end-to-end replay: recorded PCM through the real MicBroker pipeline.

No physical microphone is available on this machine (no PortAudio input
binding), so per the V4-A task book the full chain is exercised with recorded
16 kHz mono PCM16 wav fixtures pushed through:

  FilePlaybackCapture -> MicBroker (real) -> PttVoiceInputBridge (real)
    -> FunasrWorkerClient (real, WebSocket) -> real resident FunASR worker
    process (online Paraformer partials + offline SeACo-Paraformer final +
    FSMN-VAD + CT-Punc).

Measured gates (monotonic clock):
- press -> first PCM frame captured (broker path; file capture adds no device
  latency, real-mic measurement is a Q4 carry-over)
- release -> asr.final received (client-side wall clock)
- idle ASR inference growth over a no-key window (must be exactly 0)

Usage (from neko_fork_input, venv python):
  .venv/Scripts/python.exe scripts/v4a_e2e_replay.py \
      --ws ws://127.0.0.1:8765/asr --metrics http://127.0.0.1:8765/metrics \
      --turns 5 --idle-seconds 600 --out v4a_e2e_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import sys
import time
import wave
import pathlib
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from main_logic.voice_input.mic_broker import MicBroker, PttSide  # noqa: E402
from main_logic.voice_input.mic_broker.bridge import PttVoiceInputBridge  # noqa: E402
from main_logic.voice_input.mic_broker.capture import CapturePort  # noqa: E402

CHUNK_MS = 60
SAMPLE_RATE_HZ = 16_000


class FilePlaybackCapture(CapturePort):
    """CapturePort backed by a wav fixture; pump() plays frames immediately."""

    def __init__(self) -> None:
        self._on_frame = None
        self._open = False
        self.open_count = 0
        self.close_count = 0

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self, on_frame) -> None:  # noqa: ANN001
        if self._open:
            raise RuntimeError("CAPTURE_ALREADY_OPEN")
        self._on_frame = on_frame
        self._open = True
        self.open_count += 1

    def close(self) -> None:
        if not self._open:
            return
        self.close_count += 1
        self._open = False
        self._on_frame = None

    async def pump(self, chunks: list[bytes], realtime: bool = True) -> int:
        """Push every chunk through the broker callback.

        With ``realtime`` the frames are paced on the production 60 ms grid so
        the worker sees the same arrival pattern as a live microphone. Returns
        the absolute monotonic timestamp just after the first frame was
        processed by the broker (press->capture latency anchor)."""
        assert self._on_frame is not None and self._open
        first_ns = 0
        grid_ns = CHUNK_MS * 1_000_000
        started = time.monotonic_ns()
        for index, pcm16 in enumerate(chunks):
            if realtime:
                target = started + index * grid_ns
                delay = (target - time.monotonic_ns()) / 1e9
                if delay > 0:
                    await asyncio.sleep(delay)
            self._on_frame(pcm16, time.monotonic_ns())
            if index == 0:
                first_ns = time.monotonic_ns()
        return first_ns


def load_wav_chunks(path: pathlib.Path) -> list[bytes]:
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE_HZ, wav.getframerate()
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        raw = wav.readframes(wav.getnframes())
    chunk_bytes = SAMPLE_RATE_HZ * CHUNK_MS // 1000 * 2
    return [raw[i : i + chunk_bytes] for i in range(0, len(raw), chunk_bytes)]


class Routes:
    def __init__(self) -> None:
        self.conversation_finals: list[dict] = []
        self.action_finals: list[dict] = []
        self.partials: list[tuple[str, str]] = []
        self.cues: list[tuple[str, str]] = []
        self.errors: list[tuple[str | None, str, str]] = []

    async def on_conversation_final(self, envelope: dict) -> None:
        self.conversation_finals.append(envelope)

    async def on_action_final(self, envelope: dict) -> None:
        self.action_finals.append(envelope)

    async def on_partial_display(self, descriptor, text: str) -> None:  # noqa: ANN001
        self.partials.append((descriptor.voice_turn_id, text))

    async def on_cue(self, cue: str, side, voice_turn_id: str) -> None:  # noqa: ANN001
        self.cues.append((cue, voice_turn_id))

    async def on_error(self, voice_turn_id, code: str, detail: str) -> None:  # noqa: ANN001
        self.errors.append((voice_turn_id, code, detail))


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
    return ordered[index]


async def fetch_metrics(endpoint: str) -> dict[str, Any]:
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint) as response:
            response.raise_for_status()
            return await response.json(content_type=None)


async def wait_worker_ready(metrics_endpoint: str, timeout_s: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            metrics = await fetch_metrics(metrics_endpoint)
            return metrics
        except Exception as exc:  # worker still loading models
            last_error = exc
            await asyncio.sleep(2.0)
    raise RuntimeError(f"worker not ready: {last_error!r}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws", default="ws://127.0.0.1:8765/asr")
    parser.add_argument("--metrics", default="http://127.0.0.1:8765/metrics")
    parser.add_argument("--turns", type=int, default=5, help="turns per profile")
    parser.add_argument("--idle-seconds", type=int, default=0)
    parser.add_argument(
        "--fixtures",
        default=str(
            pathlib.Path(
                r"C:\Users\Administrator\Documents\999\voice_workers\funasr_worker\tests\fixtures\speech"
            )
        ),
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    fixtures = pathlib.Path(args.fixtures)

    from main_logic.voice_input.mic_broker.funasr_client import FunasrWorkerClient

    routes = Routes()
    capture = FilePlaybackCapture()
    broker = MicBroker(capture_factory=lambda: capture, sink=None)  # type: ignore[arg-type]
    client = FunasrWorkerClient(
        endpoint=args.ws,
        on_partial=None,  # type: ignore[arg-type]
        on_final=None,  # type: ignore[arg-type]
        on_error=None,  # type: ignore[arg-type]
        metrics_endpoint=args.metrics,
    )
    bridge = PttVoiceInputBridge(
        broker=broker,
        client=client,
        on_conversation_final=routes.on_conversation_final,
        on_action_final=routes.on_action_final,
        on_partial_display=routes.on_partial_display,
        on_cue=routes.on_cue,
        on_error=routes.on_error,
    )
    # Production wiring: worker downstream events flow through the bridge.
    client.on_partial = bridge.handle_partial
    client.on_final = bridge.handle_final
    client.on_error = bridge.handle_worker_error
    bridge.broker.sink = bridge

    print(f"[e2e] waiting for worker at {args.metrics} ...", flush=True)
    await wait_worker_ready(args.metrics)
    print("[e2e] worker ready", flush=True)

    await client.start()

    conv_chunks = load_wav_chunks(fixtures / "focus_then_dialogue.wav")
    act_chunks = load_wav_chunks(fixtures / "smash_it.wav")

    press_to_first: list[float] = []
    release_to_final: list[float] = []
    turn_results: list[dict[str, Any]] = []

    async def do_turn(side: PttSide, chunks: list[bytes], expect_channel: str) -> None:
        del expect_channel
        press_ns = time.monotonic_ns()
        assert bridge.handle_press(side, press_ns) is True
        first_frame_ns = await capture.pump(chunks, realtime=True)
        press_to_first_frame_ms = (first_frame_ns - press_ns) / 1e6
        release_ns = time.monotonic_ns()
        assert bridge.handle_release(side, release_ns) is True
        started_finals = len(routes.conversation_finals) + len(routes.action_finals)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            await asyncio.sleep(0.01)
            if len(routes.conversation_finals) + len(routes.action_finals) > started_finals:
                break
        final_ns = time.monotonic_ns()
        release_to_final_ms = (final_ns - release_ns) / 1e6
        press_to_first.append(press_to_first_frame_ms)
        release_to_final.append(release_to_final_ms)
        turn_results.append(
            {
                "side": side.value,
                "press_to_first_frame_ms": round(press_to_first_frame_ms, 3),
                "release_to_final_ms": round(release_to_final_ms, 3),
            }
        )
        print(
            f"[e2e] {side.value}: press->frame {press_to_first_frame_ms:.3f}ms, "
            f"release->final {release_to_final_ms:.1f}ms",
            flush=True,
        )

    for i in range(args.turns):
        await do_turn(PttSide.CONVERSATION, conv_chunks, "conversation")
    for i in range(args.turns):
        await do_turn(PttSide.ACTION_SELECT, act_chunks, "action_select")

    metrics_after_turns = await fetch_metrics(args.metrics)

    idle_result: dict[str, Any] | None = None
    if args.idle_seconds > 0:
        counters_before = {
            key: metrics_after_turns[key]
            for key in (
                "online_inference_calls",
                "offline_inference_calls",
                "idle_inference_calls",
                "frames_received",
                "turns_started",
            )
        }
        print(f"[e2e] idle window: {args.idle_seconds}s with no PTT key ...", flush=True)
        await asyncio.sleep(args.idle_seconds)
        metrics_after_idle = await fetch_metrics(args.metrics)
        counters_after = {
            key: metrics_after_idle[key]
            for key in counters_before
        }
        idle_result = {
            "idle_seconds": args.idle_seconds,
            "counters_before": counters_before,
            "counters_after": counters_after,
            "zero_growth_ok": counters_before == counters_after,
        }
        print(f"[e2e] idle gate: {idle_result['zero_growth_ok']}", flush=True)

    await client.close()

    worker_latency = metrics_after_turns.get("release_to_final_latency") or {}
    result = {
        "turns": turn_results,
        "press_to_first_frame_p95_ms": round(percentile(press_to_first, 0.95), 3),
        "release_to_final_p95_ms_client": round(percentile(release_to_final, 0.95), 1),
        "worker_release_to_final": worker_latency,
        "worker_first_partial": metrics_after_turns.get("first_partial_latency"),
        "worker_realtime_factor": metrics_after_turns.get("realtime_factor"),
        "worker_counters": {
            key: metrics_after_turns[key]
            for key in (
                "turns_started",
                "turns_completed",
                "turns_cancelled",
                "partials_emitted",
                "finals_emitted",
                "errors_emitted",
                "frames_received",
                "online_inference_calls",
                "offline_inference_calls",
                "idle_inference_calls",
            )
        },
        "routed_finals": {
            "conversation": len(routes.conversation_finals),
            "action_select": len(routes.action_finals),
        },
        "partial_display_count": len(routes.partials),
        "partial_texts_sample": [text for _tid, text in routes.partials[:5]],
        "envelope_sample": (
            routes.conversation_finals[0] if routes.conversation_finals else None
        ),
        "errors": routes.errors,
        "idle_gate": idle_result,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    ok = (
        not routes.errors
        and result["press_to_first_frame_p95_ms"] <= 50.0
        and result["release_to_final_p95_ms_client"] <= 600.0
        and (idle_result is None or idle_result["zero_growth_ok"])
        and result["routed_finals"]["conversation"] == args.turns
        and result["routed_finals"]["action_select"] == args.turns
    )
    print(f"[e2e] GATES {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
