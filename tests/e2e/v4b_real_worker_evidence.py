# Copyright 2026 999 Project (V4-B). Apache-2.0.
"""Real cross-process E2E evidence driver for the V4-B output chain.

Boots the real VoxCPM2 speech-worker as a subprocess (worker venv), then
speaks a real sentence through the frozen chain
   sentence-chunker → VoxCPM2 client → UniqueAudioSink → playback events,
writes the produced PCM to a WAV (audible end-to-end output), and measures:
  - first synthesizable token latency (final→first PCM),
  - release→first-audio latency,
  - caption-clock drift (cursor vs wall clock over playback),
  - barge-in stop latency inside 150 ms.

Run from the N.E.K.O fork root with the fork venv (worker venv is resolved
automatically):
  .venv/Scripts/python.exe tests/e2e/v4b_real_worker_evidence.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
import wave

FORK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER_ROOT = r"C:\Users\Administrator\Documents\999\voice_workers\voxcpm2_speech_worker"
WORKER_PYTHON = os.path.join(WORKER_ROOT, ".venv", "Scripts", "python.exe")
WORKER_HOST = "127.0.0.1"
WORKER_PORT = 8798  # avoids clashing with a resident worker on 8790
EVIDENCE_DIR = r"C:\Users\Administrator\Documents\999\puosui\docs\evidence\V4-B"

sys.path.insert(0, FORK_ROOT)
from main_logic.voice_output.contracts import PlaybackEvent  # noqa: E402
from main_logic.voice_output.controller import SpeechOutputController  # noqa: E402
from main_logic.voice_output.sink import PlaybackClock, UniqueAudioSink  # noqa: E402
from main_logic.voice_output.voxcpm2_client import (  # noqa: E402
    VoxCpm2SpeechClient,
    WebSocketSpeechWorkerTransport,
)


def _wait_healthy(timeout_s: float = 300.0) -> None:
    url = f"http://{WORKER_HOST}:{WORKER_PORT}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("speech-worker did not become healthy in time")


def _boot_worker() -> subprocess.Popen:
    env = dict(os.environ)
    env["VOXCPM2_HOST"] = WORKER_HOST
    env["VOXCPM2_PORT"] = str(WORKER_PORT)
    env["VOXCPM2_DEVICE"] = "cuda"
    return subprocess.Popen(
        [WORKER_PYTHON, "-m", "voxcpm2_speech_worker.server"],
        cwd=WORKER_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )


async def _measure() -> dict:
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    events: list[tuple[float, PlaybackEvent]] = []  # (wall_s, ev)
    lips: list[tuple[int, float, float]] = []  # (played, rms, peak)
    sink = UniqueAudioSink(
        on_event=lambda ev: events.append((time.perf_counter(), ev)),
        on_lipsync=lambda sid, rms, peak, played, ns: lips.append((played, rms, peak)),
        clock=PlaybackClock(),
    )
    transport = WebSocketSpeechWorkerTransport(f"ws://{WORKER_HOST}:{WORKER_PORT}/ws")
    client = VoxCpm2SpeechClient(transport)
    controller = SpeechOutputController(sink, client)
    await client.open()
    out: dict = {}

    # ── A. first-token latency + PCM capture + WAV 落盘 ───────────────────────
    pcm_all = bytearray()
    first_play_ms: list[float] = [0.0]
    t0 = [0.0]

    async def _speak_capture(speech_id: str, text: str) -> float:
        t0[0] = time.perf_counter()
        await controller.speak(
            speech_id,
            text,
            on_timeline=lambda tl: None,
            on_pcm=pcm_all.extend,
        )
        return (time.perf_counter() - t0[0]) * 1000.0

    text = "你好，我是声音引擎的验证进程。现在输出第一段测试语音。"
    for wall_s, ev in events:  # clear stale marker
        del wall_s, ev
    speak_end_ms = await _speak_capture("e2e1", text)
    # first AUDIBLE audio = first PLAYING with actually played samples
    for wall_s, ev in events:
        if (
            ev.speech_id == "e2e1"
            and ev.state.value == "playing"
            and ev.played_samples > 0
            and first_play_ms[0] == 0.0
        ):
            first_play_ms[0] = (wall_s - t0[0]) * 1000.0
    await asyncio.sleep(0.8)  # let sink pace to ENDED
    out["speak_coroutine_ms"] = round(speak_end_ms, 2)
    out["first_audio_ms"] = round(first_play_ms[0], 2)
    out["audio_seconds"] = round(len(pcm_all) / 2 / 48000, 3)
    out["pcm_bytes"] = len(pcm_all)

    # WAV 落盘（真实出声证据）
    wav_path = os.path.join(EVIDENCE_DIR, "v4b-real-audio.wav")
    rate = 48000
    for _, ev in events:
        if ev.speech_id == "e2e1" and ev.sample_rate:
            rate = ev.sample_rate
            break
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(pcm_all))
    out["wav_path"] = wav_path

    # ── B. caption-clock drift: cursor vs wall clock at every PLAYING ─────────
    # The sink paces played_samples at real time (~20ms quanta); drift here is
    # how far the caption cursor could be AHEAD of audible wall-clock audio,
    # bounded to ~one quantum at event-emission time.
    playing = [ev for _, ev in events if ev.state.value == "playing" and ev.speech_id == "e2e1"]
    drift_lead_ms: list[float] = []
    if playing:
        first_ns = playing[0].monotonic_ns
        fp = playing[0].played_samples
        for e in playing:
            expected_s = (e.played_samples - fp) / e.sample_rate
            wall_s = (e.monotonic_ns - first_ns) / 1e9
            # positive lead = cursor ahead of wall clock (caption would lead audio)
            lead_ms = (expected_s - wall_s) * 1000.0
            drift_lead_ms.append(max(lead_ms, 0.0))
    out["caption_drift_lead_max_ms"] = round(max(drift_lead_ms), 2) if drift_lead_ms else None
    out["caption_drift_lead_p95_ms"] = (
        round(sorted(drift_lead_ms)[int(len(drift_lead_ms) * 0.95)], 2) if drift_lead_ms else None
    )
    if drift_lead_ms:
        worst_idx = max(range(len(drift_lead_ms)), key=lambda i: drift_lead_ms[i])
        e = playing[worst_idx]
        out["caption_drift_worst_played"] = int(e.played_samples)
        out["caption_drift_worst_ns_offset_ms"] = round((e.monotonic_ns - first_ns) / 1e6, 2)
        out["caption_drift_play_count"] = len(playing)

    # ── C. lip-sync: RMS/peak reported for actually played audio ──────────────
    out["lipsync_samples"] = len(lips)
    if lips:
        out["lipsync_rms_max"] = round(max(l[1] for l in lips), 2)
        out["lipsync_peak_max"] = round(max(l[2] for l in lips), 2)

    # ── D. barge-in: stop within 150 ms ───────────────────────────────────────
    long_text = "这是一个很长的测试句子，用来验证打断是否能够在一百五十毫秒内完全停止。"
    cancel_wall_s = [0.0]
    terminal_wall_s: list[float] = []

    async def _long_speak() -> None:
        await controller.speak(
            "e2ecancel",
            long_text,
            on_pcm=lambda c: None,
        )

    events.clear()
    lips.clear()
    task = asyncio.create_task(_long_speak())
    # wait until audio is actually flowing
    deadline = time.time() + 120
    while time.time() < deadline and not any(
        ev.speech_id == "e2ecancel" and ev.state.value == "playing" for _, ev in events
    ):
        await asyncio.sleep(0.01)
    cancel_wall_s[0] = time.perf_counter()
    controller.cancel("e2ecancel")
    await task
    await asyncio.sleep(0.05)

    cancel_events = [ev for _, ev in events if ev.speech_id == "e2ecancel"]
    terminal = [ev for ev in cancel_events if ev.state.value in ("cancelled", "interrupted", "ended")]
    out["barge_in_terminal_state"] = terminal[-1].state.value if terminal else None
    out["barge_in_saw_terminal"] = bool(terminal)
    # stop latency = time from cancel() call to the final event of the stream
    last_event_wall = max((w for w, ev in events if ev.speech_id == "e2ecancel"), default=0.0)
    if cancel_wall_s[0] > 0.0 and last_event_wall > 0.0:
        out["barge_in_stop_ms"] = round((last_event_wall - cancel_wall_s[0]) * 1000.0, 2)
    else:
        out["barge_in_stop_ms"] = None
    # no PLAYING after the terminal state for the cancelled stream
    terminal_idx = None
    for i, ev in enumerate(cancel_events):
        if ev.state.value in ("cancelled", "interrupted"):
            terminal_idx = i
            break
    if terminal_idx is not None:
        out["barge_in_no_playing_after_terminal"] = all(
            ev.state.value != "playing" for ev in cancel_events[terminal_idx:]
        )
    else:
        out["barge_in_no_playing_after_terminal"] = None

    # ── E. metrics snapshot from the worker ────────────────────────────────────
    try:
        with urllib.request.urlopen(
            f"http://{WORKER_HOST}:{WORKER_PORT}/metrics", timeout=5
        ) as resp:
            out["worker_metrics"] = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        out["worker_metrics_error"] = str(exc)

    sink.close()
    await client.close()
    return out


async def main() -> int:
    proc = _boot_worker()
    try:
        _wait_healthy()
        out = await _measure()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    report_path = os.path.join(EVIDENCE_DIR, "real-worker-e2e.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))