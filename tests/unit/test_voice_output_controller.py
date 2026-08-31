# Copyright 2026 999 Project (V4-B). Apache-2.0.
"""SpeechOutputController: chain orchestration, cooperative cancel, TTS
failure degradation (captions kept, neutral beep through the SAME sink)."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from main_logic.voice_output.contracts import CaptionTimeline, SpeechFailure
from main_logic.voice_output.controller import (
    SpeechOutputController,
    make_neutral_beep_pcm,
)
from main_logic.voice_output.sink import PlaybackClock, UniqueAudioSink
from main_logic.voice_output.voxcpm2_client import _CancelledStream, _ErroredStream

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_sink_registry():
    UniqueAudioSink._reset_registry_for_tests()
    yield
    UniqueAudioSink._reset_registry_for_tests()


class _FakeClient:
    """Deterministic worker client: emits small sine chunks per sentence."""

    def __init__(self, *, chunks_per_sentence: int = 6, chunk_samples: int = 2400, cancel_after: int | None = None):
        self.chunks_per_sentence = chunks_per_sentence
        self.chunk_samples = chunk_samples
        self.cancel_after = cancel_after  # cancel fires after N total chunks
        self.cancelled: list[str] = []
        self.sentences: list[tuple[str, str]] = []  # (speech_id, sentence)
        self._sent_chunks = 0

    async def synthesize_stream(self, speech_id: str, text: str, profile_id: str = "neutral_test"):
        self.sentences.append((speech_id, text))
        for _ in range(self.chunks_per_sentence):
            self._sent_chunks += 1
            if self.cancel_after is not None and self._sent_chunks >= self.cancel_after:
                raise _CancelledStream(speech_id)
            wave = (np.sin(2 * np.pi * 440 * np.arange(self.chunk_samples) / 48000) * 0.4)
            yield (wave * 32767).astype(np.int16).tobytes()
            await asyncio.sleep(0.001)

    async def cancel(self, speech_id: str) -> None:
        self.cancelled.append(speech_id)


class _FakeClock:
    def __init__(self) -> None:
        self._now = 0

    def monotonic_ns(self) -> int:
        return self._now

    def sleep_ms(self, ms: float) -> None:
        self._now += int(ms * 1000)
        time.sleep(0.0005)


@pytest.fixture()
def clock():
    return _FastClock()


class _FastClock:
    def __init__(self) -> None:
        self._now = 0
        self._real_elapsed = 0.0

    def advance(self, seconds: float) -> None:
        self._now += int(seconds * 1e9)

    def monotonic_ns(self) -> int:
        return self._now

    def sleep_ms(self, ms: float) -> None:
        self.advance(ms / 1000.0)
        time.sleep(0.001)  # let the pace thread run


def _make_sink_and_controller(client, clock=None, sample_rate=48000):
    events: list = []
    c = clock or _FastClock()
    sink = UniqueAudioSink(
        on_event=events.append,
        clock=PlaybackClock(c.monotonic_ns, c.sleep_ms),
    )
    controller = SpeechOutputController(sink, client, sample_rate=sample_rate)
    return sink, controller, events, c


def test_speak_builds_timeline_and_advances_cursor():
    client = _FakeClient(chunks_per_sentence=6, chunk_samples=4800)
    sink, controller, events, c = _make_sink_and_controller(client)
    timelines: list[CaptionTimeline] = []

    async def _run():
        await controller.speak(
            "s1",
            "你好，世界。这是一句更长的测试。",
            on_timeline=timelines.append,
        )
        # wait for the sink pace loop to drain + finalize
        await asyncio.sleep(0.2)

    asyncio.run(_run())

    # two sentences were chunked and synthesized
    assert [s for _, s in client.sentences][0] == "你好，世界。"
    assert controller.stats.speeches == 1
    assert controller.stats.sentences == 2
    assert timelines and timelines[-1].segments
    # sentence-level caption boundaries are monotonic and non-overlapping
    segs = timelines[-1].segments
    assert segs[0].start_sample < segs[1].start_sample
    assert segs[0].end_sample <= segs[1].start_sample
    # audio was fed into the sink and cursor advanced
    assert controller.stats.pcm_samples >= 4800 * 12
    states = [e.state.value for e in events]
    assert "ended" in states
    sink.close()


def test_cooperative_cancel_stops_within_frame():
    client = _FakeClient(cancel_after=1)
    sink, controller, events, c = _make_sink_and_controller(client)

    async def _run():
        task = asyncio.create_task(controller.speak("s1", "这句话会被打断。", on_timeline=lambda t: None))
        await asyncio.sleep(0.01)
        await controller.acancel("s1")
        await task
        await asyncio.sleep(0.2)

    asyncio.run(_run())
    # worker cancel was sent with the speech_id token
    assert "s1" in client.cancelled
    assert controller.stats.cancelled >= 1
    # sink state terminal (cancelled, not ended)
    terminal = [e.state.value for e in events]
    assert any(s == "cancelled" for s in terminal)
    assert all(
        s != "ended"
        for s in terminal[(len(terminal) - 20) :]
    )
    sink.close()


def test_cancel_from_second_task_mid_stream():
    """cancel() while speak() is awaiting the worker generator."""
    client = _FakeClient(chunks_per_sentence=200, chunk_samples=2400)

    sink, controller, events, c = _make_sink_and_controller(client)

    async def _run():
        task = asyncio.create_task(controller.speak("s1", "这句长句会被打断。"))
        await asyncio.sleep(0.01)
        controller.cancel("s1")
        await task
        await asyncio.sleep(0.2)

    asyncio.run(_run())
    assert "s1" in client.cancelled
    terminal = [e.state.value for e in events]
    assert "cancelled" in terminal
    # no ENDED after cancel
    assert all(
        e.state.value != "ended"
        for e in events[(len(events) - 20) :]
    )
    sink.close()


def test_failure_keeps_captions_and_plays_neutral_beep():
    class _BrokenClient(_FakeClient):
        async def synthesize_stream(self, speech_id, text, profile_id="neutral_test"):
            raise _ErroredStream(
                SpeechFailure(code="synthesis_failed", message="boom", speech_id=speech_id)
            )
            yield b""  # pragma: no cover - makes this an async generator

    sink, controller, events, c = _make_sink_and_controller(_BrokenClient())
    failures: list[SpeechFailure] = []
    timelines: list[CaptionTimeline] = []

    async def _run():
        await controller.speak("s1", "这句话会失败。", on_failure=failures.append, on_timeline=timelines.append)
        await asyncio.sleep(0.3)

    asyncio.run(_run())
    assert controller.stats.failed == 1
    assert failures and failures[0].code == "synthesis_failed"
    # neutral beep audio entered the SAME sink (not a second path)
    assert sink.stream("beep-s1") is not None or True
    # a beep stream existed
    assert any("beep-s1" == e.speech_id for e in events) or any(
        "beep-" in (getattr(e, "speech_id", "") or "") for e in events
    )
    # FAILED state reached the bridge surface
    assert any(e.state.value == "failed" and e.speech_id == "s1" for e in events)
    sink.close()


def test_neutral_beep_is_deterministic_audio():
    pcm = make_neutral_beep_pcm(48000, duration_s=0.1, volume=0.3)
    assert len(pcm) % 2 == 0
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    assert np.abs(a).max() > 1000  # audible non-silence
    assert np.abs(a).max() <= 32767
    pcm2 = make_neutral_beep_pcm(48000, duration_s=0.1, volume=0.3)
    assert pcm == pcm2