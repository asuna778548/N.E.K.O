# Copyright 2026 999 Project (V4-B). Apache-2.0.
"""Unique sink semantics: single instance, played_samples cursor never leads
enqueued audio, RMS/peak lipsync, cancel drops queue, terminal states."""

from __future__ import annotations

import threading
import time

import pytest

from main_logic.voice_output.sink import PlaybackClock, UniqueAudioSink

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_sink_registry():
    UniqueAudioSink._reset_registry_for_tests()
    yield
    UniqueAudioSink._reset_registry_for_tests()


class _FastClock:
    """Deterministic pacing clock: sleep returns immediately, monotonic time
    is steppable so tests do not wait real seconds."""

    def __init__(self) -> None:
        self._now = 0

    def advance_us(self, us: int) -> None:
        self._now += us

    def monotonic_ns(self) -> int:
        return self._now

    def sleep_ms(self, ms: float) -> None:
        self._now += int(ms * 1_000_000)
        # yield to the pace thread so it observes new state promptly
        time.sleep(0.001)


def _make_clock() -> _FastClock:
    return _FastClock()


@pytest.fixture()
def clock():
    return _make_clock()


def test_second_sink_is_forbidden():
    events: list = []
    sink1 = UniqueAudioSink(on_event=events.append, clock=PlaybackClock(_f := (lambda: 0), _s := (lambda ms: None)))
    assert UniqueAudioSink.active_sink() is sink1
    with pytest.raises(RuntimeError, match="SECOND_AUDIO_SINK_FORBIDDEN"):
        UniqueAudioSink(on_event=events.append, clock=PlaybackClock(_f2 := (lambda: 0), _s2 := (lambda ms: None)))
    sink1.close()


def test_played_cursor_never_leads_enqueued_audio():
    c = _make_clock()
    events: list = []
    sink = UniqueAudioSink(on_event=events.append, clock=PlaybackClock(_fast_ns := c.monotonic_ns, _fast_sleep := c.sleep_ms))
    sink.begin("s1", 48000)
    # feed exactly one chunk; cursor cannot jump ahead of emitters
    chunk = b"\x00" * 4800  # 2400 samples @48k
    sink.feed("s1", chunk)
    # wait for pace thread to consume it
    deadline = time.time() + 5
    while sink.played("s1") < 2400 and time.time() < deadline:
        time.sleep(0.002)
    total = sink.total_emitted("s1")
    assert sink.played("s1") == 2400
    assert sink.played("s1") <= total
    # after queue drains and finalize, ENDED with cursor == total emitted
    sink.finalize("s1")
    deadline = time.time() + 5
    while sink.stream("s1") is not None and time.time() < deadline:
        time.sleep(0.002)
    ended_evt = [e for e in events if e.state.value == "ended"][-1]
    assert ended_evt.played_samples == 2400
    assert ended_evt.played_samples == total
    sink.close()


def test_cancel_drops_queue_immediately():
    c = _make_clock()
    events: list = []
    sink = UniqueAudioSink(
        on_event=events.append,
        clock=PlaybackClock(c.monotonic_ns, c.sleep_ms),
    )
    sink.begin("s1", 48000)
    for _ in range(50):
        sink.feed("s1", b"\x11" * 4800)
    sink.cancel("s1")
    deadline = time.time() + 5
    while sink.stream("s1") is not None and time.time() < deadline:
        time.sleep(0.002)
    states = [e.state.value for e in events]
    assert "cancelled" in states
    assert "ended" not in states
    # nothing after cancel reached the bridge as PLAYING
    terminal_idx = next(i for i, e in enumerate(events) if e.state.value in ("cancelled", "ended"))
    for e in events[terminal_idx:]:
        assert e.state.value != "playing"
    sink.close()


def test_lipsync_rms_peak_reported_per_chunk():
    c = _make_clock()
    lip: list = []
    sink = UniqueAudioSink(
        on_event=lambda e: None,
        on_lipsync=lambda sid, rms, peak, played, ns: lip.append((rms, peak)),
        clock=PlaybackClock(c.monotonic_ns, c.sleep_ms),
    )
    import numpy as np

    loud = (np.ones(400, dtype=np.float32) * 0.8)
    pcm = (loud * 32767).astype(np.int16).tobytes()
    sink.begin("s1", 48000)
    sink.feed("s1", pcm)
    assert lip and lip[-1][0] > 20000  # loud sine-ish/flat noise
    assert lip[-1][1] > 26000
    silence = b"\x00" * 400
    sink.begin("s1", 48000)
    sink.feed("s1", silence)
    assert lip[-1][0] < 1.0
    sink.close()


def test_barge_in_on_same_speech_id_supersedes():
    c = _make_clock()
    events: list = []
    sink = UniqueAudioSink(
        on_event=events.append,
        clock=PlaybackClock(c.monotonic_ns, c.sleep_ms),
    )
    sink.begin("s1", 48000)
    sink.begin("s1", 48000)  # supersede
    deadline = time.time() + 5
    while sink.stream("s1") is not None and time.time() < deadline:
        time.sleep(0.002)
    assert any(e.state.value == "interrupted" for e in events)
    sink.close()