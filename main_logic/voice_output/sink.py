# Copyright 2026 999 Project (V4-B). Apache-2.0.
"""The UNIQUE audio sink of the V4-B output chain.

Every PCM byte of the frozen output chain
(``VoxCPM2 speech-worker → PCM stream → N.E.K.O unique audio_sink``) enters
this class exactly once. It is the single owner of the ``played_samples``
playback cursor, computes per-chunk RMS/peak for lip-sync, and emits
:class:`PlaybackEvent` frames plus terminal states to the registered bridge
callback (which forwards them to the Godot ``speech_clock_bridge``).

Hard boundaries:
- No second sink: only one live ``UniqueAudioSink`` instance may exist per
  process (module-level registry + assertion + audit entry).
- ``played_samples`` is advanced ONLY by actually consumed PCM (cursor never
  jumps ahead of enqueued audio), so Godot captions can never lead real
  playback (ADR: captions must not lead playback by >120ms).
- Cancel drops the remaining queue immediately; nothing after a cancel may
  reach the bridge as PLAYING.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .contracts import CaptionTimeline, PlaybackEvent, PlaybackState


@dataclass(slots=True)
class _SpeechStream:
    speech_id: str
    sample_rate: int
    started_ns: int
    played_samples: int = 0
    emitted_samples: int = 0  # samples handed to the sink before pacing
    state: PlaybackState = PlaybackState.STARTED
    timeline: Optional[CaptionTimeline] = None
    queue: List[bytes] = field(default_factory=list)  # pending PCM chunks
    rms_last: float = 0.0
    peak_last: float = 0.0
    sequence: int = 0  # event ordinal so the bridge drops stale frames
    finish_requested: bool = False


EventCallback = Callable[[PlaybackEvent], None]
LipSyncCallback = Callable[[str, float, float, int, int], None]  # speech_id, rms, peak, played_samples, monotonic_ns


@dataclass(slots=True)
class PlaybackClock:
    """Real-time pacing source; injectable for deterministic tests."""

    monotonic_ns: Callable[[], int] = time.monotonic_ns
    sleep_ms: Callable[[float], None] = lambda ms: time.sleep(ms / 1000.0)


class UniqueAudioSink:
    """Single-audio-sink for the output chain (see module docstring)."""

    _instance: Optional["UniqueAudioSink"] = None

    @classmethod
    def _reset_registry_for_tests(cls) -> None:
        live, cls._instance = cls._instance, None
        if live is not None:
            live.close()

    @classmethod
    def active_sink(cls) -> Optional["UniqueAudioSink"]:
        return cls._instance

    def __init__(
        self,
        on_event: EventCallback,
        on_lipsync: Optional[LipSyncCallback] = None,
        clock: Optional[PlaybackClock] = None,
        *,
        force: bool = False,
    ) -> None:
        if UniqueAudioSink._instance is not None and not force:
            raise RuntimeError(
                "SECOND_AUDIO_SINK_FORBIDDEN: a UniqueAudioSink is already active "
                "in this process; the frozen output chain allows exactly one."
            )
        self.on_event = on_event
        self.on_lipsync = on_lipsync
        self.clock = clock or PlaybackClock()
        self._streams: Dict[str, _SpeechStream] = {}
        self._lock = threading.RLock()
        self._threads: Dict[str, threading.Thread] = {}
        self._closed = False
        UniqueAudioSink._instance = self

    # ── lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            self._closed = True
            streams = list(self._streams.values())
            self._streams.clear()
        for stream in streams:
            self._emit(stream, PlaybackState.CANCELLED)
        for thread in self._threads.values():
            thread.join(timeout=0.5)
        self._threads.clear()

    def stream(self, speech_id: str) -> Optional[_SpeechStream]:
        with self._lock:
            return self._streams.get(speech_id)

    # ── ingestion ────────────────────────────────────────────────────────────

    def begin(
        self,
        speech_id: str,
        sample_rate: int,
        timeline: Optional[CaptionTimeline] = None,
    ) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("AUDIO_SINK_CLOSED")
            old = self._streams.get(speech_id)
            if old is not None:
                self._emit(old, PlaybackState.INTERRUPTED)
                self._streams.pop(speech_id, None)
            stream = _SpeechStream(
                speech_id=speech_id,
                sample_rate=int(sample_rate),
                started_ns=self.clock.monotonic_ns(),
                timeline=timeline,
            )
            self._streams[speech_id] = stream
        self._start_pace_thread(stream)

    def feed(self, speech_id: str, pcm16_bytes: bytes) -> None:
        if not pcm16_bytes:
            return
        if len(pcm16_bytes) % 2:
            raise ValueError("PCM16 chunk contains an incomplete sample")
        with self._lock:
            stream = self._streams.get(speech_id)
            if stream is None:
                raise LookupError(f"no stream registered for {speech_id}")
            if stream.state in (
                PlaybackState.CANCELLED,
                PlaybackState.ENDED,
                PlaybackState.INTERRUPTED,
            ):
                return  # nothing after cancel reaches the bridge
            pcm = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
            stream.rms_last = float(np.sqrt(np.mean(np.square(pcm)))) if len(pcm) else 0.0
            stream.peak_last = float(np.abs(np.max(pcm))) if len(pcm) else 0.0
            stream.emitted_samples += len(pcm)
            stream.queue.append(pcm16_bytes)
            if self.on_lipsync is not None:
                self.on_lipsync(
                    speech_id,
                    stream.rms_last,
                    stream.peak_last,
                    stream.played_samples,
                    self.clock.monotonic_ns(),
                )

    def queue_depth(self, speech_id: str) -> int:
        with self._lock:
            stream = self._streams.get(speech_id)
            return len(stream.queue) if stream else 0

    def played(self, speech_id: str) -> int:
        with self._lock:
            stream = self._streams.get(speech_id)
            return stream.played_samples if stream else 0

    def total_emitted(self, speech_id: str) -> int:
        with self._lock:
            stream = self._streams.get(speech_id)
            return stream.emitted_samples if stream else 0

    # ── cancel / end / fail ──────────────────────────────────────────────────

    def cancel(self, speech_id: str) -> None:
        with self._lock:
            stream = self._streams.get(speech_id)
            if stream is None:
                return
            if stream.state in (PlaybackState.CANCELLED, PlaybackState.ENDED, PlaybackState.INTERRUPTED):
                return
            stream.queue.clear()
            stream.state = PlaybackState.CANCELLED

    def finalize(self, speech_id: str) -> None:
        """Controller calls this once the worker reports DONE: keep pacing the
        tail that already reached the sink, then emit ENDED."""
        with self._lock:
            stream = self._streams.get(speech_id)
            if stream is None:
                return
            if stream.state in (PlaybackState.CANCELLED, PlaybackState.INTERRUPTED):
                return
            stream.finish_requested = True

    def fail(self, speech_id: str, code: str, message: str) -> None:
        with self._lock:
            stream = self._streams.get(speech_id)
            if stream is None:
                return
            stream.queue.clear()
            stream.state = PlaybackState.FAILED
            stream.finish_requested = True
        self._emit(stream, PlaybackState.FAILED)

    # ── pacing loop ──────────────────────────────────────────────────────────

    def _start_pace_thread(self, stream: _SpeechStream) -> None:
        thread = threading.Thread(target=self._pace_loop, args=(stream,), daemon=True)
        self._threads[stream.speech_id] = thread
        thread.start()

    def _pace_loop(self, stream: _SpeechStream) -> None:
        started_real = self.clock.monotonic_ns()
        self._emit(stream, PlaybackState.STARTED)
        while True:
            with self._lock:
                if self._closed:
                    return
                if stream.state in (PlaybackState.CANCELLED, PlaybackState.INTERRUPTED):
                    self._emit(stream, stream.state)
                    self._streams.pop(stream.speech_id, None)
                    self._threads.pop(stream.speech_id, None)
                    return
                if stream.state is PlaybackState.FAILED:
                    # fail() already emitted FAILED; just clean up, never ENDED.
                    self._streams.pop(stream.speech_id, None)
                    self._threads.pop(stream.speech_id, None)
                    return
                ended = stream.finish_requested and not stream.queue
                if ended:
                    self._emit(stream, PlaybackState.ENDED)
                    self._streams.pop(stream.speech_id, None)
                    self._threads.pop(stream.speech_id, None)
                    return
                chunk = stream.queue.pop(0) if stream.queue else None
            if chunk is None:
                self.clock.sleep_ms(2)
                continue
            samples = len(chunk) // 2
            with self._lock:
                stream.state = PlaybackState.PLAYING
                stream.played_samples += samples
                played = stream.played_samples
            self._emit(stream, PlaybackState.PLAYING, played_samples=played)
            target_ns = started_real + int(played / stream.sample_rate * 1e9)
            now = self.clock.monotonic_ns()
            remain_ms = (target_ns - now) / 1e6
            if remain_ms > 0.5:
                self.clock.sleep_ms(min(remain_ms, 10.0))

    # ── emission ─────────────────────────────────────────────────────────────

    def _emit(
        self,
        stream: _SpeechStream,
        state: PlaybackState,
        played_samples: Optional[int] = None,
    ) -> None:
        stream.sequence += 1
        self.on_event(
            PlaybackEvent(
                speech_id=stream.speech_id,
                sample_rate=stream.sample_rate,
                played_samples=played_samples if played_samples is not None else stream.played_samples,
                monotonic_ns=self.clock.monotonic_ns(),
                state=state,
            )
        )