"""MicBroker: the unique microphone owner behind the dual PTT keys.

Invariants (all unit-tested):

- First press wins. ``begin_turn`` while a turn is active returns ``False``
  and is recorded as ``turn_ignored_other_side`` (or same-side duplicate).
- The capture stream is open if and only if a turn is active. Idle means the
  stream is closed: zero frames read, zero ASR work possible.
- ``end_turn`` only seals the owning turn id; the ring-buffer tail is flushed
  so ASR sees the full press-to-release window.
- Device changes and capture failures drop in-flight frames and recover by
  reopening the stream; they never replay stale audio into a new turn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from main_logic.asr_client.audio import AudioRingBuffer

from .contracts import (
    CAPTURE_CLOSED,
    CAPTURE_ERROR,
    CAPTURE_OPENED,
    DEVICE_CHANGED,
    INPUT_OWNER_BY_SIDE,
    ASR_PROFILE_BY_SIDE,
    CHANNEL_BY_SIDE,
    MicBrokerEvent,
    MicTurnDescriptor,
    PttSide,
    TURN_CANCELLED,
    TURN_IGNORED,
    TURN_REJECTED_ACTIVE,
    TURN_SEALED,
    TURN_STARTED,
)
from .capture import CapturePort

# Contract §2 envelope stamps absolute press/release times; the ring only has
# to hold the audio of one long PTT press.
DEFAULT_RING_CAPACITY_MS = 30_000


class VoiceFrameSink(Protocol):
    """Where captured audio flows while a turn is active."""

    def on_turn_begin(self, descriptor: MicTurnDescriptor) -> None: ...

    def on_frame(
        self,
        descriptor: MicTurnDescriptor,
        pcm16: bytes,
        monotonic_ns: int,
    ) -> None: ...

    def on_turn_sealed(
        self,
        descriptor: MicTurnDescriptor,
        pcm_tail: bytes,
        released_monotonic_ns: int,
    ) -> None: ...

    def on_turn_cancelled(
        self,
        descriptor: MicTurnDescriptor,
        reason: str,
    ) -> None: ...

    def on_capture_error(
        self,
        descriptor: MicTurnDescriptor | None,
        detail: str,
    ) -> None: ...


@dataclass(slots=True)
class MicBrokerMetrics:
    """Observable counters; the zero-idle gate reads ``frames_captured``."""

    capture_opens: int = 0
    capture_closes: int = 0
    capture_errors: int = 0
    device_recoveries: int = 0
    frames_captured: int = 0
    frames_dropped_idle: int = 0
    frames_dropped_stale: int = 0
    turns_started: int = 0
    turns_ignored: int = 0
    turns_sealed: int = 0
    turns_cancelled: int = 0
    ring_bytes_flushed: int = 0


@dataclass(slots=True)
class MicBroker:
    """Unique PTT-side microphone owner. Sync by design: capture callbacks
    arrive on the audio thread; async fan-out belongs to the sink adapter."""

    capture_factory: Callable[[], CapturePort]
    sink: VoiceFrameSink
    ring_capacity_ms: int = DEFAULT_RING_CAPACITY_MS
    now_ns: Callable[[], int] = time.monotonic_ns
    _capture: CapturePort | None = None
    _active: MicTurnDescriptor | None = None
    _ring: AudioRingBuffer | None = None
    events: list[MicBrokerEvent] = field(default_factory=list, init=False)
    metrics: MicBrokerMetrics = field(
        default_factory=MicBrokerMetrics, init=False
    )

    # -- state ------------------------------------------------------------
    @property
    def active_turn(self) -> MicTurnDescriptor | None:
        return self._active

    @property
    def capture_is_open(self) -> bool:
        return self._capture is not None and self._capture.is_open

    # -- PTT lifecycle ----------------------------------------------------
    def begin_turn(
        self,
        side: PttSide,
        voice_turn_id: str,
        pressed_monotonic_ns: int | None = None,
    ) -> bool:
        """Open the mic for ``side``. Returns False when the press cannot
        own the microphone (first-press-wins rule)."""
        pressed_ns = (
            self.now_ns() if pressed_monotonic_ns is None else pressed_monotonic_ns
        )
        if self._active is not None:
            ignored_side = side if side is not self._active.side else None
            self._emit(
                TURN_IGNORED if ignored_side else TURN_REJECTED_ACTIVE,
                voice_turn_id,
                side,
                detail=f"owner={self._active.voice_turn_id}",
            )
            self.metrics.turns_ignored += 1
            return False

        descriptor = MicTurnDescriptor(
            voice_turn_id=voice_turn_id,
            side=side,
            channel=CHANNEL_BY_SIDE[side],
            input_owner=INPUT_OWNER_BY_SIDE[side],
            asr_profile_id=ASR_PROFILE_BY_SIDE[side],
            pressed_monotonic_ns=pressed_ns,
        )
        try:
            self._open_capture()
        except Exception as exc:  # fail closed: no stream, no turn
            self._emit(
                CAPTURE_ERROR,
                voice_turn_id,
                side,
                detail=repr(exc),
            )
            self.metrics.capture_errors += 1
            self.sink.on_capture_error(descriptor, repr(exc))
            return False

        self._active = descriptor
        self._ring = AudioRingBuffer(
            capacity_ms=self.ring_capacity_ms, sample_rate_hz=16_000
        )
        self.metrics.turns_started += 1
        self._emit(TURN_STARTED, voice_turn_id, side)
        self.sink.on_turn_begin(descriptor)
        return True

    def end_turn(
        self,
        voice_turn_id: str,
        released_monotonic_ns: int | None = None,
    ) -> bool:
        """Seal the owning turn: stop capture and flush the ring tail."""
        if self._active is None or self._active.voice_turn_id != voice_turn_id:
            return False
        descriptor = self._active
        released_ns = (
            self.now_ns() if released_monotonic_ns is None else released_monotonic_ns
        )
        tail = b""
        if self._ring is not None:
            tail = self._ring.drain()
            self.metrics.ring_bytes_flushed += len(tail)
        self._close_capture()
        self._active = None
        self._ring = None
        self.metrics.turns_sealed += 1
        self._emit(TURN_SEALED, voice_turn_id, descriptor.side)
        self.sink.on_turn_sealed(descriptor, tail, released_ns)
        return True

    def cancel_turn(self, voice_turn_id: str, reason: str) -> bool:
        """Abort the owning turn; captured audio must not produce a final."""
        if self._active is None or self._active.voice_turn_id != voice_turn_id:
            return False
        descriptor = self._active
        self._close_capture()
        self._active = None
        self._ring = None
        self.metrics.turns_cancelled += 1
        self._emit(TURN_CANCELLED, voice_turn_id, descriptor.side, detail=reason)
        self.sink.on_turn_cancelled(descriptor, reason)
        return True

    # -- device / frames ---------------------------------------------------
    def on_capture_frame(self, pcm16: bytes, monotonic_ns: int) -> None:
        """Capture-thread entry point for one PCM16 chunk."""
        active = self._active
        if active is None or self._ring is None:
            # A frame arriving with no active turn means a stream leaked past
            # close(); the invariant gate treats any nonzero counter as a bug.
            self.metrics.frames_dropped_idle += 1
            return
        if len(pcm16) % 2:
            self.metrics.frames_dropped_stale += 1
            return
        self._ring.append(pcm16)
        self.metrics.frames_captured += 1
        self.sink.on_frame(active, pcm16, monotonic_ns)

    def on_device_changed(self) -> bool:
        """Handle input-device changes mid-turn; recover on the spot.

        Returns True when a stream was reopened. In-flight audio between the
        old device dying and the new stream opening is dropped (never
        replayed); a sealed turn keeps whatever was already delivered.
        """
        was_active = self._active is not None
        self._close_capture()
        self._emit(
            DEVICE_CHANGED,
            self._active.voice_turn_id if self._active else None,
            self._active.side if self._active else None,
        )
        if not was_active:
            return False
        descriptor = self._active
        self.metrics.device_recoveries += 1
        try:
            self._open_capture()
        except Exception as exc:
            self.metrics.capture_errors += 1
            self.sink.on_capture_error(descriptor, repr(exc))
            return False
        return True

    # -- internals ----------------------------------------------------------
    def _open_capture(self) -> None:
        capture = self.capture_factory()
        capture.open(self.on_capture_frame)
        self._capture = capture
        self.metrics.capture_opens += 1
        self._emit(CAPTURE_OPENED, None, None)

    def _close_capture(self) -> None:
        if self._capture is not None:
            self._capture.close()
            self._capture = None
            self.metrics.capture_closes += 1
            self._emit(CAPTURE_CLOSED, None, None)

    def _emit(
        self,
        kind: str,
        voice_turn_id: str | None,
        side: PttSide | None,
        detail: str = "",
    ) -> None:
        self.events.append(
            MicBrokerEvent(
                kind=kind,
                voice_turn_id=voice_turn_id,
                side=side,
                monotonic_ns=self.now_ns(),
                detail=detail,
            )
        )
