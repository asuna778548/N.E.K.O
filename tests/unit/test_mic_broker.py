"""MicBroker behavior invariants (contract V1 §1; ADR 2026-08-29 §10)."""

from __future__ import annotations

import pytest

from main_logic.voice_input.mic_broker import MicBroker, PttSide
from main_logic.voice_input.mic_broker.capture import ScriptedCapture
from main_logic.voice_input.mic_broker.contracts import (
    TURN_IGNORED,
    TURN_SEALED,
    TURN_STARTED,
)


class RecordingSink:
    def __init__(self) -> None:
        self.begins: list[str] = []
        self.frames: list[tuple[str, bytes, int]] = []
        self.sealed: list[tuple[str, bytes, int]] = []
        self.cancelled: list[tuple[str, str]] = []
        self.errors: list[str] = []

    def on_turn_begin(self, descriptor) -> None:  # noqa: ANN001
        self.begins.append(descriptor.voice_turn_id)

    def on_frame(self, descriptor, pcm16: bytes, monotonic_ns: int) -> None:  # noqa: ANN001
        self.frames.append((descriptor.voice_turn_id, pcm16, monotonic_ns))

    def on_turn_sealed(self, descriptor, pcm_tail: bytes, released_ns: int) -> None:  # noqa: ANN001
        self.sealed.append((descriptor.voice_turn_id, pcm_tail, released_ns))

    def on_turn_cancelled(self, descriptor, reason: str) -> None:  # noqa: ANN001
        self.cancelled.append((descriptor.voice_turn_id, reason))

    def on_capture_error(self, descriptor, detail: str) -> None:  # noqa: ANN001
        self.errors.append(detail)


def _pcm(ms: int, value: int = 1) -> bytes:
    return int(value).to_bytes(2, "little", signed=True) * (16 * ms)


def _broker() -> tuple[MicBroker, ScriptedCapture, RecordingSink]:
    capture = ScriptedCapture()
    sink = RecordingSink()
    broker = MicBroker(capture_factory=lambda: capture, sink=sink)
    return broker, capture, sink


def test_first_press_owns_and_other_side_is_ignored_until_release() -> None:
    broker, capture, _ = _broker()

    assert broker.begin_turn(PttSide.CONVERSATION, "turn-a") is True
    assert capture.is_open is True
    # Mouse5 arrives while Mouse4 owns the mic: ignored, capture untouched.
    assert broker.begin_turn(PttSide.ACTION_SELECT, "turn-b") is False
    assert broker.active_turn.voice_turn_id == "turn-a"

    assert broker.end_turn("turn-a") is True
    assert capture.is_open is False
    # Only after release may Mouse5 own a new turn.
    assert broker.begin_turn(PttSide.ACTION_SELECT, "turn-b") is True
    assert broker.active_turn.voice_turn_id == "turn-b"


def test_same_turn_id_rebegin_is_rejected() -> None:
    broker, _, _ = _broker()
    assert broker.begin_turn(PttSide.CONVERSATION, "turn-a") is True
    assert broker.begin_turn(PttSide.CONVERSATION, "turn-a") is False
    assert broker.metrics.turns_ignored == 1


def test_end_turn_seals_ring_tail_and_only_for_owner() -> None:
    broker, capture, sink = _broker()
    broker.begin_turn(PttSide.CONVERSATION, "turn-a")
    capture.emit(_pcm(100, 3), 111)
    assert broker.end_turn("not-the-owner") is False
    assert broker.end_turn("turn-a") is True
    assert len(sink.sealed) == 1
    voice_turn_id, tail, _released_ns = sink.sealed[0]
    assert voice_turn_id == "turn-a"
    assert tail == _pcm(100, 3)
    # The ring tail is flushed exactly once and the ring is gone afterwards.
    assert broker.active_turn is None


def test_idle_means_capture_closed_and_frames_dropped_counted() -> None:
    broker, capture, _ = _broker()
    assert broker.capture_is_open is False
    # A frame with no active turn is a leak marker: counted, never processed.
    broker.on_capture_frame(_pcm(10), 1)
    assert broker.metrics.frames_dropped_idle == 1
    assert broker.metrics.frames_captured == 0


def test_cancel_discards_audio_and_closes_stream() -> None:
    broker, capture, sink = _broker()
    broker.begin_turn(PttSide.ACTION_SELECT, "turn-c")
    capture.emit(_pcm(50), 1)
    assert broker.cancel_turn("turn-c", "user_cancelled") is True
    assert capture.is_open is False
    assert sink.cancelled == [("turn-c", "user_cancelled")]
    assert sink.sealed == []


def test_device_change_recovery_drops_inflight_and_reopens() -> None:
    broker, capture, _ = _broker()
    broker.begin_turn(PttSide.CONVERSATION, "turn-a")
    capture.emit(_pcm(60), 1)
    assert broker.on_device_changed() is True
    # Recovery reopens a fresh stream immediately; the ring keeps only audio
    # captured before the change (never replayed into later turns).
    assert broker.capture_is_open is True
    assert broker.active_turn.voice_turn_id == "turn-a"
    assert broker.metrics.device_recoveries == 1
    # Device change while idle must not open anything.
    broker.end_turn("turn-a")
    assert broker.on_device_changed() is False
    assert broker.capture_is_open is False


def test_capture_open_failure_fails_closed_without_turn() -> None:
    class BrokenCapture(ScriptedCapture):
        def open(self, on_frame) -> None:  # noqa: ANN001
            raise OSError("no input device")

    sink = RecordingSink()
    broker = MicBroker(capture_factory=BrokenCapture, sink=sink)
    assert broker.begin_turn(PttSide.CONVERSATION, "turn-x") is False
    assert broker.active_turn is None
    assert broker.capture_is_open is False
    assert sink.errors


def test_descriptor_carries_contract_channel_owner_profile() -> None:
    broker, _, _ = _broker()
    broker.begin_turn(PttSide.ACTION_SELECT, "turn-m5", pressed_monotonic_ns=42)
    descriptor = broker.active_turn
    assert descriptor.channel == "action_select"
    assert descriptor.input_owner == "mouse5"
    assert descriptor.asr_profile_id == "action.zh"
    assert descriptor.pressed_monotonic_ns == 42


def test_events_record_lifecycle_order() -> None:
    broker, _, _ = _broker()
    broker.begin_turn(PttSide.CONVERSATION, "turn-a")
    broker.end_turn("turn-a")
    kinds = [event.kind for event in broker.events]
    assert kinds == ["capture_opened", TURN_STARTED, "capture_closed", TURN_SEALED]
