"""Stable contracts for the PTT microphone broker.

These types never leak provider or capture-backend details. The broker is the
sole authority on which side owns the microphone; consumers only see turn
descriptors and terminal events.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Freeze: voice-focus-and-action-routing-contract-v1 §1/§2.
CONVERSATION_PROFILE_ID = "conversation.zh"
ACTION_PROFILE_ID = "action.zh"


class PttSide(str, Enum):
    """The two registered PTT side keys (project.godot input map)."""

    CONVERSATION = "mouse4"  # Mouse4 / XBUTTON1 -> conversation_ptt
    ACTION_SELECT = "mouse5"  # Mouse5 / XBUTTON2 -> action_select_ptt


CHANNEL_BY_SIDE: dict[PttSide, str] = {
    PttSide.CONVERSATION: "conversation",
    PttSide.ACTION_SELECT: "action_select",
}

INPUT_OWNER_BY_SIDE: dict[PttSide, str] = {
    PttSide.CONVERSATION: "mouse4",
    PttSide.ACTION_SELECT: "mouse5",
}

ASR_PROFILE_BY_SIDE: dict[PttSide, str] = {
    PttSide.CONVERSATION: CONVERSATION_PROFILE_ID,
    PttSide.ACTION_SELECT: ACTION_PROFILE_ID,
}


@dataclass(frozen=True, slots=True)
class MicTurnDescriptor:
    """Identity of one press-to-release microphone turn."""

    voice_turn_id: str
    side: PttSide
    channel: str
    input_owner: str
    asr_profile_id: str
    pressed_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class MicBrokerEvent:
    """One observable broker transition; carrying no transcript payload."""

    kind: str
    voice_turn_id: str | None
    side: PttSide | None
    monotonic_ns: int
    detail: str = ""


# Event kinds emitted by MicBroker.
TURN_STARTED = "turn_started"
TURN_IGNORED = "turn_ignored_other_side"
TURN_REJECTED_ACTIVE = "turn_rejected_owner_busy"
TURN_SEALED = "turn_sealed"
TURN_CANCELLED = "turn_cancelled"
CAPTURE_OPENED = "capture_opened"
CAPTURE_CLOSED = "capture_closed"
CAPTURE_ERROR = "capture_error"
DEVICE_CHANGED = "device_changed"
