"""Unique-mic-owner PTT broker for the 999 dual-PTT voice production chain.

Freeze reference: ``puosui/docs/contracts/voice-focus-and-action-routing-contract-v1.md``
(§1 input channels, §2 ``VoiceTurnEnvelope``) and
``puosui/docs/reviews/companion-runtime-final-adr-2026-08-03.md`` §5.3.

The broker is the only owner of the microphone capture stream:

- The first pressed PTT side owns the current ``voice_turn_id``. The other
  side is ignored until the owning turn is released or cancelled.
- Capture is open exactly while a turn is active. When no side key is held,
  the capture stream is closed and zero frames are read (no capture, no ASR
  inference: the FunASR worker stays resident but idle).
- ``end_turn`` seals the turn: the capture stream is closed and the retained
  ring-buffer tail is flushed to the sink so the ASR final is produced from
  the complete press-to-release window.
- Device changes / stream drop are recovered by reopening the capture on the
  next turn; frames lost during the outage are dropped, never replayed.
"""

from .contracts import (
    ASR_PROFILE_BY_SIDE,
    CHANNEL_BY_SIDE,
    INPUT_OWNER_BY_SIDE,
    MicBrokerEvent,
    MicTurnDescriptor,
    PttSide,
)
from .broker import MicBroker

__all__ = [
    "ASR_PROFILE_BY_SIDE",
    "CHANNEL_BY_SIDE",
    "INPUT_OWNER_BY_SIDE",
    "MicBroker",
    "MicBrokerEvent",
    "MicTurnDescriptor",
    "PttSide",
]
