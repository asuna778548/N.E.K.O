# Copyright 2026 999 Project (V4-B). Apache-2.0.
"""Frozen contracts for the V4-B voice output chain.

These types are the message shapes exchanged with the Godot
``speech_clock_bridge`` (see the bridge protocol proposal under
``puosui/docs/contracts/proposals``). They never leak worker/device
implementation details; the ONLY authority a consumer may rely on is
``played_samples`` (sample-absolute, sample_rate-relative) plus a terminal
``state``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class PlaybackState(str, Enum):
    """Terminal- and transient playback states reported to the clock bridge.

    ``played_samples`` is the ONLY cursor a consumer may advance captions with.
    Terminal states are absolute: after ``CANCELLED``/``ENDED``/``FAILED`` the
    bridge stops advancing captions for that speech_id on the next render
    frame and may render nothing further for it.
    """

    STARTED = "started"
    PLAYING = "playing"
    ENDED = "ended"
    CANCELLED = "cancelled"
    FAILED = "failed"
    INTERRUPTED = "interrupted"  # a barge-in superseded this utterance mid-flight


@dataclass(frozen=True, slots=True)
class PlaybackEvent:
    """One observable playback-cursor emission from the unique audio sink.

    ``monotonic_ns`` is ``time.monotonic_ns()`` at the moment THIS sample
    count was the live cursor; it lets the Godot bridge detect stale/duplicate
    frames and bound transport latency.
    """

    speech_id: str
    sample_rate: int
    played_samples: int
    monotonic_ns: int
    state: PlaybackState

    def to_dict(self) -> dict:
        return {
            "speech_id": self.speech_id,
            "sample_rate": self.sample_rate,
            "played_samples": self.played_samples,
            "monotonic_ns": self.monotonic_ns,
            "state": self.state.value,
        }


@dataclass(frozen=True, slots=True)
class CaptionSegment:
    """One displayable caption slice, positioned by the playback cursor only."""

    start_sample: int
    end_sample: int
    text: str
    speaker: str = ""
    color: str = ""

    def to_dict(self) -> dict:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "text": self.text,
            "speaker": self.speaker,
            "color": self.color,
        }


@dataclass(frozen=True, slots=True)
class CaptionTimeline:
    """The caption plan for one utterance. Assembled by the sentence
    chunker / output controller, consumed read-only by the Godot bridge."""

    speech_id: str
    sample_rate: int
    segments: List[CaptionSegment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "speech_id": self.speech_id,
            "sample_rate": self.sample_rate,
            "segments": [seg.to_dict() for seg in self.segments],
        }


@dataclass(frozen=True, slots=True)
class SpeechFailure:
    """TTS failure surface: code + stable message (never raw exception text
    containing credentials). Captions stay visible; a neutral beep plays."""

    code: str
    message: str
    speech_id: str = ""
    request_id: str = ""

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "speech_id": self.speech_id,
            "request_id": self.request_id,
        }