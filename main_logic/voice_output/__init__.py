# Copyright 2026 999 Project (V4-B). Apache-2.0.
"""V4-B voice output side: the unique audio sink, sentence output chain,
VoxCPM2 speech-worker client, cancel chain, semantic cue mapping and the
caption/lip-sync clock surface consumed by the Godot speech_clock_bridge.

Frozen boundary (ADR companion-runtime-final §1/§5/§7, V4-B package spec):
  PCM enters this package's :class:`UniqueAudioSink` only; it is the sole
  audio sink and the sole owner of the ``played_samples`` playback cursor.
  Nothing in this package pushes the same audio to a second output path.
"""

from __future__ import annotations

from .contracts import (
    CaptionSegment,
    CaptionTimeline,
    PlaybackEvent,
    PlaybackState,
    SpeechFailure,
)
from .sink import UniqueAudioSink
from .sentencizer import SafeSentenceChunker
from .controller import SpeechOutputController
from .semantic_cue import CueToPerformanceMapper, PERFORMANCE_PROFILES

__all__ = [
    "UniqueAudioSink",
    "SafeSentenceChunker",
    "SpeechOutputController",
    "CueToPerformanceMapper",
    "PERFORMANCE_PROFILES",
    "CaptionSegment",
    "CaptionTimeline",
    "PlaybackEvent",
    "PlaybackState",
    "SpeechFailure",
]