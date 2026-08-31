# Copyright 2026 999 Project (V4-B). Apache-2.0.
"""Semantic cue → performance-profile mapping.

Frozen boundary (ADR a-world-resident-will §10):
  `focus_enter / attention / listen_start / listen_loop / listen_end` are the
  ONLY semantic cues the input side may emit for character expression. The
  concrete Live2D motion is decided by the character's performance PROFILE,
  never by the model. Cues arriving here must NOT carry Cubism parameters,
  motion file names, or frame numbers — this mapper refuses and normalizes
  any such payload down to the cue token only.

This wave ships a programmatic fallback profile (``neutral``): instead of a
Live2D name it carries an integer "drive level" plus a set of semantic motion
tokens that a Live2D adapter (later wave, with real assets) can bind. The
mapper is pure: it emits a :class:`ProfileAction` with no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Optional

# The whitelist of semantic performance cues (ADR §10). Input side must emit
# exactly these tokens; anything else is refused.
SEMANTIC_CUES: FrozenSet[str] = frozenset(
    {"focus_enter", "attention", "listen_start", "listen_loop", "listen_end"}
)

# Tokens a cue maps to a neutral, asset-free performance "intent".  These are
# NOT runtime motion names — they are contract-level intents the profile maps.
_INTENT_BY_CUE: Dict[str, str] = {
    "focus_enter": "on_focus",
    "attention": "awaiting_turn",  # Mouse4 press → interrupt → attention
    "listen_start": "begin_attention",
    "listen_loop": "sustained_attention",
    "listen_end": "release_attention",
}


class CueKind(str, Enum):
    SEMANTIC = "semantic"  # whitelisted semantic cue
    UNKNOWN = "unknown"    # refused: not in the whitelist
    DIRTY = "dirty"        # refused: carried model-specific params


@dataclass(frozen=True, slots=True)
class ProfileAction:
    """One resolved, asset-free action for a performance profile."""

    cue: str
    intent: str
    drive_level: int  # neutral 0..3 intensity, profile/asset agnostic
    profile_id: str
    duration_hint_ms: int  # only a HINT; never authored by the model

    def to_dict(self) -> dict:
        return {
            "cue": self.cue,
            "intent": self.intent,
            "drive_level": self.drive_level,
            "profile_id": self.profile_id,
            "duration_hint_ms": self.duration_hint_ms,
        }


@dataclass(frozen=True, slots=True)
class PerformanceProfile:
    profile_id: str
    description: str
    actions: Dict[str, ProfileAction] = field(default_factory=dict)


def _neutral_action(cue: str, intent: str, level: int, hint_ms: int) -> ProfileAction:
    return ProfileAction(
        cue=cue,
        intent=intent,
        drive_level=level,
        profile_id="neutral_test",
        duration_hint_ms=hint_ms,
    )


# The single procedural fallback profile (this wave has no formal assets).
NEUTRAL_PROFILE = PerformanceProfile(
    profile_id="neutral_test",
    description="Programmatic fallback character performance; carries only an "
    "asset-free drive level plus a semantic intent token. No model-authored "
    "motion directives are accepted by this tier.",
    actions={
        "focus_enter": _neutral_action("focus_enter", "on_focus", 1, 1800),
        "attention": _neutral_action("attention", "awaiting_turn", 2, 300),
        "listen_start": _neutral_action("listen_start", "begin_attention", 2, 500),
        "listen_loop": _neutral_action("listen_loop", "sustained_attention", 1, 2500),
        "listen_end": _neutral_action("listen_end", "release_attention", 0, 350),
    },
)

PERFORMANCE_PROFILES: Dict[str, PerformanceProfile] = {"neutral_test": NEUTRAL_PROFILE}


class CueToPerformanceMapper:
    """Maps a semantic cue token to a profile action; refuses model-tainted
    payloads. Pure — the caller applies the action to Live2D/fallback."""

    def __init__(self, profiles: Optional[Dict[str, PerformanceProfile]] = None) -> None:
        self.profiles = profiles if profiles is not None else dict(PERFORMANCE_PROFILES)

    MAP_CUE = _INTENT_BY_CUE

    def resolve(self, cue: str, payload: Optional[dict] = None) -> tuple[CueKind, Optional[ProfileAction]]:
        """Return (kind, action). Refused cues yield (UNKNOWN|DIRTY, None)."""
        if not isinstance(cue, str) or cue not in SEMANTIC_CUES:
            return CueKind.UNKNOWN, None
        if payload:
            # Refuse any model-authored motion command: Cubism params,
            # filenames, frame numbers.
            probe = {k.lower(): v for k, v in payload.items()}
            if any(k in probe for k in ("motion", "file", "param", "frame", "cubism", "expression")):
                return CueKind.DIRTY, None
        profile = self.profiles.get("neutral_test")
        if profile is None or cue not in profile.actions:
            return CueKind.SEMANTIC, None
        return CueKind.SEMANTIC, profile.actions[cue]