# Copyright 2026 999 Project (V4-B). Apache-2.0.
"""Sentence chunker + semantic cue mapper unit tests."""

from __future__ import annotations

from main_logic.voice_output.semantic_cue import (
    CueKind,
    CueToPerformanceMapper,
    NEUTRAL_PROFILE,
    SEMANTIC_CUES,
)
from main_logic.voice_output.sentencizer import SafeSentenceChunker

import pytest

pytestmark = pytest.mark.unit


# ── sentence chunker ─────────────────────────────────────────────────────────

def test_splits_cjk_sentences_and_strips_control_chars():
    chunker = SafeSentenceChunker()
    out = chunker.chunk("s1", "你好，世界。\u200b这是一句测试！\x00\t没问题？")
    assert "你好，世界。" in out.sentences
    assert "这是一句测试！" in out.sentences
    assert all("\u200b" not in s and "\x00" not in s for s in out.sentences)


def test_punct_only_input_falls_back_neutrally():
    chunker = SafeSentenceChunker()
    out = chunker.chunk("s1", "。。。 ？？\n")
    assert out.sentences  # fallback sentence, captions never vanish
    assert out.sentences[0].startswith("抱歉")


def test_short_fragments_merge_and_overflow_splits():
    chunker = SafeSentenceChunker(max_chars=20)
    out = chunker.chunk("s1", "一二三四五六七八九十")
    # 10 chars maps to one merged sentence under the 20 cap
    assert len(out.sentences) == 1
    long = "呀" * 55
    out2 = chunker.chunk("s1", long)
    assert all(len(s) <= 28 for s in out2.sentences)  # merged frame cap


def test_chunker_never_touches_when_caption_advances():
    """Chunker must not contain timers — captions are driven purely by the
    playback cursor on the Godot side."""
    import inspect

    src = inspect.getsource(SafeSentenceChunker)
    assert "sleep" not in src
    assert "time" not in src
    assert "played_samples" not in src


# ── semantic cue mapper ──────────────────────────────────────────────────────

def test_whitelist_contains_expected_cues():
    assert SEMANTIC_CUES == {"focus_enter", "attention", "listen_start", "listen_loop", "listen_end"}


def _mapper():
    return CueToPerformanceMapper()


def test_cue_resolves_to_asset_free_action():
    m = _mapper()
    kind, action = m.resolve("focus_enter")
    assert kind is CueKind.SEMANTIC
    assert action is not None
    assert action.profile_id == NEUTRAL_PROFILE.profile_id
    assert action.intent == "on_focus"


def test_cue_rejects_model_specific_performance_params():
    m = _mapper()
    kind, action = m.resolve("listen_start", {"motion": "dance.wav"})
    assert kind is CueKind.DIRTY
    assert action is None
    kind, action = m.resolve("listen_start", {"frame": 12})
    assert kind is CueKind.DIRTY
    kind, action = m.resolve("listen_start", {"cubism": "parameter.h"})
    assert kind is CueKind.DIRTY
    kind, action = m.resolve("listen_start", {"expression": "scared"})
    assert kind is CueKind.DIRTY


def test_cue_not_in_whitelist_refused():
    m = _mapper()
    kind, action = m.resolve("run_dance")
    assert kind is CueKind.UNKNOWN
    assert action is None
    kind, action = m.resolve("")
    assert kind is CueKind.UNKNOWN


def test_mapper_is_pure_no_cubism_names_leak():
    """The neutral profile must never name a Live2D file or parameter."""
    text = str(NEUTRAL_PROFILE)
    for bad in (".motion3.json", ".model3.json", "param.", "expression", "frame"):
        assert bad not in text


def test_all_whitelisted_cues_map_to_neutral_profile():
    m = _mapper()
    for cue in SEMANTIC_CUES:
        kind, action = m.resolve(cue)
        assert kind is CueKind.SEMANTIC
        if action is not None:
            assert action.profile_id == "neutral_test"