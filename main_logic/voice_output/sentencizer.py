# Copyright 2026 999 Project (V4-B). Apache-2.0.
"""Safety filter + sentence chunker for the output chain.

Frozen chain: ``text delta → 安全过滤 → 句子分块 → VoxCPM2 speech-worker``.

Rules:
- The chunker emits displayable, TTS-safe sentences. Control/zero-width
  characters, markdown artifacts and stray XML-ish tags are scrubbed BEFORE
  any model text reaches the worker.
- Caption timelines are assembled from ACTUAL consumed PCM boundaries by the
  output controller, never by "estimated model-text duration" — this module
  only ever decides WHAT to speak, never WHEN captions advance.
- Punct-only / whitespace-only chunks are dropped (including VoxCPM2's
  tendency to voice lingering punctuation). Failure to produce a single
  chunk yields a neutral fallback sentence so captions never silently vanish.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# Sentence boundaries, CJK-aware. "……" is one boundary; "。" inside quotes is
# kept because 白蚀-style long thoughts usually span several clauses.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?…])|(?<=[；;])\s*|(?<=[\n\v\f\r])")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\uFEFF]")
_TAG_RE = re.compile(r"<[^>]{0,40}>")  # survives only real tags; shown text is scrubbed
_WS_COLLAPSE_RE = re.compile(r"[ \t]+")
_STRIP_WS = re.compile(r"^\s+|\s+$")

MAX_SENTENCE_CHARS = 120  # conservative cap; VoxCPM2 handles far more but a
# single sentence over this risks latency spikes and late barge-in.


@dataclass(frozen=True, slots=True)
class ChunkedUtterance:
    speech_id: str
    source_text: str
    sentences: List[str]
    filtered_out: List[str] = field(default_factory=list)


class SafeSentenceChunker:
    """Deterministic sentence splitter + scrubbing for one utterance."""

    def __init__(
        self,
        *,
        max_chars: int = MAX_SENTENCE_CHARS,
        fallback_sentence: str = "抱歉，这句话我没能听懂。",
    ) -> None:
        self.max_chars = int(max_chars)
        self.fallback_sentence = fallback_sentence

    def chunk(self, speech_id: str, text: str) -> ChunkedUtterance:
        clean = self._scrub(text)
        if not clean:
            return ChunkedUtterance(speech_id, text, [self.fallback_sentence], [])
        raw_parts = _SENTENCE_SPLIT_RE.split(clean)
        sentences: List[str] = []
        filtered: List[str] = []

        for part in raw_parts:
            part = _STRIP_WS.sub("", part)
            part = _WS_COLLAPSE_RE.sub(" ", part)
            if not part:
                continue
            stripped = part.rstrip("。，、；：！？!?… ~")
            if not stripped:
                filtered.append(part)
                continue
            sentences.append(part)

        # Respect natural sentence boundaries (caption granularity), only
        # hard-splitting overlong sentences so worker latency stays bounded.
        final: List[str] = []
        for sentence in sentences:
            final.extend(self._hard_split(sentence) if len(sentence) > self.max_chars else [sentence])
        if not final:
            return ChunkedUtterance(speech_id, text, [self.fallback_sentence], filtered)
        return ChunkedUtterance(speech_id, text, final, filtered)

    def _scrub(self, text: str) -> str:
        if text is None:
            return ""
        text = _CONTROL_RE.sub("", text)
        text = _ZERO_WIDTH_RE.sub("", text)
        text = _TAG_RE.sub("", text)
        text = _WS_COLLAPSE_RE.sub(" ", text).strip()
        return text

    def _hard_split(self, sentence: str) -> List[str]:
        parts: List[str] = []
        while len(sentence) > self.max_chars:
            cut = self.max_chars
            if 0 <= cut < len(sentence):
                parts.append(sentence[:cut])
                sentence = sentence[cut:]
            else:
                break
        if sentence:
            parts.append(sentence)
        return parts