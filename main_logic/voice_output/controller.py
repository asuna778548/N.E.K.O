# Copyright 2026 999 Project (V4-B). Apache-2.0.
"""Speech output controller: the single orchestration of the frozen chain.

    text delta
      → SafeSentenceChunker (安全过滤+句子分块)
      → VoxCpm2SpeechClient (one sentence at a time, same speech_id)
      → UniqueAudioSink (played_samples cursor + RMS/peak lipsync)
      → PlaybackEvent + CaptionTimeline → Godot speech_clock_bridge

Cancel chain: one token (``speech_id``). ``cancel(speech_id)`` walks, in
order:
  1. stop feeding new sentences (controller drops the pending sentence loop)
  2. worker-level cancel         (VoxCPM2 engine stops the model stream)
  3. PCM queue drain             (sink drops queued chunks, emits CANCELLED)
  4. lip-sync                    (RMS/peak feeding stops with the sink)
  5. captions                    (bridge stops advancing on the CANCELLED frame)
Everything shares the single speech_id token — no second cancel path exists.

Failure degradation: on worker error the caption timeline is retained (visible
until the FAILED frame), a generated neutral beep is played through the SAME
sink, and the failure is surfaced as a trackable :class:`SpeechFailure`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .contracts import CaptionSegment, CaptionTimeline, PlaybackEvent, SpeechFailure
from .sentencizer import SafeSentenceChunker
from .sink import UniqueAudioSink
from .voxcpm2_client import VoxCpm2SpeechClient, _CancelledStream, _ErroredStream

BEEP_SAMPLE_RATE = 48000


def make_neutral_beep_pcm(sample_rate: int = BEEP_SAMPLE_RATE, duration_s: float = 0.12, volume: float = 0.28) -> bytes:
    """Programmatically generated neutral cue tone (no model, no assets)."""
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    fade = np.minimum(1.0, np.minimum(t / 0.01, (duration_s - t) / 0.01))
    tone = np.sin(2 * np.pi * 660.0 * t) + 0.5 * np.sin(2 * np.pi * 990.0 * t)
    pcm = np.clip(volume * fade * tone, -1.0, 1.0) * 32767
    return pcm.astype(np.int16).tobytes()


@dataclass(slots=True)
class OutputControllerStats:
    speeches: int = 0
    sentences: int = 0
    cancelled: int = 0
    failed: int = 0
    pcm_samples: int = 0
    errors: List[SpeechFailure] = field(default_factory=list)


class SpeechOutputController:
    """Orchestrates one utterance through the frozen chain (see module doc)."""

    def __init__(
        self,
        sink: UniqueAudioSink,
        client: VoxCpm2SpeechClient,
        *,
        chunker: Optional[SafeSentenceChunker] = None,
        sample_rate: int = BEEP_SAMPLE_RATE,
    ) -> None:
        self.sink = sink
        self.client = client
        self.chunker = chunker or SafeSentenceChunker()
        self.sample_rate = sample_rate
        self.stats = OutputControllerStats()
        self._active: set = set()

    @property
    def active(self) -> List[str]:
        return sorted(self._active)

    async def speak(
        self,
        speech_id: str,
        text: str,
        *,
        on_timeline: Optional[Callable[[CaptionTimeline], None]] = None,
        on_failure: Optional[Callable[[SpeechFailure], None]] = None,
        on_pcm: Optional[Callable[[bytes], None]] = None,
    ) -> None:
        """Synthesize one utterance and feed PCM into the unique sink.

        Playback/failure events always flow through the sink's own ``on_event``
        / the ``on_failure`` hook; this method adds the caption-timeline and
        raw-PCM hooks only.
        """
        self.stats.speeches += 1
        if speech_id in self._active:
            await asyncio.to_thread(self.cancel, speech_id)
        utterance = self.chunker.chunk(speech_id, text)

        self.sink.begin(speech_id, self.sample_rate)

        self._active.add(speech_id)
        try:
            segments: List[CaptionSegment] = []
            for sentence in utterance.sentences:
                if speech_id not in self._active:
                    return
                self.stats.sentences += 1
                # Segment start = total audio already collected for prior
                # sentence(s) — a stable coordinate the cursor will reach.
                segment_start = self.sink.total_emitted(speech_id)
                async for chunk in self.client.synthesize_stream(speech_id, sentence):
                    if speech_id not in self._active:
                        return
                    self.sink.feed(speech_id, chunk)
                    self.stats.pcm_samples += len(chunk) // 2
                    if on_pcm is not None:
                        on_pcm(chunk)
                end_sample = self.sink.total_emitted(speech_id)
                segments.append(
                    CaptionSegment(
                        start_sample=segment_start,
                        end_sample=max(end_sample, segment_start + 1),
                        text=sentence,
                    )
                )
                if on_timeline is not None:
                    on_timeline(
                        CaptionTimeline(
                            speech_id=speech_id,
                            sample_rate=self.sample_rate,
                            segments=list(segments),
                        )
                    )
            self._active.discard(speech_id)
            self.sink.finalize(speech_id)
        except _CancelledStream:
            self.stats.cancelled += 1
            self.sink.cancel(speech_id)
        except _ErroredStream as exc:
            self._handle_failure(speech_id, exc.failure, on_failure)
        except asyncio.CancelledError:
            self.sink.cancel(speech_id)
            raise
        finally:
            self._active.discard(speech_id)

    async def acancel(self, speech_id: str) -> None:
        """Async barge-in cancel chain (see module docstring)."""
        self._active.discard(speech_id)
        try:
            await self.client.cancel(speech_id)
        except Exception:
            pass
        self.sink.cancel(speech_id)
        self.stats.cancelled += 1

    def cancel(self, speech_id: str) -> None:
        """Sync barge-in cancel for callers without a running loop; the worker
        cancel is fired in the background through the injected client."""
        self._active.discard(speech_id)
        try:
            maybe_awaitable = self.client.cancel(speech_id)
            if hasattr(maybe_awaitable, "__await__") and not asyncio.iscoroutinefunction(getattr(maybe_awaitable, "__call__", None)):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None and not loop.is_closed():
                    loop.create_task(maybe_awaitable)
        except Exception:
            pass
        self.sink.cancel(speech_id)
        self.stats.cancelled += 1

    def _handle_failure(
        self,
        speech_id: str,
        failure: SpeechFailure,
        on_failure: Optional[Callable[[SpeechFailure], None]],
    ) -> None:
        """TTS failure degradation: keep captions, play neutral beep, track."""
        self.stats.failed += 1
        self.stats.errors.append(failure)
        self.sink.fail(speech_id, failure.code, failure.message)
        # Neutral beep: played through the SAME sink, never a second path.
        try:
            beep_id = f"beep-{speech_id}"
            self.sink.begin(beep_id, self.sample_rate)
            self.sink.feed(beep_id, make_neutral_beep_pcm(self.sample_rate))
            self.sink.finalize(beep_id)
        except Exception:
            pass
        if on_failure is not None:
            on_failure(failure)