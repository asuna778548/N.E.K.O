"""PTT-to-routing bridge: InputAuthority events to MicBroker to finals.

Responsibilities (contract V1 §1/§2, ADR 2026-08-29 §10):

- Build the ``VoiceTurnEnvelope`` for every final, carrying
  ``voice_turn_id``/``channel``/``input_owner``/timestamps/``asr_profile_id``
  plus ``space_epoch`` and ``conversation_focus_before``.
- Drop stale turns: finals arriving after the turn deadline, or stamped with
  an old ``space_epoch``, are discarded before any routing.
- Partials are display-only: they reach ``on_partial`` (HUD/subtitles) and
  nothing else. No focus change, no action selection, no transactions.
- On Mouse4 press, emit the TTS interrupt cue (V4-B provides the cancel
  endpoint; here it is just a signal). On Mouse5 press, emit duck.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .broker import MicBroker, VoiceFrameSink
from .contracts import MicTurnDescriptor, PttSide
from .funasr_client import FunasrWorkerClient

ConversationFinalCallback = Callable[[dict], Awaitable[None]]
ActionFinalCallback = Callable[[dict], Awaitable[None]]
PartialCallback = Callable[[MicTurnDescriptor, str], Awaitable[None]]
CueCallback = Callable[[str, PttSide, str], Awaitable[None]]  # cue, side, voice_turn_id
ErrorCallback = Callable[[str | None, str, str], Awaitable[None]]

DEFAULT_TURN_DEADLINE_S = 60.0


@dataclass(slots=True)
class _PendingTurn:
    descriptor: MicTurnDescriptor
    space_epoch: int
    focus_before: str
    pressed_monotonic_ns: int


@dataclass(slots=True)
class PttVoiceInputBridge:
    """Sync MicBroker sink + async final router. Injected callbacks keep the
    Godot-facing transport (and V4-B output cues) mockable."""

    broker: MicBroker
    client: FunasrWorkerClient
    on_conversation_final: ConversationFinalCallback
    on_action_final: ActionFinalCallback
    on_partial_display: PartialCallback
    on_cue: CueCallback
    on_error: ErrorCallback
    current_space_epoch: int = 0
    current_focus_character_id: str = ""
    turn_deadline_s: float = DEFAULT_TURN_DEADLINE_S
    now_ns: Callable[[], int] = time.monotonic_ns
    client_session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _pending: dict[str, _PendingTurn] = field(default_factory=dict, init=False)
    routed_finals: int = 0
    dropped_stale_finals: int = 0

    # -- InputAuthority entry points -----------------------------------------
    def handle_press(self, side: PttSide, monotonic_ns: int | None = None) -> bool:
        pressed_ns = monotonic_ns if monotonic_ns is not None else self.now_ns()
        voice_turn_id = str(uuid.uuid4())
        accepted = self.broker.begin_turn(side, voice_turn_id, pressed_ns)
        if not accepted:
            return False
        descriptor = self.broker.active_turn
        assert descriptor is not None
        self._pending[voice_turn_id] = _PendingTurn(
            descriptor=descriptor,
            space_epoch=self.current_space_epoch,
            focus_before=self.current_focus_character_id,
            pressed_monotonic_ns=pressed_ns,
        )
        # voice.begin is sent via the MicBroker sink path (on_turn_begin), so
        # there is exactly one begin per accepted turn.
        # Contract §1: Mouse4 interrupts current character TTS; Mouse5 only
        # ducks/pauses it. Actual cancellation lives in the output stack.
        if side is PttSide.CONVERSATION:
            self._fire_cue("attention", descriptor, tts_interrupt=True)
        else:
            self._fire_cue("listen_start", descriptor, tts_interrupt=False)
        return True

    def handle_release(self, side: PttSide, monotonic_ns: int | None = None) -> bool:
        released_ns = monotonic_ns if monotonic_ns is not None else self.now_ns()
        pending = self._active_pending_for(side)
        if pending is None:
            return False
        # voice.end is sent via the MicBroker sink path (on_turn_sealed).
        sealed = self.broker.end_turn(pending.descriptor.voice_turn_id, released_ns)
        if not sealed:
            return False
        self._fire_listen_end(pending.descriptor)
        return True

    def handle_cancel(self, voice_turn_id: str, reason: str) -> bool:
        pending = self._pending.pop(voice_turn_id, None)
        if pending is None:
            return False
        cancelled = self.broker.cancel_turn(voice_turn_id, reason)
        self.client.cancel_turn(voice_turn_id, reason)
        return cancelled

    # -- MicBroker sink -------------------------------------------------------
    def on_turn_begin(self, descriptor: MicTurnDescriptor) -> None:
        self.client.begin_turn(descriptor, self.client_session_id)

    def on_frame(
        self, descriptor: MicTurnDescriptor, pcm16: bytes, monotonic_ns: int
    ) -> None:
        self.client.queue_frame(descriptor, pcm16, monotonic_ns)

    def on_turn_sealed(
        self,
        descriptor: MicTurnDescriptor,
        pcm_tail: bytes,
        released_monotonic_ns: int,
    ) -> None:
        self.client.seal_turn(descriptor, pcm_tail, released_monotonic_ns)

    def on_turn_cancelled(self, descriptor: MicTurnDescriptor, reason: str) -> None:
        self.client.cancel_turn(descriptor.voice_turn_id, reason)

    def on_capture_error(self, descriptor: MicTurnDescriptor | None, detail: str) -> None:
        voice_turn_id = descriptor.voice_turn_id if descriptor else None
        self._schedule_error(voice_turn_id, "CAPTURE_ERROR", detail)

    # -- FunASR worker downstream ----------------------------------------------
    async def handle_partial(
        self, voice_turn_id: str, text: str, monotonic_ns: int
    ) -> None:
        del monotonic_ns
        pending = self._pending.get(voice_turn_id)
        if pending is None:
            return  # stale/cancelled partial: display nothing, route nothing
        await self.on_partial_display(pending.descriptor, text)

    async def handle_final(
        self, voice_turn_id: str, text: str, monotonic_ns: int
    ) -> None:
        pending = self._pending.pop(voice_turn_id, None)
        if pending is None:
            return
        envelope = self._build_envelope(pending, text, monotonic_ns)
        if self._is_stale(pending, envelope):
            self.dropped_stale_finals += 1
            await self.on_error(
                voice_turn_id, "VOICE_TURN_STALE", "final arrived after deadline or old epoch"
            )
            return
        self.routed_finals += 1
        if pending.descriptor.side is PttSide.CONVERSATION:
            await self.on_conversation_final(envelope)
        else:
            await self.on_action_final(envelope)

    async def handle_worker_error(
        self, voice_turn_id: str | None, code: str, detail: str
    ) -> None:
        if voice_turn_id is not None:
            self._pending.pop(voice_turn_id, None)
        await self.on_error(voice_turn_id, code, detail)

    # -- helpers ---------------------------------------------------------------
    def _build_envelope(
        self, pending: _PendingTurn, final_text: str, monotonic_ns: int
    ) -> dict:
        descriptor = pending.descriptor
        return {
            "voice_turn_id": descriptor.voice_turn_id,
            "channel": descriptor.channel,
            "input_owner": descriptor.input_owner,
            "pressed_monotonic_ns": pending.pressed_monotonic_ns,
            "released_monotonic_ns": monotonic_ns,
            "asr_profile_id": descriptor.asr_profile_id,
            "final_transcript": final_text,
            "conversation_focus_before": pending.focus_before,
            "client_session_id": self.client_session_id,
            "space_epoch": pending.space_epoch,
        }

    def _is_stale(self, pending: _PendingTurn, envelope: dict) -> bool:
        released_ns = int(envelope["released_monotonic_ns"])
        deadline_ns = pending.pressed_monotonic_ns + int(
            self.turn_deadline_s * 1_000_000_000
        )
        if released_ns > deadline_ns:
            return True
        if pending.space_epoch != self.current_space_epoch:
            return True
        return False

    def _active_pending_for(self, side: PttSide) -> _PendingTurn | None:
        for pending in self._pending.values():
            if pending.descriptor.side is side:
                return pending
        return None

    def _fire_cue(
        self, cue: str, descriptor: MicTurnDescriptor, *, tts_interrupt: bool
    ) -> None:
        async def _run() -> None:
            if tts_interrupt:
                await self.on_cue("attention", descriptor.side, descriptor.voice_turn_id)
            await self.on_cue(cue, descriptor.side, descriptor.voice_turn_id)

        self._spawn(_run())

    def _fire_listen_end(self, descriptor: MicTurnDescriptor) -> None:
        self._spawn(
            self.on_cue("listen_end", descriptor.side, descriptor.voice_turn_id)
        )

    def _schedule_error(self, voice_turn_id: str | None, code: str, detail: str) -> None:
        self._spawn(self.handle_worker_error(voice_turn_id, code, detail))

    def _spawn(self, coroutine: Awaitable[None]) -> None:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(coroutine)
