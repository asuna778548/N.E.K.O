"""PttVoiceInputBridge routing invariants (contract V1 §2/§3/§5, §6)."""

from __future__ import annotations

import pytest

from main_logic.voice_input.mic_broker import MicBroker, PttSide
from main_logic.voice_input.mic_broker.bridge import PttVoiceInputBridge
from main_logic.voice_input.mic_broker.capture import ScriptedCapture
from main_logic.voice_input.mic_broker.funasr_client import FunasrWorkerClient


class NullSink:
    def on_turn_begin(self, descriptor) -> None: ...  # noqa: ANN001

    def on_frame(self, descriptor, pcm16, monotonic_ns) -> None: ...  # noqa: ANN001

    def on_turn_sealed(self, descriptor, pcm_tail, released_ns) -> None: ...  # noqa: ANN001

    def on_turn_cancelled(self, descriptor, reason) -> None: ...  # noqa: ANN001

    def on_capture_error(self, descriptor, detail) -> None: ...  # noqa: ANN001


class FakeWorker:
    """Stands in for FunasrWorkerClient; records protocol messages."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def begin_turn(self, descriptor, client_session_id) -> None:  # noqa: ANN001
        self.messages.append(("voice.begin", descriptor.voice_turn_id, descriptor.asr_profile_id))

    def queue_frame(self, descriptor, pcm16, monotonic_ns) -> None:  # noqa: ANN001
        self.messages.append(("audio.frame", descriptor.voice_turn_id, len(pcm16)))

    def seal_turn(self, descriptor, pcm_tail, released_ns) -> None:  # noqa: ANN001
        self.messages.append(("voice.end", descriptor.voice_turn_id))

    def cancel_turn(self, voice_turn_id, reason) -> None:
        self.messages.append(("voice.cancel", voice_turn_id, reason))


class RecordingRoutes:
    def __init__(self) -> None:
        self.conversation_finals: list[dict] = []
        self.action_finals: list[dict] = []
        self.partials: list[str] = []
        self.cues: list[tuple[str, PttSide, str]] = []
        self.errors: list[tuple[str | None, str]] = []

    async def on_conversation_final(self, envelope: dict) -> None:
        self.conversation_finals.append(envelope)

    async def on_action_final(self, envelope: dict) -> None:
        self.action_finals.append(envelope)

    async def on_partial_display(self, descriptor, text: str) -> None:
        self.partials.append(text)

    async def on_cue(self, cue: str, side: PttSide, voice_turn_id: str) -> None:
        self.cues.append((cue, side, voice_turn_id))

    async def on_error(self, voice_turn_id: str | None, code: str, detail: str) -> None:
        self.errors.append((voice_turn_id, code))


def _bridge() -> tuple[PttVoiceInputBridge, RecordingRoutes, FakeWorker, ScriptedCapture]:
    capture = ScriptedCapture()
    routes = RecordingRoutes()
    worker = FakeWorker()
    broker = MicBroker(capture_factory=lambda: capture, sink=NullSink())
    bridge = PttVoiceInputBridge(
        broker=broker,
        client=worker,  # type: ignore[arg-type]
        on_conversation_final=routes.on_conversation_final,
        on_action_final=routes.on_action_final,
        on_partial_display=routes.on_partial_display,
        on_cue=routes.on_cue,
        on_error=routes.on_error,
        current_space_epoch=42,
        current_focus_character_id="white_eclipse",
    )
    # Production wiring: the bridge is the MicBroker's one and only sink.
    bridge.broker.sink = bridge
    return bridge, routes, worker, capture


def _descriptor(bridge: PttVoiceInputBridge, voice_turn_id: str):
    pending = bridge._pending[voice_turn_id]
    return pending.descriptor


async def test_mouse4_press_emits_interrupt_and_begin_profile() -> None:
    bridge, routes, worker, _ = _bridge()
    assert bridge.handle_press(PttSide.CONVERSATION) is True
    voice_turn_id = next(iter(bridge._pending))
    import asyncio

    await asyncio.sleep(0)  # let cue tasks run
    assert ("voice.begin", voice_turn_id, "conversation.zh") in worker.messages
    assert ("attention", PttSide.CONVERSATION, voice_turn_id) in routes.cues


async def test_mouse5_press_does_not_interrupt_tts_only_ducks() -> None:
    bridge, routes, worker, _ = _bridge()
    assert bridge.handle_press(PttSide.ACTION_SELECT) is True
    voice_turn_id = next(iter(bridge._pending))
    import asyncio

    await asyncio.sleep(0)
    cues = [cue for cue, _side, _tid in routes.cues]
    assert "attention" not in cues
    assert ("voice.begin", voice_turn_id, "action.zh") in worker.messages


async def test_partial_never_reaches_routing() -> None:
    bridge, routes, _worker, _capture = _bridge()
    assert bridge.handle_press(PttSide.CONVERSATION) is True
    voice_turn_id = next(iter(bridge._pending))
    await bridge.handle_partial(voice_turn_id, "白蚀，你怎", 1)
    assert routes.partials == ["白蚀，你怎"]
    assert routes.conversation_finals == []
    assert routes.action_finals == []
    assert bridge.routed_finals == 0


async def test_conversation_final_builds_contract_envelope() -> None:
    bridge, routes, _worker, _capture = _bridge()
    assert bridge.handle_press(PttSide.CONVERSATION, monotonic_ns=1_000) is True
    voice_turn_id = next(iter(bridge._pending))
    await bridge.handle_final(voice_turn_id, "白蚀，你怎么看", 2_000)
    assert len(routes.conversation_finals) == 1
    envelope = routes.conversation_finals[0]
    assert envelope["voice_turn_id"] == voice_turn_id
    assert envelope["channel"] == "conversation"
    assert envelope["input_owner"] == "mouse4"
    assert envelope["pressed_monotonic_ns"] == 1_000
    assert envelope["released_monotonic_ns"] == 2_000
    assert envelope["asr_profile_id"] == "conversation.zh"
    assert envelope["final_transcript"] == "白蚀，你怎么看"
    assert envelope["conversation_focus_before"] == "white_eclipse"
    assert envelope["space_epoch"] == 42
    assert bridge.client_session_id


async def test_action_final_routes_to_selector_path_only() -> None:
    bridge, routes, _worker, _capture = _bridge()
    assert bridge.handle_press(PttSide.ACTION_SELECT) is True
    voice_turn_id = next(iter(bridge._pending))
    await bridge.handle_final(voice_turn_id, "砸烂它", 500)
    assert len(routes.action_finals) == 1
    assert routes.action_finals[0]["channel"] == "action_select"
    assert routes.conversation_finals == []


async def test_final_with_old_space_epoch_is_dropped() -> None:
    bridge, routes, _worker, _capture = _bridge()
    assert bridge.handle_press(PttSide.CONVERSATION) is True
    voice_turn_id = next(iter(bridge._pending))
    bridge.current_space_epoch = 43  # world moved on while the mic was held
    await bridge.handle_final(voice_turn_id, "白蚀", 600)
    assert routes.conversation_finals == []
    assert bridge.dropped_stale_finals == 1
    assert routes.errors and routes.errors[0][1] == "VOICE_TURN_STALE"


async def test_final_after_deadline_is_dropped() -> None:
    bridge, routes, _worker, _capture = _bridge()
    bridge.turn_deadline_s = 60.0
    assert bridge.handle_press(PttSide.CONVERSATION, monotonic_ns=0) is True
    voice_turn_id = next(iter(bridge._pending))
    await bridge.handle_final(voice_turn_id, "白蚀", 61 * 1_000_000_000)
    assert routes.conversation_finals == []
    assert bridge.dropped_stale_finals == 1


async def test_cancel_stops_final_from_being_routed() -> None:
    bridge, routes, worker, _capture = _bridge()
    assert bridge.handle_press(PttSide.CONVERSATION) is True
    voice_turn_id = next(iter(bridge._pending))
    assert bridge.handle_cancel(voice_turn_id, "key_released_early") is True
    await bridge.handle_final(voice_turn_id, "白蚀", 900)
    assert routes.conversation_finals == []
    assert ("voice.cancel", voice_turn_id, "key_released_early") in worker.messages


async def test_release_before_press_of_other_side_is_noop() -> None:
    bridge, routes, _worker, _capture = _bridge()
    assert bridge.handle_release(PttSide.ACTION_SELECT) is False
    assert routes.cues == []
