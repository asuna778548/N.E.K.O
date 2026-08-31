"""FunASR worker client protocol behavior (proposal voice-streaming v1)."""

from __future__ import annotations

import json

import pytest

from main_logic.voice_input.mic_broker import MicBroker, PttSide
from main_logic.voice_input.mic_broker.contracts import MicTurnDescriptor
from main_logic.voice_input.mic_broker.funasr_client import FunasrWorkerClient


class FakeWsTransport:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.incoming: list[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        if not self.incoming:
            await __import__("asyncio").sleep(0.01)
            return json.dumps({"type": "noop"})
        return self.incoming.pop(0)

    async def close(self) -> None:
        self.closed = True


def _descriptor(voice_turn_id: str = "t1", side: PttSide = PttSide.CONVERSATION):
    return MicTurnDescriptor(
        voice_turn_id=voice_turn_id,
        side=side,
        channel="conversation" if side is PttSide.CONVERSATION else "action_select",
        input_owner="mouse4" if side is PttSide.CONVERSATION else "mouse5",
        asr_profile_id="conversation.zh" if side is PttSide.CONVERSATION else "action.zh",
        pressed_monotonic_ns=123,
    )


class RecordingCallbacks:
    def __init__(self) -> None:
        self.partials: list[tuple[str, str, int]] = []
        self.finals: list[tuple[str, str, int]] = []
        self.errors: list[tuple[str | None, str, str]] = []

    async def on_partial(self, voice_turn_id: str, text: str, ns: int) -> None:
        self.partials.append((voice_turn_id, text, ns))

    async def on_final(self, voice_turn_id: str, text: str, ns: int) -> None:
        self.finals.append((voice_turn_id, text, ns))

    async def on_error(self, voice_turn_id: str | None, code: str, detail: str) -> None:
        self.errors.append((voice_turn_id, code, detail))


def _client(transport: FakeWsTransport) -> tuple[FunasrWorkerClient, RecordingCallbacks]:
    callbacks = RecordingCallbacks()
    client = FunasrWorkerClient(
        endpoint="ws://test",
        on_partial=callbacks.on_partial,
        on_final=callbacks.on_final,
        on_error=callbacks.on_error,
    )
    client._ws = transport  # inject transport without real connect
    client._outgoing = __import__("asyncio").Queue()
    client._loop = __import__("asyncio").get_running_loop()
    client._loop_thread_id = __import__("threading").get_ident()
    client._tasks.append(__import__("asyncio").create_task(client._send_pump()))
    client._tasks.append(__import__("asyncio").create_task(client._recv_pump()))
    return client, callbacks


async def test_upstream_messages_carry_contract_fields() -> None:
    transport = FakeWsTransport()
    client, _ = _client(transport)
    descriptor = _descriptor()
    client.begin_turn(descriptor, "session-1")
    client.queue_frame(descriptor, b"\x01\x02", 456)
    # The ring tail is the full press-to-release capture: the already
    # live-streamed prefix (2 bytes) plus 2 bytes that live streaming missed.
    # seal_turn must forward only the not-yet-streamed remainder.
    client.seal_turn(descriptor, b"\x01\x02\x03\x04", 789)
    await client.close()

    types = [json.loads(raw)["type"] for raw in transport.sent]
    assert types == ["voice.begin", "audio.frame", "audio.frame", "voice.end"]
    begin = json.loads(transport.sent[0])
    assert begin["channel"] == "conversation"
    assert begin["input_owner"] == "mouse4"
    assert begin["asr_profile_id"] == "conversation.zh"
    assert begin["client_session_id"] == "session-1"
    frame = json.loads(transport.sent[1])
    assert frame["voice_turn_id"] == "t1"
    assert frame["pcm16_base64"] == "AQI="  # the live-streamed frame
    tail_frame = json.loads(transport.sent[2])
    assert tail_frame["pcm16_base64"] == "AwQ="  # only the missed remainder
    # seal_turn flushed the tail remainder as a frame before voice.end
    assert json.loads(transport.sent[3])["released_monotonic_ns"] == 789


async def test_downstream_partial_and_final_dispatch() -> None:
    transport = FakeWsTransport()
    transport.incoming = [
        json.dumps(
            {
                "type": "asr.partial",
                "voice_turn_id": "t1",
                "asr_profile_id": "conversation.zh",
                "text": "白蚀",
                "final": False,
                "monotonic_ns": 5,
            }
        ),
        json.dumps(
            {
                "type": "asr.final",
                "voice_turn_id": "t1",
                "asr_profile_id": "conversation.zh",
                "text": "白蚀，你怎么看",
                "final": True,
                "monotonic_ns": 6,
            }
        ),
    ]
    client, callbacks = _client(transport)
    for _ in range(50):
        if len(callbacks.finals) == 1:
            break
        await __import__("asyncio").sleep(0.01)
    await client.close()
    assert callbacks.partials == [("t1", "白蚀", 5)]
    assert callbacks.finals == [("t1", "白蚀，你怎么看", 6)]


async def test_worker_error_is_surfaced() -> None:
    transport = FakeWsTransport()
    transport.incoming = [
        json.dumps(
            {
                "type": "voice.error",
                "voice_turn_id": None,
                "code": "FUNASR_MODEL_LOAD_FAILED",
                "detail": "offline model missing",
            }
        )
    ]
    client, callbacks = _client(transport)
    for _ in range(50):
        if callbacks.errors:
            break
        await __import__("asyncio").sleep(0.01)
    await client.close()
    assert callbacks.errors == [(None, "FUNASR_MODEL_LOAD_FAILED", "offline model missing")]


async def test_partial_carrying_final_flag_is_dropped_never_routed() -> None:
    """Protocol violation on the wire must be dropped, not upgraded to a final."""
    transport = FakeWsTransport()
    transport.incoming = [
        json.dumps(
            {
                "type": "asr.partial",
                "voice_turn_id": "t1",
                "asr_profile_id": "conversation.zh",
                "text": "白蚀",
                "final": True,
                "monotonic_ns": 5,
            }
        )
    ]
    client, callbacks = _client(transport)
    for _ in range(50):
        if client.client_metrics.malformed_downstream == 1:
            break
        await __import__("asyncio").sleep(0.01)
    await client.close()
    assert callbacks.partials == []
    assert callbacks.finals == []
    assert client.client_metrics.malformed_downstream == 1


async def test_cancel_message_shape() -> None:
    transport = FakeWsTransport()
    client, _ = _client(transport)
    client.cancel_turn("t1", "ptt_cancelled")
    await client.close()
    cancel = json.loads(transport.sent[0])
    assert cancel["type"] == "voice.cancel"
    assert cancel["voice_turn_id"] == "t1"
    assert cancel["reason"] == "ptt_cancelled"
