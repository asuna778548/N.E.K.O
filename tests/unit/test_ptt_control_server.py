"""PttControlService + HTTP/WS wire tests (no real worker / mic needed)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from main_logic.voice_input.mic_broker import MicBroker, PttSide
from main_logic.voice_input.mic_broker.bridge import PttVoiceInputBridge
from main_logic.voice_input.mic_broker.capture import ScriptedCapture
from main_logic.voice_input.mic_broker.control_server import (
    PttControlService,
    build_app,
)
from main_logic.voice_input.mic_broker.funasr_client import FunasrWorkerClient


class FakeWorker:
    """Stands in for the FunASR worker client; records upstream messages."""

    def __init__(self) -> None:
        self.messages: list[tuple] = []
        self.client_metrics = SimpleNamespace(
            begins_sent=0,
            frames_sent=0,
            ends_sent=0,
            cancels_sent=0,
            partials_received=0,
            finals_received=0,
            errors_received=0,
            reconnects=0,
            send_failures=0,
        )

    def begin_turn(self, descriptor, client_session_id) -> None:  # noqa: ANN001
        self.messages.append(("voice.begin", descriptor.voice_turn_id, descriptor.asr_profile_id))

    def queue_frame(self, descriptor, pcm16, monotonic_ns) -> None:  # noqa: ANN001
        del monotonic_ns
        self.messages.append(("audio.frame", descriptor.voice_turn_id, len(pcm16)))

    def seal_turn(self, descriptor, pcm_tail, released_ns) -> None:  # noqa: ANN001
        del pcm_tail, released_ns
        self.messages.append(("voice.end", descriptor.voice_turn_id))

    def cancel_turn(self, voice_turn_id, reason) -> None:  # noqa: ANN001
        self.messages.append(("voice.cancel", voice_turn_id, reason))


class RecordingRoutes:
    def __init__(self) -> None:
        self.conversation_finals: list[dict] = []
        self.action_finals: list[dict] = []
        self.partials: list[tuple[str, str]] = []
        self.cues: list[str] = []
        self.errors: list[tuple[str | None, str]] = []

    async def on_conversation_final(self, envelope: dict) -> None:
        self.conversation_finals.append(envelope)

    async def on_action_final(self, envelope: dict) -> None:
        self.action_finals.append(envelope)

    async def on_partial_display(self, descriptor, text: str) -> None:  # noqa: ANN001
        self.partials.append((descriptor.voice_turn_id, text))

    async def on_cue(self, cue: str, side, voice_turn_id: str) -> None:  # noqa: ANN001
        self.cues.append(cue)

    async def on_error(self, voice_turn_id: str | None, code: str, detail: str) -> None:
        self.errors.append((voice_turn_id, code))


@pytest.fixture
async def control():
    capture = ScriptedCapture()
    routes = RecordingRoutes()
    worker = FakeWorker()
    broker = MicBroker(capture_factory=lambda: capture, sink=None)
    bridge = PttVoiceInputBridge(
        broker=broker,
        client=worker,  # type: ignore[arg-type]
        on_conversation_final=routes.on_conversation_final,
        on_action_final=routes.on_action_final,
        on_partial_display=routes.on_partial_display,
        on_cue=routes.on_cue,
        on_error=routes.on_error,
    )
    bridge.broker.sink = bridge  # type: ignore[assignment]
    service = PttControlService(bridge)
    app = build_app(service)
    client = TestClient(TestServer(app))
    return client, service, routes, worker, capture


async def _start(control):
    client, _service, _routes, _worker, _capture = control
    await client.start_server()
    return client


async def _get_json(client: TestClient, path: str) -> dict:
    response = await client.get(path)
    assert response.status == 200
    return await response.json()


async def test_press_opens_capture_and_begin_is_sent(control) -> None:
    client = await _start(control)
    try:
        response = await client.post("/ptt/press", json={"side": "mouse4"})
        assert response.status == 200
        payload = await response.json()
        assert payload["accepted"] is True
        assert payload["voice_turn_id"]

        health = await _get_json(client, "/health")
        assert health["capture_open"] is True
        assert health["active_side"] == "mouse4"

        metrics = await _get_json(client, "/metrics")
        assert metrics["broker"]["turns_started"] == 1
        assert metrics["broker"]["capture_opens"] == 1
    finally:
        await client.close()


async def test_first_press_wins_over_http(control) -> None:
    client, service, _routes, _worker, _capture = control
    await client.start_server()
    try:
        first = await (await client.post("/ptt/press", json={"side": "mouse4"})).json()
        second = await (await client.post("/ptt/press", json={"side": "mouse5"})).json()
        assert first["accepted"] is True
        assert second["accepted"] is False
        assert service.bridge.broker.active_turn.side is PttSide.CONVERSATION
        assert (await _get_json(client, "/health"))["active_side"] == "mouse4"
    finally:
        await client.close()


async def test_release_seals_and_closes_capture(control) -> None:
    client, service, routes, _worker, capture = control
    await client.start_server()
    try:
        await client.post("/ptt/press", json={"side": "mouse4"})
        capture.emit(b"\x00\x01" * 80, 10)
        release = await (await client.post("/ptt/release", json={"side": "mouse4"})).json()
        assert release["sealed"] is True
        assert (await _get_json(client, "/health"))["capture_open"] is False
        metrics = await _get_json(client, "/metrics")
        assert metrics["broker"]["turns_sealed"] == 1
        assert metrics["broker"]["ring_bytes_flushed"] == 160
        assert service.bridge.broker.active_turn is None
    finally:
        await client.close()


async def test_final_and_partial_stream_to_ws(control) -> None:
    client, service, routes, _worker, _capture = control
    await client.start_server()
    try:
        press = await (await client.post("/ptt/press", json={"side": "mouse4"})).json()
        voice_turn_id = press["voice_turn_id"]

        ws = await client.ws_connect("/events")
        await service.bridge.handle_partial(voice_turn_id, "白蚀，你", 1)
        await service.bridge.handle_final(voice_turn_id, "白蚀，你怎么看", 2)

        received: list[dict] = []
        while len(received) < 2:
            msg = await ws.receive(timeout=5)
            assert msg.type != WSMsgType.ERROR
            received.append(json.loads(msg.data))

        partial = next(m for m in received if m["type"] == "asr.partial")
        final = next(m for m in received if m["type"] == "asr.final")
        assert partial["voice_turn_id"] == voice_turn_id
        assert partial["asr_profile_id"] == "conversation.zh"
        assert final["channel"] == "conversation"
        assert final["input_owner"] == "mouse4"
        assert final["final_transcript"] == "白蚀，你怎么看"
        assert final["asr_profile_id"] == "conversation.zh"
        assert "monotonic_ns" in final
        assert routes.conversation_finals  # original callback still fired
    finally:
        await client.close()


async def test_cancel_via_http_discards_turn(control) -> None:
    client, service, routes, worker, capture = control
    await client.start_server()
    try:
        press = await (await client.post("/ptt/press", json={"side": "mouse5"})).json()
        voice_turn_id = press["voice_turn_id"]
        capture.emit(b"\x00\x01" * 80, 3)
        cancel = await (
            await client.post("/ptt/cancel", json={"voice_turn_id": voice_turn_id, "reason": "user"})
        ).json()
        assert cancel["cancelled"] is True
        assert routes.conversation_finals == []
        assert routes.action_finals == []
        metrics = await _get_json(client, "/metrics")
        assert metrics["broker"]["turns_cancelled"] == 1
        assert metrics["broker"]["frames_captured"] == 1
        assert ("voice.cancel", voice_turn_id, "user") in worker.messages
    finally:
        await client.close()


async def test_unknown_side_is_rejected_with_400(control) -> None:
    client, _service, _routes, _worker, _capture = control
    await client.start_server()
    try:
        bad = await client.post("/ptt/press", json={"side": "left_button"})
        assert bad.status == 400
        release_bad = await client.post("/ptt/release", json={"side": "keyboard"})
        assert release_bad.status == 400
    finally:
        await client.close()


async def test_frames_without_active_turn_are_dropped_idle(control) -> None:
    client, service, _routes, _worker, _capture = control
    await client.start_server()
    try:
        # No PTT key held: the capture stream is closed, so any frame that
        # arrives (stream leak marker) must be dropped, never processed.
        service.bridge.broker.on_capture_frame(b"\x00\x01" * 20, 1)
        assert service.bridge.broker.metrics.frames_dropped_idle == 1
        assert service.bridge.broker.metrics.frames_captured == 0
    finally:
        await client.close()