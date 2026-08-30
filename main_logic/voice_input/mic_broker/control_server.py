"""Local PTT control service: the wire Godot's InputAuthority talks to.

Production chain (frozen, contract V1 §1): Godot InputAuthority detects the
``conversation_ptt`` / ``action_select_ptt`` side buttons, then this local
service turns press/release/cancel into MicBroker turn transitions. Downstream
``asr.partial`` (display-only) and ``asr.final`` (VoiceTurnEnvelope) are pushed
back to Godot over a WebSocket, plus ``voice.error``.

This service owns no microphone and no ASR; it only bridges events. The
MicBroker remains the unique capture owner and the bridge the sole final
router (space_epoch / turn-deadline staleness live in PttVoiceInputBridge).

Endpoint map (proposal protocol v1):

- ``POST /ptt/press``   ``{"side": "mouse4"|"mouse5"}``
- ``POST /ptt/release`` ``{"side": "mouse4"|"mouse5"}``
- ``POST /ptt/cancel``  ``{"voice_turn_id": str, "reason": str}``
- ``GET  /events``       WebSocket; pushes ``asr.partial`` / ``asr.final`` /
                         ``voice.error`` to every connected client, and accepts
                         ``{"type": "voice.cancel", "voice_turn_id": ...}``.
- ``GET  /metrics``      broker + client counters (zero-idle gate reads this).
- ``GET  /health``
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from aiohttp import WSMsgType, web

from .bridge import PttVoiceInputBridge
from .contracts import PttSide
from .controller import WorkerPipeline

logger = logging.getLogger(__name__)

PTT_SIDE_BY_NAME = {"mouse4", "mouse5"}


async def _noop(*_args: Any) -> None:
    return None


@dataclass
class PttControlService:
    """Owns the bridge and fans downstream events out over per-client queues."""

    bridge: PttVoiceInputBridge
    subscribers: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        bridge = self.bridge
        bridge.broker.sink = bridge  # type: ignore[assignment]
        bridge.client.on_partial = bridge.handle_partial  # type: ignore[assignment]
        bridge.client.on_final = bridge.handle_final  # type: ignore[assignment]
        bridge.client.on_error = bridge.handle_worker_error  # type: ignore[assignment]
        self._original_conversation_final = bridge.on_conversation_final or _noop
        self._original_action_final = bridge.on_action_final or _noop
        self._original_partial = bridge.on_partial_display or _noop
        self._original_error = bridge.on_error or _noop
        bridge.on_conversation_final = self._fanout("asr.final", self._original_conversation_final)
        bridge.on_action_final = self._fanout("asr.final", self._original_action_final)
        bridge.on_partial_display = self._fanout("asr.partial", self._original_partial)
        bridge.on_error = self._fanout("voice.error", self._original_error)

    def _fanout(
        self, event_type: str, original: Callable[..., Awaitable[None]]
    ) -> Callable[..., Awaitable[None]]:
        async def _fanout(*args: Any) -> None:
            payload = self._reshape(event_type, args)
            await self.broadcast(payload)
            await original(*args)

        return _fanout

    def _reshape(self, event_type: str, args: tuple[Any, ...]) -> dict[str, Any]:
        monotonic_ns = time.monotonic_ns()
        if event_type == "asr.final":
            envelope = args[0]
            return {"type": event_type, **envelope, "monotonic_ns": monotonic_ns}
        if event_type == "asr.partial":
            descriptor: MicTurnDescriptor = args[0]
            text: str = args[1]
            return {
                "type": event_type,
                "voice_turn_id": descriptor.voice_turn_id,
                "asr_profile_id": descriptor.asr_profile_id,
                "text": text,
                "monotonic_ns": monotonic_ns,
            }
        # voice.error: (voice_turn_id, code, detail)
        return {
            "type": event_type,
            "voice_turn_id": args[0],
            "code": args[1],
            "detail": args[2],
            "monotonic_ns": monotonic_ns,
        }

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1024)
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        try:
            self.subscribers.remove(queue)
        except ValueError:
            pass

    async def broadcast(self, message: dict[str, Any]) -> None:
        dead: list[asyncio.Queue[dict[str, Any]]] = []
        for queue in self.subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                dead.append(queue)  # slow consumer: drop the subscription
        for queue in dead:
            self.unsubscribe(queue)


async def _safe_json_body(request: web.Request) -> dict[str, Any]:
    try:
        return await request.json()
    except Exception:
        return {}


async def handle_press(request: web.Request, service: PttControlService) -> web.Response:
    body = await _safe_json_body(request)
    side_name = str(body.get("side", ""))
    if side_name not in PTT_SIDE_BY_NAME:
        return web.json_response(
            {"ok": False, "error": "UNKNOWN_PTT_SIDE", "side": side_name}, status=400
        )
    accepted = service.bridge.handle_press(PttSide(side_name))
    active = service.bridge.broker.active_turn
    return web.json_response(
        {
            "ok": accepted,
            "accepted": accepted,
            "voice_turn_id": active.voice_turn_id if active is not None else None,
        }
    )


async def handle_release(request: web.Request, service: PttControlService) -> web.Response:
    body = await _safe_json_body(request)
    side_name = str(body.get("side", ""))
    if side_name not in PTT_SIDE_BY_NAME:
        return web.json_response(
            {"ok": False, "error": "UNKNOWN_PTT_SIDE", "side": side_name}, status=400
        )
    sealed = service.bridge.handle_release(PttSide(side_name))
    return web.json_response({"ok": sealed, "sealed": sealed})


async def handle_cancel(request: web.Request, service: PttControlService) -> web.Response:
    body = await _safe_json_body(request)
    voice_turn_id = str(body.get("voice_turn_id", ""))
    reason = str(body.get("reason", "control_cancel"))
    if not voice_turn_id:
        return web.json_response(
            {"ok": False, "error": "MISSING_VOICE_TURN_ID"}, status=400
        )
    cancelled = service.bridge.handle_cancel(voice_turn_id, reason)
    return web.json_response({"ok": cancelled, "cancelled": cancelled})


async def handle_events(request: web.Request, service: PttControlService) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    queue = service.subscribe()
    try:
        forwarder = asyncio.create_task(_drain_to_ws(ws, queue))
        async for msg in ws:
            if msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
            if msg.type == WSMsgType.TEXT:
                # Control-plane messages (e.g. explicit cancel) from Godot.
                try:
                    payload = json.loads(msg.data)
                except (TypeError, ValueError):
                    await ws.send_str(
                        json.dumps(
                            {
                                "type": "voice.error",
                                "code": "MALFORMED_MESSAGE",
                                "detail": "bad json",
                            }
                        )
                    )
                    continue
                if payload.get("type") == "voice.cancel":
                    service.bridge.handle_cancel(
                        str(payload.get("voice_turn_id", "")), "ws_cancel"
                    )
        forwarder.cancel()
        return ws
    finally:
        service.unsubscribe(queue)


async def _drain_to_ws(
    ws: web.WebSocketResponse, queue: asyncio.Queue[dict[str, Any]]
) -> None:
    while True:
        message = await queue.get()
        try:
            await ws.send_str(json.dumps(message, ensure_ascii=False))
        except (ConnectionResetError, RuntimeError):
            return


async def handle_metrics(request: web.Request, service: PttControlService) -> web.Response:
    del request
    broker = service.bridge.broker.metrics
    client = service.bridge.client.client_metrics
    return web.json_response(
        {
            "broker": {
                "capture_opens": broker.capture_opens,
                "capture_closes": broker.capture_closes,
                "capture_errors": broker.capture_errors,
                "device_recoveries": broker.device_recoveries,
                "frames_captured": broker.frames_captured,
                "frames_dropped_idle": broker.frames_dropped_idle,
                "frames_dropped_stale": broker.frames_dropped_stale,
                "turns_started": broker.turns_started,
                "turns_ignored": broker.turns_ignored,
                "turns_sealed": broker.turns_sealed,
                "turns_cancelled": broker.turns_cancelled,
                "ring_bytes_flushed": broker.ring_bytes_flushed,
            },
            "client": {
                "begins_sent": client.begins_sent,
                "frames_sent": client.frames_sent,
                "ends_sent": client.ends_sent,
                "cancels_sent": client.cancels_sent,
                "partials_received": client.partials_received,
                "finals_received": client.finals_received,
                "errors_received": client.errors_received,
                "reconnects": client.reconnects,
                "send_failures": client.send_failures,
            },
        }
    )


async def handle_health(request: web.Request, service: PttControlService) -> web.Response:
    del request
    broker = service.bridge.broker
    active = broker.active_turn
    return web.json_response(
        {
            "status": "ok",
            "capture_open": broker.capture_is_open,
            "active_turn_id": active.voice_turn_id if active else None,
            "active_side": active.side.value if active else None,
            "client_session_id": service.bridge.client_session_id,
        }
    )


def build_app(
    service: PttControlService,
    startup: Callable[[web.Application], Awaitable[None]] | None = None,
    cleanup: Callable[[web.Application], Awaitable[None]] | None = None,
) -> web.Application:
    app = web.Application()

    async def _press(request: web.Request) -> web.Response:
        return await handle_press(request, service)

    async def _release(request: web.Request) -> web.Response:
        return await handle_release(request, service)

    async def _cancel(request: web.Request) -> web.Response:
        return await handle_cancel(request, service)

    async def _events(request: web.Request) -> web.WebSocketResponse:
        return await handle_events(request, service)

    async def _metrics(request: web.Request) -> web.Response:
        return await handle_metrics(request, service)

    async def _health(request: web.Request) -> web.Response:
        return await handle_health(request, service)

    app.router.add_post("/ptt/press", _press)
    app.router.add_post("/ptt/release", _release)
    app.router.add_post("/ptt/cancel", _cancel)
    app.router.add_get("/events", _events)
    app.router.add_get("/metrics", _metrics)
    app.router.add_get("/health", _health)
    if startup is not None:
        app.on_startup.append(startup)
    if cleanup is not None:
        app.on_cleanup.append(cleanup)
    return app


def main() -> int:
    """Run the control service standalone (production sidecar next to N.E.K.O).

    Binds the help pipelines: FUNASR_WORKER_WS (worker WebSocket),
    FUNASR_WORKER_METRICS (worker /metrics), and this server's own bind
    FUNASR_EVENTS_HTTP_HOST / FUNASR_EVENTS_HTTP_PORT.
    """
    import os

    import aiohttp  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    pipeline = WorkerPipeline(
        worker_ws=os.environ.get("FUNASR_WORKER_WS", "ws://127.0.0.1:8765/asr"),
        worker_metrics=os.environ.get(
            "FUNASR_WORKER_METRICS", "http://127.0.0.1:8765/metrics"
        ),
        connect_timeout_s=float(os.environ.get("FUNASR_CLIENT_CONNECT_TIMEOUT_S", "5")),
    )
    service = PttControlService(pipeline.bridge)
    host = os.environ.get("FUNASR_EVENTS_HTTP_HOST", "127.0.0.1")
    port = int(os.environ.get("FUNASR_EVENTS_HTTP_PORT", "8766"))
    app = build_app(service, startup=pipeline.start, cleanup=pipeline.close)
    web.run_app(app, host=host, port=port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())