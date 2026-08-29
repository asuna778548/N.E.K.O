"""Async client for the resident FunASR 2-pass worker.

Speaks the proposal protocol from
``puosui/docs/contracts/proposals/voice-streaming-contract-v1-proposal.md``:
``voice.begin`` / ``audio.frame`` / ``voice.end`` / ``voice.cancel`` upstream,
``asr.partial`` / ``asr.final`` / ``voice.error`` downstream, with
``/metrics`` served over HTTP by the worker.

The client is transport-faithful: it never synthesizes a final from partials
and it fails closed (``voice.error`` surfaced) when the worker is
unreachable, so upstream routing can never mistake silence for speech.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

import aiohttp
import websockets

from .contracts import MicTurnDescriptor

logger = logging.getLogger(__name__)

PartialCallback = Callable[[str, str, int], Awaitable[None]]  # turn_id, text, ns
FinalCallback = Callable[[str, str, int], Awaitable[None]]  # turn_id, text, ns
ErrorCallback = Callable[[str | None, str, str], Awaitable[None]]  # turn_id, code, detail


class WorkerTransport(Protocol):
    """Minimal WS surface so tests can inject a fake server."""

    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | Any: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class FunasrClientMetrics:
    frames_sent: int = 0
    begins_sent: int = 0
    ends_sent: int = 0
    cancels_sent: int = 0
    partials_received: int = 0
    finals_received: int = 0
    errors_received: int = 0
    reconnects: int = 0
    send_failures: int = 0
    malformed_downstream: int = 0


@dataclass(slots=True)
class FunasrWorkerClient:
    """One client connection; frame pump + downstream dispatcher."""

    endpoint: str
    on_partial: PartialCallback
    on_final: FinalCallback
    on_error: ErrorCallback
    metrics_endpoint: str | None = None
    connect_timeout_s: float = 5.0
    _ws: WorkerTransport | None = None
    _outgoing: asyncio.Queue[tuple[str, ...] | None] | None = None
    _tasks: list[asyncio.Task[None]] = field(default_factory=list, init=False)
    _loop: asyncio.AbstractEventLoop | None = None
    _loop_thread_id: int | None = None
    _closed: bool = False
    _live_bytes_by_turn: dict[str, int] = field(default_factory=dict, init=False)
    client_metrics: FunasrClientMetrics = field(
        default_factory=FunasrClientMetrics, init=False
    )

    # -- lifecycle ----------------------------------------------------------
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._loop_thread_id = threading.get_ident()
        self._outgoing = asyncio.Queue()
        await self._connect()
        self._tasks.append(asyncio.create_task(self._send_pump()))
        self._tasks.append(asyncio.create_task(self._recv_pump()))

    async def _connect(self) -> None:
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(self.endpoint), timeout=self.connect_timeout_s
            )
        except Exception as exc:
            raise RuntimeError(f"FUNASR_WORKER_UNREACHABLE: {exc!r}") from exc

    async def close(self) -> None:
        self._closed = True
        if self._outgoing is not None:
            self._outgoing.put_nowait(None)
        try:
            # The send pump drains the queue (including the None sentinel) so
            # every enqueued protocol message is flushed before shutdown.
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True), timeout=2.0
            )
        except asyncio.TimeoutError:
            for task in self._tasks:
                task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    # -- upstream (thread-safe entry points for the MicBroker sink) ---------
    def begin_turn(
        self, descriptor: MicTurnDescriptor, client_session_id: str
    ) -> None:
        self._enqueue(
            "voice.begin",
            descriptor,
            voice_turn_id=descriptor.voice_turn_id,
            channel=descriptor.channel,
            input_owner=descriptor.input_owner,
            asr_profile_id=descriptor.asr_profile_id,
            client_session_id=client_session_id,
            pressed_monotonic_ns=descriptor.pressed_monotonic_ns,
            monotonic_ns=descriptor.pressed_monotonic_ns,
        )

    def queue_frame(
        self, descriptor: MicTurnDescriptor, pcm16: bytes, monotonic_ns: int
    ) -> None:
        self._live_bytes_by_turn[descriptor.voice_turn_id] = (
            self._live_bytes_by_turn.get(descriptor.voice_turn_id, 0) + len(pcm16)
        )
        self._enqueue(
            "audio.frame",
            descriptor,
            voice_turn_id=descriptor.voice_turn_id,
            pcm16_base64=base64.b64encode(pcm16).decode("ascii"),
            monotonic_ns=monotonic_ns,
        )

    def seal_turn(
        self, descriptor: MicTurnDescriptor, pcm_tail: bytes, released_monotonic_ns: int
    ) -> None:
        # The ring tail is the reliable full press-to-release capture. Frames
        # were already live-streamed via queue_frame; only forward the part of
        # the tail that live streaming did not cover, so the offline final sees
        # exactly once all audio and never a duplicated utterance.
        live = self._live_bytes_by_turn.get(descriptor.voice_turn_id, 0)
        remaining = pcm_tail[live:] if live < len(pcm_tail) else b""
        self._live_bytes_by_turn.pop(descriptor.voice_turn_id, None)
        if remaining:
            self.queue_frame(descriptor, remaining, released_monotonic_ns)
        self._enqueue(
            "voice.end",
            descriptor,
            voice_turn_id=descriptor.voice_turn_id,
            released_monotonic_ns=released_monotonic_ns,
            monotonic_ns=released_monotonic_ns,
        )

    def cancel_turn(self, voice_turn_id: str, reason: str) -> None:
        if self._outgoing is None or self._loop is None:
            return
        self._queue_raw(
            (
                "voice.cancel",
                json.dumps(
                    {
                        "type": "voice.cancel",
                        "voice_turn_id": voice_turn_id,
                        "reason": reason,
                        "monotonic_ns": time.monotonic_ns(),
                    },
                    ensure_ascii=False,
                ),
            )
        )

    def _enqueue(self, message_type: str, descriptor: MicTurnDescriptor, **payload: Any) -> None:
        if self._outgoing is None or self._loop is None:
            return
        message = {"type": message_type, **payload}
        self._queue_raw(
            (message_type, json.dumps(message, ensure_ascii=False))
        )

    def _queue_raw(self, item: tuple[str, ...]) -> None:
        """Queue an upstream message; safe from the audio thread."""
        assert self._outgoing is not None and self._loop is not None
        if threading.get_ident() == self._loop_thread_id:
            self._outgoing.put_nowait(item)
        else:
            self._loop.call_soon_threadsafe(self._outgoing.put_nowait, item)

    async def _send_pump(self) -> None:
        assert self._outgoing is not None
        while True:
            item = await self._outgoing.get()
            if item is None:
                return
            message_type, raw = item
            if message_type == "voice.begin":
                self.client_metrics.begins_sent += 1
            elif message_type == "audio.frame":
                self.client_metrics.frames_sent += 1
            elif message_type == "voice.end":
                self.client_metrics.ends_sent += 1
            elif message_type == "voice.cancel":
                self.client_metrics.cancels_sent += 1
            try:
                if self._ws is None:
                    raise RuntimeError("FUNASR_WORKER_NOT_CONNECTED")
                await self._ws.send(raw)
            except Exception as exc:
                self.client_metrics.send_failures += 1
                logger.warning("funasr send failed: %r", exc)
                await self._safe_error(
                    json.loads(raw).get("voice_turn_id"),
                    "FUNASR_SEND_FAILED",
                    repr(exc),
                )

    # -- downstream ----------------------------------------------------------
    async def _recv_pump(self) -> None:
        while not self._closed:
            if self._ws is None:
                await asyncio.sleep(0.05)
                continue
            try:
                raw = await self._ws.recv()
            except Exception:
                if self._closed:
                    return
                await asyncio.sleep(0.1)
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                continue
            await self._dispatch(message)

    async def _dispatch(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        voice_turn_id = message.get("voice_turn_id")
        if message_type in ("asr.partial", "asr.final"):
            # Downstream shape contract (proposal voice-streaming v1): a
            # partial must never carry final=True and both carry
            # asr_profile_id; anything else is dropped, never routed.
            if (message_type == "asr.partial" and message.get("final") is True) or (
                message_type == "asr.final"
                and (message.get("final") is not True or message.get("asr_profile_id") is None)
            ):
                self.client_metrics.malformed_downstream += 1
                logger.warning("malformed downstream %r dropped", message_type)
                return
        if message_type == "asr.partial":
            self.client_metrics.partials_received += 1
            await self.on_partial(
                voice_turn_id, str(message.get("text", "")), int(message.get("monotonic_ns", 0))
            )
        elif message_type == "asr.final":
            self.client_metrics.finals_received += 1
            await self.on_final(
                voice_turn_id, str(message.get("text", "")), int(message.get("monotonic_ns", 0))
            )
        elif message_type == "voice.error":
            self.client_metrics.errors_received += 1
            await self.on_error(
                voice_turn_id, str(message.get("code", "unknown")), str(message.get("detail", ""))
            )
        # Unknown types are ignored for forward compatibility.

    async def _safe_error(self, voice_turn_id: str | None, code: str, detail: str) -> None:
        try:
            await self.on_error(voice_turn_id, code, detail)
        except Exception:
            logger.exception("error callback failed")

    # -- metrics -------------------------------------------------------------
    async def fetch_metrics(self, timeout_s: float = 3.0) -> dict[str, Any]:
        """Read the worker's HTTP ``/metrics`` snapshot."""
        endpoint = self.metrics_endpoint
        if endpoint is None:
            raise RuntimeError("FUNASR_METRICS_ENDPOINT_NOT_SET")
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(endpoint) as response:
                response.raise_for_status()
                return await response.json(content_type=None)


def new_client_session_id() -> str:
    return str(uuid.uuid4())
