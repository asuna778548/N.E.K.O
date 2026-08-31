# Copyright 2026 999 Project (V4-B). Apache-2.0.
"""Async client for the VoxCPM2 speech-worker wire protocol (V1).

Speaks the frozen protocol from ``voice_workers/voxcpm2_speech_worker``:
  client→server: {"type": "synthesize"|"cancel"|"ping"}
  server→client: ready / speech_started / pcm_meta+bytes / speech_done /
                 speech_error / cancelled / pong

The transport is injectable (:class:`SpeechWorkerTransport`) so tests run the
controller against a deterministic fake without a live worker; the default
:class:`WebSocketSpeechWorkerTransport` connects to the resident worker.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Protocol

from .contracts import SpeechFailure


class SpeechWorkerTransport(Protocol):
    async def connect(self, url: str) -> None: ...

    async def send_json(self, payload: dict) -> None: ...

    async def recv(self) -> bytes: ...  # text frames are JSON-encoded bytes

    async def close(self) -> None: ...


@dataclass(slots=True)
class SynthesizeResult:
    sample_rate: int
    first_pcm_at_s: float
    realtime_factor: float | None
    samples: int
    cancelled: bool
    failed: SpeechFailure | None


class WebSocketSpeechWorkerTransport:
    """Real transport over ``websockets`` (fork pins websockets~=15.0.1)."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._ws = None

    async def connect(self, url: str | None = None) -> None:
        import websockets

        target = url or self.url
        # websockets 15: asyncio client at websockets.asyncio.client.connect
        try:
            from websockets.asyncio.client import connect as ws_connect
        except ImportError:  # older API surface
            ws_connect = websockets.connect
        self._ws = await ws_connect(target, open_timeout=10)
        return

    async def send_json(self, payload: dict) -> None:
        await self._ws.send(json.dumps(payload, ensure_ascii=False))

    async def recv(self) -> bytes:
        msg = await self._ws.recv()
        if isinstance(msg, bytes):
            return msg
        return msg.encode("utf-8")

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


class VoxCpm2SpeechClient:
    """One WS session exposing streaming synthesis + cancel.

    ``synthesize_stream`` yields per-chunk PCM bytes; the caller feeds them to
    the unique sink. A concurrent ``cancel()`` stops generation at the worker
    (engine-level) AND the local generator (client closes the stream).
    """

    def __init__(self, transport: SpeechWorkerTransport, *, connect_url: Optional[str] = None) -> None:
        self.transport = transport
        self.connect_url = connect_url
        self.ready_info: dict = {}

    async def open(self) -> dict:
        await self.transport.connect(self.connect_url)
        raw = await self.transport.recv()
        msg = json.loads(raw.decode("utf-8"))
        if msg.get("type") != "ready":
            raise RuntimeError(f"speech-worker did not send ready: {msg}")
        self.ready_info = msg.get("engine", {})
        return self.ready_info

    async def close(self) -> None:
        await self.transport.close()

    async def synthesize_stream(
        self,
        speech_id: str,
        text: str,
        request_id: str = "",
        profile_id: str = "neutral_test",
    ) -> AsyncIterator[bytes]:
        """Yield int16 PCM chunks until done/cancelled/error."""
        await self.transport.send_json(
            {
                "type": "synthesize",
                "speech_id": speech_id,
                "request_id": request_id,
                "text": text,
                "profile": profile_id,
            }
        )
        sample_rate = 0
        while True:
            raw = await self.transport.recv()
            if raw.startswith(b"{"):
                msg = json.loads(raw.decode("utf-8"))
                mtype = msg.get("type")
                if mtype == "speech_started":
                    sample_rate = int(msg.get("sample_rate", 0))
                    continue
                if mtype == "pcm_meta":
                    samples = int(msg.get("samples", 0))
                    pcm = await self.transport.recv()
                    if len(pcm) != samples * 2:
                        raise RuntimeError(
                            f"speech-worker pcm_meta/bytes mismatch: {samples} vs {len(pcm)//2}"
                        )
                    yield pcm
                    continue
                if mtype == "speech_done":
                    # Drain done; generator ends naturally on StopAsyncIteration
                    self._done_payload = msg
                    return
                if mtype == "cancelled":
                    raise _CancelledStream(speech_id)
                if mtype == "speech_error":
                    raise _ErroredStream(
                        SpeechFailure(
                            code=str(msg.get("code", "speech_error")),
                            message=str(msg.get("message", "")),
                            speech_id=str(msg.get("speech_id", speech_id)),
                            request_id=str(msg.get("request_id", request_id)),
                        )
                    )
                if mtype == "pong":
                    continue
                raise RuntimeError(f"unexpected speech-worker frame {mtype}")
            raise RuntimeError("unexpected binary frame from speech-worker")

    async def cancel(self, speech_id: str) -> None:
        await self.transport.send_json({"type": "cancel", "speech_id": speech_id})


class _CancelledStream(Exception):
    def __init__(self, speech_id: str):
        super().__init__(f"stream cancelled: {speech_id}")
        self.speech_id = speech_id


class _ErroredStream(Exception):
    def __init__(self, failure: SpeechFailure):
        super().__init__(failure.message)
        self.failure = failure