"""Assembly of the V4-A input chain into one runnable pipeline.

The MicBroker (unique capture owner) + the PTT bridge (final router) +
the FunASR worker client (transport) are wired here exactly once, so a
standalone sidecar (``control_server.main``) and the E2E driver share the
same assembly. Tests inject a scripted capture so no microphone is needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .bridge import PttVoiceInputBridge
from .broker import MicBroker
from .contracts import PttSide
from .funasr_client import FunasrWorkerClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkerPipeline:
    """One assembled input chain, ready to run against a live worker."""

    worker_ws: str
    worker_metrics: str
    connect_timeout_s: float = 5.0
    capture_factory: Callable[[], Any] | None = None
    client: FunasrWorkerClient = field(init=False)
    broker: MicBroker = field(init=False)
    bridge: PttVoiceInputBridge = field(init=False)

    def __post_init__(self) -> None:
        if self.capture_factory is None:
            from .capture import SoundDeviceCapture  # noqa: PLC0415

            self.capture_factory = SoundDeviceCapture
        self.broker = MicBroker(capture_factory=self.capture_factory, sink=None)
        self.client = FunasrWorkerClient(
            endpoint=self.worker_ws,
            on_partial=None,  # type: ignore[arg-type]
            on_final=None,  # type: ignore[arg-type]
            on_error=None,  # type: ignore[arg-type]
            metrics_endpoint=self.worker_metrics,
            connect_timeout_s=self.connect_timeout_s,
        )
        self.bridge = PttVoiceInputBridge(
            broker=self.broker,
            client=self.client,
            on_conversation_final=None,  # type: ignore[arg-type]
            on_action_final=None,  # type: ignore[arg-type]
            on_partial_display=None,  # type: ignore[arg-type]
            on_cue=None,  # type: ignore[arg-type]
            on_error=None,  # type: ignore[arg-type]
        )
        # Production wiring identical to the E2E driver: the bridge is the one
        # and only MicBroker sink and the client's dispatcher.
        self.bridge.broker.sink = self.bridge  # type: ignore[assignment]
        self.client.on_partial = self.bridge.handle_partial  # type: ignore[assignment]
        self.client.on_final = self.bridge.handle_final  # type: ignore[assignment]
        self.client.on_error = self.bridge.handle_worker_error  # type: ignore[assignment]

    async def start(self, _app: Any = None) -> None:
        await self.client.start()
        logger.info("funasr worker client connected (%s)", self.worker_ws)

    async def close(self, _app: Any = None) -> None:
        await self.client.close()