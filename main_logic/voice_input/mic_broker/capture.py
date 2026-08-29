"""Capture backends for the PTT microphone broker.

``CapturePort`` is the only surface the broker knows about. Production uses
``SoundDeviceCapture`` (16 kHz mono PCM16); tests use ``ScriptedCapture``.
Capture callbacks arrive on the PortAudio audio thread, so implementations
must not allocate heavily or block.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol

from utils.audio_processor import AudioProcessor

CaptureFrameCallback = Callable[[bytes, int], None]

# Production capture format: 16 kHz mono PCM16, 60 ms callback chunks.
CAPTURE_SAMPLE_RATE_HZ = 16_000
CAPTURE_CHUNK_MS = 60


class CapturePort(Protocol):
    """One exclusive microphone capture stream."""

    @property
    def is_open(self) -> bool: ...

    def open(self, on_frame: CaptureFrameCallback) -> None: ...

    def close(self) -> None: ...


class ScriptedCapture:
    """Deterministic capture double driven by the test."""

    def __init__(self) -> None:
        self._on_frame: CaptureFrameCallback | None = None
        self.open_count = 0
        self.close_count = 0
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self, on_frame: CaptureFrameCallback) -> None:
        if self._open:
            raise RuntimeError("CAPTURE_ALREADY_OPEN")
        self._on_frame = on_frame
        self.open_count += 1
        self._open = True

    def close(self) -> None:
        if not self._open:
            return
        self.close_count += 1
        self._open = False
        self._on_frame = None

    def emit(self, pcm16: bytes, monotonic_ns: int | None = None) -> None:
        """Push one frame as if the device callback fired."""
        if not self._open or self._on_frame is None:
            raise RuntimeError("CAPTURE_NOT_OPEN")
        self._on_frame(
            pcm16,
            time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
        )


class SoundDeviceCapture:
    """Real Windows microphone capture via sounddevice (PortAudio).

    Fails closed: if the device cannot be opened, ``open`` raises and the
    broker reports a capture error instead of pretending to listen.
    """

    def __init__(
        self,
        *,
        sample_rate_hz: int = CAPTURE_SAMPLE_RATE_HZ,
        chunk_ms: int = CAPTURE_CHUNK_MS,
        noise_reduce_enabled: bool = False,
        device_index: int | None = None,
    ) -> None:
        self._sample_rate_hz = int(sample_rate_hz)
        self._frames_per_chunk = max(1, int(chunk_ms) * sample_rate_hz // 1000)
        self._device_index = device_index
        self._noise_reduce_enabled = bool(noise_reduce_enabled)
        self._stream: object | None = None
        self._processor: AudioProcessor | None = None
        if noise_reduce_enabled:
            self._processor = AudioProcessor(noise_reduce_enabled=True)

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def open(self, on_frame: CaptureFrameCallback) -> None:
        if self._stream is not None:
            raise RuntimeError("CAPTURE_ALREADY_OPEN")

        import sounddevice as sd

        loop = 0
        while loop < 2:  # one retry after a device default-change race
            try:
                stream = sd.RawInputStream(
                    samplerate=self._sample_rate_hz,
                    channels=1,
                    dtype="int16",
                    blocksize=self._frames_per_chunk,
                    device=self._device_index,
                    callback=self._make_callback(on_frame),
                )
                stream.start()
                self._stream = stream
                return
            except Exception:
                loop += 1
                if loop >= 2:
                    raise
                # The default device may have just changed under us; refresh
                # and retry once with the system default.
                self._device_index = None

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            # A dead device must not break the broker's seal path; the stream
            # handle is discarded either way.
            pass

    def _make_callback(self, on_frame: CaptureFrameCallback):
        def _callback(in_data, frames, time_info, status):  # noqa: ANN001
            del frames, time_info
            if status:
                # Overflows/resets are surfaced by the broker's device-change
                # path; here we simply keep streaming.
                pass
            pcm16 = bytes(in_data)
            if self._processor is not None:
                try:
                    pcm16 = self._processor.process_chunk(pcm16)
                except Exception:
                    pcm16 = bytes(in_data)
            on_frame(pcm16, time.monotonic_ns())

        return _callback
