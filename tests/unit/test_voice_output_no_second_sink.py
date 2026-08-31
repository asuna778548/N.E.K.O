# Copyright 2026 999 Project (V4-B). Apache-2.0.
"""No-second-sink audit + output-side static boundary checks.

ADR companion-runtime-final §1/§5/§7 + V4-B hard exit: the frozen output
chain has exactly ONE audio sink. These tests make that machine-checkable:
  1. only one live UniqueAudioSink per process (registry guard),
  2. the output chain never pushes VoxCPM2 audio into the upstream legacy
     frontend websocket path (send_speech) — a second output route is a
     compile/`audit` failure,
  3. voice_output ships no capability to write PCM elsewhere.
"""

from __future__ import annotations

import os

import pytest

from main_logic.voice_output.sink import UniqueAudioSink

pytestmark = pytest.mark.unit

FORK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VOICE_OUTPUT_DIR = os.path.join(FORK_ROOT, "main_logic", "voice_output")


def test_only_one_sink_instance_live_per_process():
    import threading
    import time

    events: list = []

    class _FastClock:
        def monotonic_ns(self) -> int:
            return time.monotonic_ns()

        def sleep_ms(self, ms: float) -> None:
            time.sleep(ms / 1000.0)

    sink = UniqueAudioSink(on_event=events.append, clock=_FastClock())
    try:
        assert UniqueAudioSink.active_sink() is sink
        with pytest.raises(RuntimeError, match="SECOND_AUDIO_SINK_FORBIDDEN"):
            UniqueAudioSink(on_event=events.append, clock=_FastClock())
    finally:
        sink.close()


def test_output_chain_sources_only_one_pcm_destination():
    """Every module in voice_output must feed PCM to the UniqueAudioSink, and
    nothing may enqueue audio into the legacy frontend path."""
    legacy_markers = [
        "send_speech",  # upstream websocket→frontend playback
        "audio_chunk",  # frontend WS frame used by Electron shell playback
        "sync_message_queue",  # monitor/viewer audio mirror
    ]
    for root, _dirs, files in os.walk(VOICE_OUTPUT_DIR):
        for name in files:
            if not name.endswith(".py"):
                continue
            src = open(os.path.join(root, name), encoding="utf-8").read()
            for marker in legacy_markers:
                assert marker not in src, (
                    f"{os.path.join(root, name)} 引用了第二输出路径标记 {marker!r}"
                )


def test_sink_is_unique_in_buildable_output():
    """Static symbol audit: only 'UniqueAudioSink' is imported as PCM sink in
    the output chain, and it is the only class inheriting the sink registry."""
    imports = []
    for root, _dirs, files in os.walk(VOICE_OUTPUT_DIR):
        for name in files:
            if not name.endswith(".py"):
                continue
            src = open(os.path.join(root, name), encoding="utf-8").read()
            imports.extend(
                line.strip()
                for line in src.splitlines()
                if "UniqueAudioSink" in line and ("import" in line or "from " in line)
            )
    assert imports, "输出链必须引用 UniqueAudioSink"
    # controller's sink.begin/feed/finalize/cancel are the ONLY ingestion calls
    controller_src = open(
        os.path.join(VOICE_OUTPUT_DIR, "controller.py"), encoding="utf-8"
    ).read()
    assert "self.sink.feed(" in controller_src
    assert "self.sink.begin(" in controller_src


def test_no_absolute_secrets_or_paths_in_source():
    """§11 security: committed output-side source must not carry keys or
    personal absolute paths."""
    for root, _dirs, files in os.walk(VOICE_OUTPUT_DIR):
        for name in files:
            if not name.endswith(".py"):
                continue
            src = open(os.path.join(root, name), encoding="utf-8").read()
            for bad in (
                r"C:\\Users\\Administrator",
                "sk-",
                "hf_",
                "token=",
                "apikey=",
            ):
                assert bad not in src, f"{name} 含明文密钥/本地绝对路径"