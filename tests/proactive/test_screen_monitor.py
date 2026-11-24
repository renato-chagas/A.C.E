"""Local-only and data-lifetime checks for screen perception."""

from __future__ import annotations

import base64
import subprocess
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openjarvis.cli import _screen
from openjarvis.engine.multi import MultiEngine
from openjarvis.proactive.reasoning import SerializedInferenceEngine
from openjarvis.proactive.screen_monitor import LocalScreenMonitor
from openjarvis.security.guardrails import GuardrailsEngine


def test_screen_monitor_refuses_remote_engine() -> None:
    engine = SimpleNamespace(
        engine_id="custom-http",
        is_local=True,
        _host="https://inference.example/v1",
    )

    with pytest.raises(ValueError, match="verified local"):
        LocalScreenMonitor(engine, "vision-model")


def test_screen_monitor_checks_selected_route_in_mixed_engine_pool() -> None:
    class _ModelEngine:
        def __init__(self, engine_id, models, *, is_cloud=False):
            self.engine_id = engine_id
            self.models = models
            self.is_cloud = is_cloud
            self.is_local = not is_cloud
            self._host = ""

        def list_models(self):
            return self.models

    engine = MultiEngine(
        [
            ("ollama", _ModelEngine("ollama", ["local-vision"])),
            ("cloud", _ModelEngine("cloud", ["gpt-vision"], is_cloud=True)),
        ]
    )

    assert LocalScreenMonitor(engine, "local-vision").model == "local-vision"
    with pytest.raises(ValueError, match="verified local model route"):
        LocalScreenMonitor(engine, "gpt-vision")
    with pytest.raises(ValueError, match="verified local model route"):
        LocalScreenMonitor(engine, "unknown-vision")


def test_screen_monitor_sends_image_only_to_local_engine() -> None:
    engine = MagicMock()
    engine.engine_id = "custom-local"
    engine.is_local = True
    engine.is_cloud = False
    engine._host = "http://127.0.0.1:1234"
    engine.generate.return_value = {"content": "No observation."}
    secured_engine = SerializedInferenceEngine(GuardrailsEngine(engine, scanners=[]))
    monitor = LocalScreenMonitor(secured_engine, "vision-model")

    result = monitor._analyze(b"png-image")

    assert result == "No observation."
    messages = engine.generate.call_args.args[0]
    assert messages[1].images == [base64.b64encode(b"png-image").decode("ascii")]
    kwargs = engine.generate.call_args.kwargs
    assert kwargs["model"] == "vision-model"
    assert kwargs["max_tokens"] == 320
    assert "tools" not in kwargs


def test_non_windows_screen_bytes_capture_to_memory(monkeypatch) -> None:
    monkeypatch.setattr(_screen.sys, "platform", "linux")
    screenshot = SimpleNamespace(rgb=b"rgb", size=(1, 1))
    primary_monitor = {"left": 0, "top": 0, "width": 1, "height": 1}

    class _FakeCapture:
        monitors = [
            {"left": 0, "top": 0, "width": 2, "height": 1},
            primary_monitor,
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return None

        def grab(self, monitor):
            assert monitor == primary_monitor
            return screenshot

    def to_png(rgb, size):
        return b"\x89PNG\r\n\x1a\n" + rgb + bytes(size)

    fake_mss = types.ModuleType("mss")
    fake_mss.mss = _FakeCapture
    fake_tools = types.ModuleType("mss.tools")
    fake_tools.to_png = to_png
    monkeypatch.setitem(sys.modules, "mss", fake_mss)
    monkeypatch.setitem(sys.modules, "mss.tools", fake_tools)

    image = _screen.capture_screen_bytes()

    assert image == b"\x89PNG\r\n\x1a\n" + b"rgb" + bytes((1, 1))


def test_windows_screen_capture_uses_system_powershell_without_a_window(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(_screen.sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))
    monkeypatch.setattr(
        _screen.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False
    )
    image = b"\x89PNG\r\n\x1a\nfixture"

    def run(args, **options):
        assert args[0] == str(
            tmp_path
            / "Windows"
            / "System32"
            / "WindowsPowerShell"
            / "v1.0"
            / "powershell.exe"
        )
        assert "-NoProfile" in args
        assert options["creationflags"] == 0x08000000
        assert options["stdin"] == subprocess.DEVNULL
        assert options["capture_output"] is True
        script = base64.b64decode(args[-1]).decode("utf-16-le")
        assert "CopyFromScreen" in script
        assert "MAX_PNG_BYTES" not in script
        assert "[System.Drawing.Imaging.ImageFormat]::Png" in script
        return subprocess.CompletedProcess(args, 0, base64.b64encode(image), b"")

    monkeypatch.setattr(_screen.subprocess, "run", run)
    assert _screen.capture_screen_bytes() == image
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("return_code", "output", "expected"),
    [
        (1, b"", "screen capture failed"),
        (0, b"not base64", "invalid image bytes"),
        (0, base64.b64encode(b"not PNG"), "did not return a PNG"),
        (0, base64.b64encode(b"\x89PNG\r\n\x1a\nlarge"), "20 MiB image limit"),
    ],
)
def test_windows_screen_rejects_failed_invalid_or_oversize_capture(
    monkeypatch, return_code, output, expected
) -> None:
    monkeypatch.setattr(_screen.sys, "platform", "win32")
    monkeypatch.setattr(_screen, "_MAX_PNG_BYTES", 8)
    monkeypatch.setattr(
        _screen.subprocess,
        "run",
        lambda args, **_kwargs: subprocess.CompletedProcess(
            args, return_code, output, b"fixture"
        ),
    )
    with pytest.raises(RuntimeError, match=expected):
        _screen.capture_screen_bytes()
