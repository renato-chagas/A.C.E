"""Tests for configured local runtime startup policy."""

from __future__ import annotations

import sys
from unittest import mock

from openjarvis.engine import _local_runtime


class _Process:
    def poll(self) -> None:
        return None


def test_find_local_ollama_executable_checks_windows_install_path(
    tmp_path, monkeypatch
) -> None:
    executable = tmp_path / "Programs" / "Ollama" / "ollama.exe"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    monkeypatch.setattr(_local_runtime.shutil, "which", lambda _name: None)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert _local_runtime.find_local_ollama_executable() == str(executable)


def test_ollama_bind_address_requires_loopback_http_root() -> None:
    assert (
        _local_runtime._ollama_bind_address("http://127.0.0.1:11434")
        == "127.0.0.1:11434"
    )
    assert _local_runtime._ollama_bind_address("http://[::1]:11434") == "[::1]:11434"
    assert _local_runtime._ollama_bind_address("127.0.0.1:11434") == "127.0.0.1:11434"
    assert _local_runtime._ollama_bind_address("http://example.com:11434") is None
    assert _local_runtime._ollama_bind_address("https://localhost:11434") is None
    assert _local_runtime._ollama_bind_address("http://localhost:11434/v1") is None


def test_start_local_ollama_uses_fixed_argv_and_loopback_environment(
    monkeypatch,
) -> None:
    processes: dict[str, object] = {}
    monkeypatch.setattr(_local_runtime, "_STARTED_PROCESSES", processes)
    monkeypatch.setattr(_local_runtime.shutil, "which", lambda _name: r"C:\ollama.exe")
    process = _Process()
    popen = mock.Mock(return_value=process)
    monkeypatch.setattr(_local_runtime.subprocess, "Popen", popen)

    assert _local_runtime.start_local_ollama("http://127.0.0.1:11434") is True
    assert _local_runtime.start_local_ollama("http://127.0.0.1:11434") is True

    popen.assert_called_once()
    args, kwargs = popen.call_args
    assert args[0] == [r"C:\ollama.exe", "serve"]
    assert "shell" not in kwargs
    assert kwargs["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert kwargs["stdout"] == _local_runtime.subprocess.DEVNULL
    assert kwargs["stderr"] == _local_runtime.subprocess.DEVNULL


def test_start_local_ollama_rejects_remote_host_before_process_launch(
    monkeypatch,
) -> None:
    which = mock.Mock(return_value=r"C:\ollama.exe")
    popen = mock.Mock()
    monkeypatch.setattr(_local_runtime.shutil, "which", which)
    monkeypatch.setattr(_local_runtime.subprocess, "Popen", popen)

    assert _local_runtime.start_local_ollama("http://192.0.2.10:11434") is False
    which.assert_not_called()
    popen.assert_not_called()
