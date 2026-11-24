"""Tests for opt-in Windows scheduler login startup."""

from __future__ import annotations

import base64
import json
import subprocess

import pytest

from openjarvis.scheduler import startup


def test_install_registers_hidden_least_privilege_logon_task(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(startup.sys, "platform", "win32")
    python = tmp_path / "venv with spaces" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    pythonw = python.with_name("pythonw.exe")
    pythonw.touch()
    monkeypatch.setattr(startup.sys, "executable", str(python))
    monkeypatch.setattr(startup.shutil, "which", lambda _name: "powershell.exe")
    captured: dict[str, object] = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="installed", stderr="")

    monkeypatch.setattr(startup.subprocess, "run", run)

    result = startup.install_scheduler_startup()

    command = captured["command"]
    assert isinstance(command, list)
    script = base64.b64decode(command[-1]).decode("utf-16le")
    assert str(pythonw) in script
    assert "-m openjarvis.cli scheduler start" in script
    assert "New-ScheduledTaskTrigger -AtLogOn" in script
    assert "-LogonType Interactive -RunLevel Limited" in script
    assert "-MultipleInstances IgnoreNew" in script
    assert "-ExecutionTimeLimit ([TimeSpan]::Zero)" in script
    assert "-RunLevel Highest" not in script
    assert result["output"] == "installed"


def test_remove_unregisters_only_the_current_users_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(startup.sys, "platform", "win32")
    captured: list[str] = []

    def run_script(body: str) -> str:
        captured.append(body)
        return "removed"

    monkeypatch.setattr(startup, "_run_task_script", run_script)

    result = startup.remove_scheduler_startup()

    assert "Get-ScheduledTask -TaskName $taskName" in captured[0]
    assert "Unregister-ScheduledTask -TaskName $taskName -Confirm:$false" in captured[0]
    assert result["output"] == "removed"


def test_startup_status_parses_task_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        startup,
        "_run_task_script",
        lambda _body: json.dumps(
            {
                "installed": True,
                "state": "Running",
                "last_run": "2026-11-18T09:00:00.0000000-03:00",
                "last_result": 0,
            }
        ),
    )

    assert startup.get_scheduler_startup_status() == {
        "installed": True,
        "state": "Running",
        "last_run": "2026-11-18T09:00:00.0000000-03:00",
        "last_result": 0,
    }


def test_startup_management_is_windows_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(startup.sys, "platform", "linux")

    with pytest.raises(RuntimeError, match="supported on Windows"):
        startup.install_scheduler_startup()
