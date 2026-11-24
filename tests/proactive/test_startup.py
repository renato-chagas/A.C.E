from __future__ import annotations

import base64
import os
import plistlib
import subprocess
import sys
from unittest.mock import Mock

import pytest


def _mock_windows(monkeypatch: pytest.MonkeyPatch) -> Mock:
    from openjarvis.proactive import startup

    monkeypatch.setattr(startup.sys, "platform", "win32")
    monkeypatch.setattr(
        startup.shutil,
        "which",
        lambda name: r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
    )
    run = Mock(
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok\n", stderr=""
        )
    )
    monkeypatch.setattr(startup.subprocess, "run", run)
    return run


def _script_from_call(run: Mock) -> str:
    command = run.call_args.args[0]
    encoded = command[command.index("-EncodedCommand") + 1]
    return base64.b64decode(encoded).decode("utf-16le")


def test_install_registers_least_privilege_logon_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openjarvis.proactive.startup import install_monitor_startup

    run = _mock_windows(monkeypatch)
    monkeypatch.setattr(sys, "executable", "C:\\Tools\\Rena't\\python.exe")

    result = install_monitor_startup()

    script = _script_from_call(run)
    assert result["output"] == "ok"
    assert "Register-ScheduledTask" in script
    assert "New-ScheduledTaskTrigger -AtLogOn" in script
    assert "-RunLevel Limited" in script
    assert "Rena''t" in script
    assert "monitor --background" in script


def test_startup_status_reads_json_without_installing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openjarvis.proactive import startup

    run = _mock_windows(monkeypatch)
    run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout='{"installed":true,"state":"Ready","last_run":"never"}\n',
        stderr="",
    )

    status = startup.get_monitor_startup_status()

    assert status == {"installed": True, "state": "Ready", "last_run": "never"}
    assert "Get-ScheduledTask" in _script_from_call(run)
    assert "Register-ScheduledTask" not in _script_from_call(run)


def test_startup_management_refuses_non_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openjarvis.proactive import startup

    monkeypatch.setattr(startup.sys, "platform", "freebsd")

    with pytest.raises(RuntimeError, match="not supported on freebsd"):
        startup.remove_monitor_startup()


def test_linux_startup_registers_a_user_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from openjarvis.proactive import startup

    monkeypatch.setattr(startup.sys, "platform", "linux")
    monkeypatch.setattr(startup.sys, "executable", "/home/user/venv bin/python")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(startup.shutil, "which", lambda name: "/usr/bin/systemctl")
    run = Mock(
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
    )
    monkeypatch.setattr(startup.subprocess, "run", run)

    result = startup.install_monitor_startup()

    unit_path = startup._systemd_unit_path()
    unit = unit_path.read_text(encoding="utf-8")
    assert result["unit"] == "openjarvis-ace-monitor.service"
    assert '"/home/user/venv bin/python"' in unit
    assert "WantedBy=default.target" in unit
    assert "ExecStop=" in unit
    assert run.call_args_list[0].args[0][-1] == "daemon-reload"
    assert run.call_args_list[1].args[0][-2:] == [
        "enable",
        "openjarvis-ace-monitor.service",
    ]
    if os.name == "posix":
        assert unit_path.stat().st_mode & 0o777 == 0o600


def test_linux_startup_status_and_remove_are_per_user(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from openjarvis.proactive import startup

    monkeypatch.setattr(startup.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(startup.shutil, "which", lambda name: "/usr/bin/systemctl")
    run = Mock(
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
    )
    monkeypatch.setattr(startup.subprocess, "run", run)
    unit_path = startup._systemd_unit_path()
    unit_path.parent.mkdir(parents=True)
    unit_path.write_text("unit", encoding="utf-8")
    enabled = unit_path.parent / "default.target.wants" / unit_path.name
    enabled.parent.mkdir()
    enabled.write_text("enabled", encoding="utf-8")

    assert startup.get_monitor_startup_status() == {
        "installed": True,
        "state": "enabled",
    }
    startup.remove_monitor_startup()

    assert not unit_path.exists()
    assert [call.args[0][-1] for call in run.call_args_list] == [
        "openjarvis-ace-monitor.service",
        "daemon-reload",
    ]


def test_macos_startup_writes_a_per_user_launch_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from openjarvis.proactive import startup

    monkeypatch.setattr(startup.sys, "platform", "darwin")
    monkeypatch.setattr(startup.sys, "executable", "/Users/user/venv/bin/python")
    monkeypatch.setattr(startup.Path, "home", lambda: tmp_path)

    result = startup.install_monitor_startup()

    agent_path = startup._launch_agent_path()
    with agent_path.open("rb") as handle:
        agent = plistlib.load(handle)
    assert result["agent"] == "com.openjarvis.ace-monitor"
    assert agent["Label"] == "com.openjarvis.ace-monitor"
    assert agent["RunAtLoad"] is True
    assert agent["KeepAlive"] is False
    assert agent["ProgramArguments"] == [
        "/Users/user/venv/bin/python",
        "-m",
        "openjarvis.cli",
        "monitor",
        "--background",
    ]
    assert startup.get_monitor_startup_status() == {
        "installed": True,
        "state": "registered",
    }
    startup.remove_monitor_startup()
    assert not agent_path.exists()
