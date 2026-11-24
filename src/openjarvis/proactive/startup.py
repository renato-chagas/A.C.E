"""Opt-in per-user login startup for ACE's local resource monitor."""

from __future__ import annotations

import base64
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_TASK_PREFIX = "OpenJarvis-ACE-LocalMonitor-"
_SYSTEMD_UNIT = "openjarvis-ace-monitor.service"
_LAUNCHD_LABEL = "com.openjarvis.ace-monitor"


def _powershell() -> str:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if executable is None:
        raise RuntimeError("Windows PowerShell was not found.")
    return executable


def _ps_quote(value: str) -> str:
    """Quote a value as a PowerShell single-quoted literal."""
    return "'" + value.replace("'", "''") + "'"


def _run_task_script(body: str) -> dict[str, Any]:
    """Run a short, non-interactive ScheduledTasks operation for this user."""
    if sys.platform != "win32":
        raise RuntimeError("ACE login startup is currently supported on Windows.")

    script = (
        "$ErrorActionPreference = 'Stop'; "
        "Import-Module ScheduledTasks -ErrorAction Stop; "
        "$sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value; "
        f"$taskName = {_ps_quote(_TASK_PREFIX)} + $sid; " + body
    )
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    options: dict[str, Any] = {}
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        result = subprocess.run(
            [
                _powershell(),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            **options,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not manage the ACE login task: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            detail or f"PowerShell exited with status {result.returncode}."
        )
    return {
        "task_name": _TASK_PREFIX + "<current-user>",
        "output": result.stdout.strip(),
    }


def _systemd_unit_path() -> Path:
    configured = os.environ.get("XDG_CONFIG_HOME", "")
    config_dir = (
        Path(configured).expanduser() if configured else Path.home() / ".config"
    )
    if not config_dir.is_absolute():
        config_dir = Path.home() / ".config"
    return config_dir / "systemd" / "user" / _SYSTEMD_UNIT


def _launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_LAUNCHD_LABEL}.plist"


def _private_atomic_write(path: Path, content: bytes) -> None:
    """Write a per-user startup file atomically with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        try:
            temporary_path.chmod(0o600)
        except OSError:
            pass
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _systemd_quote(value: str) -> str:
    """Quote one argument for systemd's ExecStart command-line parser."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
        .replace("$", "$$")
    )
    return f'"{escaped}"'


def _systemctl(*arguments: str) -> str:
    executable = shutil.which("systemctl")
    if executable is None:
        raise RuntimeError(
            "systemctl was not found; a systemd user session is required."
        )
    try:
        result = subprocess.run(
            [executable, "--user", *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not manage the systemd user service: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            detail or f"systemctl exited with status {result.returncode}."
        )
    return result.stdout.strip()


def _install_systemd_startup() -> dict[str, Any]:
    unit_path = _systemd_unit_path()
    command = " ".join(
        _systemd_quote(argument)
        for argument in (
            sys.executable,
            "-m",
            "openjarvis.cli",
            "monitor",
            "--background",
        )
    )
    stop_command = " ".join(
        _systemd_quote(argument)
        for argument in (sys.executable, "-m", "openjarvis.cli", "monitor", "--stop")
    )
    unit = (
        "[Unit]\n"
        "Description=OpenJarvis ACE local resource monitor\n\n"
        "[Service]\n"
        "Type=oneshot\n"
        "RemainAfterExit=yes\n"
        f"ExecStart={command}\n"
        f"ExecStop={stop_command}\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )
    _private_atomic_write(unit_path, unit.encode("utf-8"))
    _systemctl("daemon-reload")
    _systemctl("enable", _SYSTEMD_UNIT)
    return {
        "unit": _SYSTEMD_UNIT,
        "output": "ACE monitor will start when this user's systemd session starts.",
    }


def _remove_systemd_startup() -> dict[str, Any]:
    unit_path = _systemd_unit_path()
    if not unit_path.exists():
        return {"unit": _SYSTEMD_UNIT, "output": "ACE login startup is not installed."}
    _systemctl("disable", _SYSTEMD_UNIT)
    unit_path.unlink(missing_ok=True)
    _systemctl("daemon-reload")
    return {"unit": _SYSTEMD_UNIT, "output": "ACE login startup was removed."}


def _systemd_startup_status() -> dict[str, Any]:
    unit_path = _systemd_unit_path()
    enabled_path = unit_path.parent / "default.target.wants" / _SYSTEMD_UNIT
    installed = unit_path.is_file()
    return {
        "installed": installed,
        "state": "enabled" if installed and enabled_path.exists() else "disabled",
    }


def _install_launch_agent() -> dict[str, Any]:
    path = _launch_agent_path()
    payload = {
        "Label": _LAUNCHD_LABEL,
        "ProgramArguments": [
            sys.executable,
            "-m",
            "openjarvis.cli",
            "monitor",
            "--background",
        ],
        "RunAtLoad": True,
        "KeepAlive": False,
    }
    _private_atomic_write(path, plistlib.dumps(payload, fmt=plistlib.FMT_XML))
    return {
        "agent": _LAUNCHD_LABEL,
        "output": "ACE monitor will start at this user's next login.",
    }


def _remove_launch_agent() -> dict[str, Any]:
    path = _launch_agent_path()
    if not path.exists():
        return {
            "agent": _LAUNCHD_LABEL,
            "output": "ACE login startup is not installed.",
        }
    path.unlink()
    return {"agent": _LAUNCHD_LABEL, "output": "ACE login startup was removed."}


def _launch_agent_status() -> dict[str, Any]:
    path = _launch_agent_path()
    if not path.is_file():
        return {"installed": False, "state": "not installed"}
    try:
        with path.open("rb") as handle:
            payload = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException) as exc:
        raise RuntimeError("ACE login agent file is unreadable or invalid.") from exc
    if not isinstance(payload, dict) or payload.get("Label") != _LAUNCHD_LABEL:
        raise RuntimeError("ACE login agent file has an unexpected label.")
    return {"installed": True, "state": "registered"}


def install_monitor_startup() -> dict[str, Any]:
    """Register a non-elevated task that starts ACE monitoring at user logon."""
    if sys.platform == "linux":
        return _install_systemd_startup()
    if sys.platform == "darwin":
        return _install_launch_agent()
    if sys.platform != "win32":
        raise RuntimeError(f"ACE login startup is not supported on {sys.platform}.")

    action = _ps_quote(sys.executable)
    arguments = _ps_quote("-m openjarvis.cli monitor --background")
    description = _ps_quote(
        "Starts the OpenJarvis ACE local resource monitor when this user logs in."
    )
    body = (
        f"$action = New-ScheduledTaskAction -Execute {action} -Argument {arguments}; "
        "$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name; "
        "$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user; "
        "$settings = New-ScheduledTaskSettingsSet "
        "-StartWhenAvailable -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew "
        "-ExecutionTimeLimit ([TimeSpan]::Zero); "
        "$principal = New-ScheduledTaskPrincipal -UserId $user "
        "-LogonType Interactive -RunLevel Limited; "
        "Register-ScheduledTask -TaskName $taskName -Action $action "
        "-Trigger $trigger -Settings $settings -Principal $principal "
        f"-Description {description} -Force | Out-Null; "
        "Write-Output 'ACE monitor will start at the next login.'"
    )
    return _run_task_script(body)


def remove_monitor_startup() -> dict[str, Any]:
    """Remove this user's ACE login task without requiring administrator rights."""
    if sys.platform == "linux":
        return _remove_systemd_startup()
    if sys.platform == "darwin":
        return _remove_launch_agent()
    if sys.platform != "win32":
        raise RuntimeError(f"ACE login startup is not supported on {sys.platform}.")

    body = (
        "$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue; "
        "if ($null -eq $task) { Write-Output "
        "'ACE login startup is not installed.'; exit 0 }; "
        "Unregister-ScheduledTask -TaskName $taskName -Confirm:$false; "
        "Write-Output 'ACE login startup was removed.'"
    )
    return _run_task_script(body)


def get_monitor_startup_status() -> dict[str, Any]:
    """Return whether this user's ACE login task is registered."""
    if sys.platform == "linux":
        return _systemd_startup_status()
    if sys.platform == "darwin":
        return _launch_agent_status()
    if sys.platform != "win32":
        raise RuntimeError(f"ACE login startup is not supported on {sys.platform}.")

    body = (
        "$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue; "
        "if ($null -eq $task) { Write-Output '{\"installed\":false}'; exit 0 }; "
        "$info = Get-ScheduledTaskInfo -TaskName $taskName; "
        "$result = @{ installed = $true; state = [string]$task.State; "
        "last_run = [string]$info.LastRunTime }; "
        "$result | ConvertTo-Json -Compress"
    )
    result = _run_task_script(body)
    output = result["output"]
    if output.startswith("{"):
        import json

        try:
            status = json.loads(output)
        except ValueError as exc:
            raise RuntimeError("PowerShell returned invalid task status.") from exc
        if isinstance(status, dict):
            return status
    raise RuntimeError("PowerShell did not return ACE task status.")


__all__ = [
    "get_monitor_startup_status",
    "install_monitor_startup",
    "remove_monitor_startup",
]
