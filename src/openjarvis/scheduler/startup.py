"""Opt-in per-user Windows login startup for the scheduled task service."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_TASK_PREFIX = "OpenJarvis-ACE-Scheduler-"


def _powershell() -> str:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if executable is None:
        raise RuntimeError("Windows PowerShell was not found.")
    return executable


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _run_task_script(body: str) -> str:
    if sys.platform != "win32":
        raise RuntimeError("Scheduler login startup is currently supported on Windows.")
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
        raise RuntimeError(
            f"Could not manage the ACE scheduler login task: {exc}"
        ) from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            detail or f"PowerShell exited with status {result.returncode}."
        )
    return result.stdout.strip()


def _pythonw() -> str:
    executable = Path(sys.executable).with_name("pythonw.exe")
    if not executable.is_file():
        raise RuntimeError(
            "pythonw.exe was not found beside the active Python interpreter; "
            "cannot start the scheduler without opening a console window."
        )
    return str(executable)


def install_scheduler_startup() -> dict[str, str]:
    """Register a least-privilege task to start the scheduler at user logon."""
    if sys.platform != "win32":
        raise RuntimeError("Scheduler login startup is currently supported on Windows.")
    action = _ps_quote(_pythonw())
    arguments = _ps_quote("-m openjarvis.cli scheduler start")
    description = _ps_quote(
        "Runs approved OpenJarvis ACE scheduled tasks for this user."
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
        "Write-Output 'ACE scheduler will start at the next login.'"
    )
    output = _run_task_script(body)
    return {"task_name": _TASK_PREFIX + "<current-user>", "output": output}


def remove_scheduler_startup() -> dict[str, str]:
    """Remove the current user's scheduler login task, if it exists."""
    body = (
        "$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue; "
        "if ($null -eq $task) { Write-Output "
        "'ACE scheduler login startup is not installed.'; exit 0 }; "
        "Unregister-ScheduledTask -TaskName $taskName -Confirm:$false; "
        "Write-Output 'ACE scheduler login startup was removed.'"
    )
    output = _run_task_script(body)
    return {"task_name": _TASK_PREFIX + "<current-user>", "output": output}


def get_scheduler_startup_status() -> dict[str, Any]:
    """Read this user's scheduler login task state and last result."""
    body = (
        "$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue; "
        "if ($null -eq $task) { "
        '\'{"installed":false,"state":"not installed"}\'; exit 0 }; '
        "$info = Get-ScheduledTaskInfo -TaskName $taskName; "
        "$lastRun = if ($info.LastRunTime -eq [DateTime]::MinValue) "
        "{ '' } else { $info.LastRunTime.ToString('o') }; "
        "[pscustomobject]@{ installed = $true; state = $task.State.ToString(); "
        "last_run = $lastRun; last_result = $info.LastTaskResult } "
        "| ConvertTo-Json -Compress"
    )
    output = _run_task_script(body)
    try:
        status = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Windows returned an invalid scheduler startup status."
        ) from exc
    if not isinstance(status, dict) or not isinstance(status.get("installed"), bool):
        raise RuntimeError("Windows returned an invalid scheduler startup status.")
    return status


__all__ = [
    "get_scheduler_startup_status",
    "install_scheduler_startup",
    "remove_scheduler_startup",
]
