"""Read-only Windows Defender and firewall status for explicit diagnostics."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

_POWERSHELL_DIAGNOSTIC = r'''
$defender = $null
if (Get-Command Get-MpComputerStatus -ErrorAction SilentlyContinue) {
    try {
        $status = Get-MpComputerStatus -ErrorAction Stop
        $defender = [ordered]@{
            service_enabled = [bool]$status.AMServiceEnabled
            antivirus_enabled = [bool]$status.AntivirusEnabled
            realtime_protection_enabled = [bool]$status.RealTimeProtectionEnabled
            behavior_monitor_enabled = [bool]$status.BehaviorMonitorEnabled
        }
    } catch { }
}

$firewall = @()
if (Get-Command Get-NetFirewallProfile -ErrorAction SilentlyContinue) {
    try {
        $firewall = @(
            Get-NetFirewallProfile -PolicyStore ActiveStore -ErrorAction Stop |
                ForEach-Object {
                    [ordered]@{
                        profile = [string]$_.Name
                        enabled = [bool]$_.Enabled
                    }
                }
        )
    } catch { }
}

[ordered]@{
    defender = $defender
    firewall_profiles = $firewall
} | ConvertTo-Json -Depth 4 -Compress
'''


def _is_windows() -> bool:
    return os.name == "nt"


def _powershell_path() -> str | None:
    for name in ("powershell.exe", "pwsh.exe", "powershell", "pwsh"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _state(value: object) -> str:
    if value is True:
        return "enabled"
    if value is False:
        return "disabled"
    return "unknown"


@ToolRegistry.register("system_security")
class SystemSecurityTool(BaseTool):
    """Report selected Windows security settings without changing the host."""

    tool_id = "system_security"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="system_security",
            description=(
                "On explicit request, read the local Microsoft Defender status and "
                "Windows Firewall profile states. This is a limited, read-only "
                "check; it does not scan files, change settings, or use the network."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            category="system",
        )

    def execute(self, **params: Any) -> ToolResult:
        del params
        if not _is_windows():
            return ToolResult(
                tool_name=self.spec.name,
                content=(
                    "This security check currently supports Windows Defender and "
                    "Windows Firewall only. No security state was inspected."
                ),
                success=False,
                metadata={"supported": False},
            )

        powershell = _powershell_path()
        if powershell is None:
            return ToolResult(
                tool_name=self.spec.name,
                content=(
                    "PowerShell was not found; Windows security status is unavailable."
                ),
                success=False,
                metadata={"supported": True, "available": False},
            )

        try:
            result = subprocess.run(
                [
                    powershell,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    _POWERSHELL_DIAGNOSTIC,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            reason = (
                "timed out"
                if isinstance(exc, subprocess.TimeoutExpired)
                else "could not start"
            )
            return ToolResult(
                tool_name=self.spec.name,
                content=(
                    f"PowerShell {reason}; Windows security status is unavailable."
                ),
                success=False,
                metadata={"supported": True, "available": False},
            )

        if result.returncode != 0:
            return ToolResult(
                tool_name=self.spec.name,
                content="Windows security status could not be read by PowerShell.",
                success=False,
                metadata={"supported": True, "available": False},
            )
        try:
            data = json.loads(result.stdout.strip().lstrip("\ufeff"))
        except (json.JSONDecodeError, TypeError):
            return ToolResult(
                tool_name=self.spec.name,
                content="Windows security status returned an unreadable result.",
                success=False,
                metadata={"supported": True, "available": False},
            )

        defender = data.get("defender") if isinstance(data, dict) else None
        profiles = (
            data.get("firewall_profiles", []) if isinstance(data, dict) else []
        )
        if not isinstance(profiles, list):
            profiles = []
        if not isinstance(defender, dict):
            defender = None
        if defender is None and not profiles:
            return ToolResult(
                tool_name=self.spec.name,
                content=(
                    "Windows Defender and Firewall status could not be read. "
                    "Their PowerShell status commands may be unavailable or blocked."
                ),
                success=False,
                metadata={"supported": True, "available": False},
            )

        lines = ["Read-only Windows security status (limited scope):"]
        if defender is None:
            lines.append("Microsoft Defender: status unavailable.")
        else:
            lines.extend(
                [
                    "Microsoft Defender:",
                    "- Antivirus service: "
                    f"{_state(defender.get('service_enabled'))}",
                    "- Antivirus protection: "
                    f"{_state(defender.get('antivirus_enabled'))}",
                    "- Real-time protection: "
                    f"{_state(defender.get('realtime_protection_enabled'))}",
                    "- Behavior monitoring: "
                    f"{_state(defender.get('behavior_monitor_enabled'))}",
                ]
            )

        if not profiles:
            lines.append("Windows Firewall profiles: status unavailable.")
        else:
            lines.append("Windows Firewall profiles:")
            for profile in profiles:
                if not isinstance(profile, dict):
                    continue
                name = str(profile.get("profile", "Unknown"))[:32]
                lines.append(f"- {name}: {_state(profile.get('enabled'))}")

        lines.append(
            "This checks selected Defender and Firewall settings only; it is not "
            "a malware scan, a complete security audit, or proof that no other "
            "antivirus is installed."
        )
        return ToolResult(
            tool_name=self.spec.name,
            content="\n".join(lines),
            success=True,
            metadata={
                "supported": True,
                "available": True,
                "defender": defender,
                "firewall_profiles": profiles,
            },
        )


__all__ = ["SystemSecurityTool"]
