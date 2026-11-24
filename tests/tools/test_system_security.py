"""Tests for the explicitly requested, read-only Windows security report."""

from __future__ import annotations

import json
from types import SimpleNamespace

from openjarvis.tools import system_security


def test_security_report_does_not_inspect_non_windows_platform(monkeypatch) -> None:
    monkeypatch.setattr(system_security, "_is_windows", lambda: False)
    monkeypatch.setattr(
        system_security,
        "_powershell_path",
        lambda: (_ for _ in ()).throw(AssertionError("PowerShell must not run")),
    )

    result = system_security.SystemSecurityTool().execute()

    assert not result.success
    assert result.metadata["supported"] is False
    assert "No security state was inspected" in result.content


def test_security_report_formats_only_selected_local_status(monkeypatch) -> None:
    payload = {
        "defender": {
            "service_enabled": True,
            "antivirus_enabled": True,
            "realtime_protection_enabled": False,
            "behavior_monitor_enabled": True,
        },
        "firewall_profiles": [
            {"profile": "Domain", "enabled": True},
            {"profile": "Private", "enabled": False},
        ],
    }
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(system_security, "_is_windows", lambda: True)
    monkeypatch.setattr(system_security, "_powershell_path", lambda: "powershell.exe")
    monkeypatch.setattr(system_security.subprocess, "run", fake_run)

    result = system_security.SystemSecurityTool().execute()

    assert result.success
    assert result.metadata["defender"]["realtime_protection_enabled"] is False
    assert result.metadata["firewall_profiles"] == payload["firewall_profiles"]
    assert "Real-time protection: disabled" in result.content
    assert "Private: disabled" in result.content
    assert "not a malware scan" in result.content
    command, kwargs = calls[0]
    assert command[0] == "powershell.exe"
    assert "-NoProfile" in command
    assert kwargs["timeout"] == 20
    assert kwargs["check"] is False


def test_security_report_marks_missing_status_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(system_security, "_is_windows", lambda: True)
    monkeypatch.setattr(system_security, "_powershell_path", lambda: "powershell.exe")
    monkeypatch.setattr(
        system_security.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"defender": null, "firewall_profiles": []}',
            stderr="",
        ),
    )

    result = system_security.SystemSecurityTool().execute()

    assert not result.success
    assert result.metadata["available"] is False
    assert "could not be read" in result.content
