from __future__ import annotations

import ctypes
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openjarvis.proactive import monitor_daemon, notifications


def test_notification_message_excludes_measured_values_and_process_details() -> None:
    message = notifications._notification_message(
        "Uso de CPU alto: 98% (limite 95%). top.exe usa 80%."
    )

    assert "CPU" in message
    assert "98" not in message
    assert "95" not in message
    assert "top.exe" not in message
    assert "jarvis monitor --history" in message


def test_desktop_notice_links_back_to_the_monitoring_panel(monkeypatch) -> None:
    notify = Mock(return_value=True)
    monkeypatch.setattr(notifications.sys, "platform", "win32")
    monkeypatch.setattr(notifications, "_notify_windows", notify)

    assert notifications.notify_monitor_event(
        "Uso de CPU alto: 98%. Processo local.exe.", history_hint="System Monitoring"
    )
    notify.assert_called_once_with(
        "A.C.E.", "CPU: precisa de atenção. Consulte System Monitoring."
    )


def test_recovery_notification_is_generic_and_says_condition_recovered() -> None:
    message = notifications._notification_message(
        "Memória disponível voltou acima do limite configurado."
    )

    assert message == (
        "Memória disponível: voltou ao normal. Consulte jarvis monitor --history."
    )


def test_linux_notification_uses_native_argv_without_shell(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(notifications.sys, "platform", "linux")
    monkeypatch.setattr(
        notifications.shutil, "which", lambda name: "/usr/bin/notify-send"
    )

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(notifications.subprocess, "run", fake_run)

    assert notifications.notify_monitor_event("Uso de CPU alto: 98%.") is True
    args, kwargs = calls[0]
    assert args[:3] == [
        "/usr/bin/notify-send",
        "--app-name=A.C.E.",
        "--expire-time=7000",
    ]
    assert args[-1] == "CPU: precisa de atenção. Consulte jarvis monitor --history."
    assert kwargs["timeout"] == 3
    assert "shell" not in kwargs


def test_macos_notification_escapes_script_text(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(
        notifications.shutil, "which", lambda name: "/usr/bin/osascript"
    )
    monkeypatch.setattr(
        notifications.subprocess,
        "run",
        lambda args, **kwargs: (
            calls.append((args, kwargs)) or subprocess.CompletedProcess(args, 0)
        ),
    )

    assert notifications._notify_macos('A "title"', 'a "message"') is True
    assert calls[0][0][0:2] == ["/usr/bin/osascript", "-e"]
    assert '\\"title\\"' in calls[0][0][2]
    assert '\\"message\\"' in calls[0][0][2]
    assert "shell" not in calls[0][1]


def test_missing_native_backend_leaves_notification_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(notifications.sys, "platform", "linux")
    monkeypatch.setattr(notifications.shutil, "which", lambda name: None)

    assert notifications.notify_monitor_event("Uso de CPU alto.") is False


def test_windows_notification_starts_a_background_worker(monkeypatch) -> None:
    seen = []

    class FakeThread:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def start(self):
            seen.append("started")

    monkeypatch.setattr(notifications.sys, "platform", "win32")
    monkeypatch.setattr(notifications.threading, "Thread", FakeThread)

    assert notifications.notify_monitor_event("Uso de CPU alto: 98%.") is True
    assert seen[0]["args"] == (
        "A.C.E.",
        "CPU: precisa de atenção. Consulte jarvis monitor --history.",
    )
    assert seen[0]["daemon"] is True
    assert seen[1] == "started"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ABI")
@pytest.mark.parametrize("shell_accepts", [True, False])
def test_native_balloon_receives_windows_abi_and_releases_icon(
    monkeypatch, shell_accepts
) -> None:
    emitted = []

    def shell(operation, pointer):
        data = pointer._obj
        assert data.cbSize == (976 if ctypes.sizeof(ctypes.c_void_p) == 8 else 956)
        assert data.hWnd == 123
        emitted.append((operation, data.uFlags, data.szInfoTitle, data.szInfo))
        return shell_accepts or operation != 1

    user32 = SimpleNamespace(
        CreateWindowExW=Mock(return_value=123),
        DestroyWindow=Mock(return_value=True),
        LoadIconW=Mock(return_value=456),
    )
    shell32 = SimpleNamespace(Shell_NotifyIconW=Mock(side_effect=shell))
    monkeypatch.setattr(
        ctypes, "WinDLL", lambda name, **_: {"user32": user32, "shell32": shell32}[name]
    )
    sleep = Mock()
    monkeypatch.setattr(notifications.time, "sleep", sleep)

    if shell_accepts:
        notifications._windows_balloon("A.C.E.", "CPU: precisa de atenção.")
        sleep.assert_called_once()
    else:
        with pytest.raises(OSError):
            notifications._windows_balloon("A.C.E.", "CPU: precisa de atenção.")
        sleep.assert_not_called()

    assert [item[0] for item in emitted] == [0, 1, 2]
    assert emitted[1] == (1, 0x10, "A.C.E.", "CPU: precisa de atenção.")
    user32.DestroyWindow.assert_called_once_with(123)


def test_background_monitor_notice_respects_notification_setting(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(
        notifications,
        "notify_monitor_event",
        lambda notice: seen.append(notice) or True,
    )

    assert monitor_daemon._notify_monitor_notice("CPU alto.", enabled=False) is True
    assert seen == []
    assert monitor_daemon._notify_monitor_notice("CPU alto.", enabled=True) is True
    assert seen == ["CPU alto."]
