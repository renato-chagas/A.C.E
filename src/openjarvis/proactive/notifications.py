"""Local desktop notices for ACE resource incidents; no network delivery."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


def _plain_text(value: str, limit: int) -> str:
    return "".join(char for char in str(value) if char.isprintable())[:limit]


def _notification_message(
    notice: str, *, history_hint: str = "jarvis monitor --history"
) -> str:
    """Return a generic category notice without measured or process details."""
    lowered = notice.casefold()
    if "cpu" in lowered:
        category = "CPU"
    elif "memória" in lowered or "memoria" in lowered:
        category = "Memória disponível"
    elif "disco" in lowered or "espaço" in lowered or "espaco" in lowered:
        category = "Espaço em disco"
    else:
        category = "Recursos do computador"

    recovered = "voltou" in lowered or "normalizou" in lowered
    state = "voltou ao normal" if recovered else "precisa de atenção"
    return f"{category}: {state}. Consulte {history_hint}."


def _notify_linux(title: str, message: str) -> bool:
    command = shutil.which("notify-send")
    if not command:
        return False
    result = subprocess.run(
        [command, "--app-name=A.C.E.", "--expire-time=7000", title, message],
        check=False,
        capture_output=True,
        timeout=3,
    )
    return result.returncode == 0


def _escape_applescript(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _notify_macos(title: str, message: str) -> bool:
    command = shutil.which("osascript")
    if not command:
        return False
    script = (
        f'display notification "{_escape_applescript(message)}" '
        f'with title "{_escape_applescript(title)}"'
    )
    result = subprocess.run(
        [command, "-e", script],
        check=False,
        capture_output=True,
        timeout=3,
    )
    return result.returncode == 0


def _windows_balloon(title: str, message: str) -> None:
    """Show a short Shell_NotifyIcon balloon using only Windows system APIs."""
    import ctypes
    from ctypes import wintypes

    class NotifyIconData(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("hWnd", wintypes.HWND),
            ("uID", wintypes.UINT),
            ("uFlags", wintypes.UINT),
            ("uCallbackMessage", wintypes.UINT),
            ("hIcon", wintypes.HICON),
            ("szTip", wintypes.WCHAR * 128),
            ("dwState", wintypes.DWORD),
            ("dwStateMask", wintypes.DWORD),
            ("szInfo", wintypes.WCHAR * 256),
            ("uTimeout", wintypes.UINT),
            ("szInfoTitle", wintypes.WCHAR * 64),
            ("dwInfoFlags", wintypes.DWORD),
            ("guidItem", ctypes.c_byte * 16),
            ("hBalloonIcon", wintypes.HICON),
        ]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        wintypes.LPVOID,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.DestroyWindow.restype = wintypes.BOOL
    user32.LoadIconW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR]
    user32.LoadIconW.restype = wintypes.HICON
    shell32.Shell_NotifyIconW.argtypes = [
        wintypes.DWORD,
        ctypes.POINTER(NotifyIconData),
    ]
    shell32.Shell_NotifyIconW.restype = wintypes.BOOL

    hwnd = user32.CreateWindowExW(
        0, "STATIC", "A.C.E. local monitor", 0, 0, 0, 0, 0, None, None, None, None
    )
    if not hwnd:
        raise ctypes.WinError(ctypes.get_last_error())

    icon_resource = ctypes.cast(ctypes.c_void_p(32512), wintypes.LPCWSTR)
    icon = user32.LoadIconW(None, icon_resource)
    data = NotifyIconData()
    data.cbSize = ctypes.sizeof(NotifyIconData)
    data.hWnd = hwnd
    data.uID = 1
    data.uFlags = 0x0002 | 0x0004  # NIF_ICON | NIF_TIP
    data.hIcon = icon
    data.szTip = "A.C.E."
    shell_notify_icon = shell32.Shell_NotifyIconW
    added = bool(shell_notify_icon(0, ctypes.byref(data)))  # NIM_ADD
    if not added:
        user32.DestroyWindow(hwnd)
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        data.uFlags = 0x0010  # NIF_INFO
        data.szInfoTitle = _plain_text(title, 63)
        data.szInfo = _plain_text(message, 255)
        data.uTimeout = 7000
        data.dwInfoFlags = 0x0001  # NIIF_INFO
        if not shell_notify_icon(1, ctypes.byref(data)):  # NIM_MODIFY
            raise ctypes.WinError(ctypes.get_last_error())
        # Keep the tray entry alive while Windows displays the balloon.
        time.sleep(8)
    finally:
        shell_notify_icon(2, ctypes.byref(data))  # NIM_DELETE
        user32.DestroyWindow(hwnd)


def _show_windows_balloon(title: str, message: str) -> None:
    try:
        _windows_balloon(title, message)
    except OSError:
        logger.warning(
            "Could not display the local desktop notification", exc_info=True
        )


def _notify_windows(title: str, message: str) -> bool:
    thread = threading.Thread(
        target=_show_windows_balloon,
        args=(title, message),
        name="ace-desktop-notification",
        daemon=True,
    )
    thread.start()
    return True


def notify_monitor_event(
    notice: str, *, history_hint: str = "jarvis monitor --history"
) -> bool:
    """Send or queue a generic notice through the local operating system."""
    title = "A.C.E."
    message = _notification_message(notice, history_hint=history_hint)
    if sys.platform == "win32":
        try:
            return _notify_windows(title, message)
        except Exception:
            logger.warning("Could not start the Windows notification", exc_info=True)
            return False

    notifier: Callable[[str, str], bool] | None
    if sys.platform.startswith("linux"):
        notifier = _notify_linux
    elif sys.platform == "darwin":
        notifier = _notify_macos
    else:
        notifier = None
    if notifier is None:
        return False
    try:
        return notifier(title, message)
    except (OSError, subprocess.SubprocessError):
        logger.warning(
            "Could not display the local desktop notification", exc_info=True
        )
        return False


__all__ = ["notify_monitor_event"]
