"""Secure file and directory creation helpers.

All OpenJarvis data files under ``~/.openjarvis/`` should be created
through these helpers to ensure consistent, restrictive permissions.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _set_owner_only_windows_acl(path: Path, *, directory: bool) -> None:
    """Replace *path*'s DACL with full access for the current user only.

    ``chmod(0o600/0o700)`` does not restrict other Windows accounts, so
    sensitive local stores need an explicit, protected NTFS DACL. The ACE is
    inheritable for directories so files created beneath them remain private.
    """
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _SidAndAttributes(ctypes.Structure):
        _fields_ = [("sid", ctypes.c_void_p), ("attributes", wintypes.DWORD)]

    class _TokenUser(ctypes.Structure):
        _fields_ = [("user", _SidAndAttributes)]

    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.InitializeAcl.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD]
    advapi32.InitializeAcl.restype = wintypes.BOOL
    advapi32.AddAccessAllowedAceEx.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    advapi32.AddAccessAllowedAceEx.restype = wintypes.BOOL
    advapi32.SetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    advapi32.SetNamedSecurityInfoW.restype = wintypes.DWORD

    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        required = wintypes.DWORD()
        advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(required))
        if not required.value:
            raise ctypes.WinError(ctypes.get_last_error())

        token_data = ctypes.create_string_buffer(required.value)
        if not advapi32.GetTokenInformation(
            token, 1, token_data, required, ctypes.byref(required)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        user_sid = ctypes.cast(
            token_data, ctypes.POINTER(_TokenUser)
        ).contents.user.sid

        acl = ctypes.create_string_buffer(1024)
        if not advapi32.InitializeAcl(acl, len(acl), 2):
            raise ctypes.WinError(ctypes.get_last_error())

        # FILE_ALL_ACCESS matches the owner's ability to manage its own
        # private files. Child objects inherit this ACE from private dirs.
        inheritance = 0x3 if directory else 0  # OBJECT_INHERIT | CONTAINER_INHERIT
        if not advapi32.AddAccessAllowedAceEx(
            acl, 2, inheritance, 0x001F01FF, user_sid
        ):
            raise ctypes.WinError(ctypes.get_last_error())

        # SE_FILE_OBJECT | DACL_SECURITY_INFORMATION |
        # PROTECTED_DACL_SECURITY_INFORMATION. A protected DACL prevents
        # broad ACLs on parent directories from being inherited.
        error = advapi32.SetNamedSecurityInfoW(
            str(path.absolute()),
            1,
            0x00000004 | 0x80000000,
            None,
            None,
            acl,
            None,
        )
        if error:
            raise OSError(error, ctypes.FormatError(error), str(path))
    finally:
        kernel32.CloseHandle(token)


def _set_restrictive_permissions(path: Path, mode: int, *, directory: bool) -> None:
    if os.name == "nt":
        _set_owner_only_windows_acl(path, directory=directory)
    else:
        os.chmod(path, mode)


def secure_mkdir(path: Path, mode: int = 0o700) -> Path:
    """Create a directory with restrictive permissions.

    Creates parent directories as needed, then sets *mode* on the
    target directory (even if it already exists).
    """
    path.mkdir(parents=True, exist_ok=True)
    _set_restrictive_permissions(path, mode, directory=True)
    return path


def secure_create(path: Path, mode: int = 0o600) -> Path:
    """Ensure a file exists with restrictive permissions.

    Creates the parent directory with ``0o700`` if needed, touches the
    file if it doesn't exist, and sets *mode* on it.
    """
    secure_mkdir(path.parent, mode=0o700)
    if not path.exists():
        path.touch()
    _set_restrictive_permissions(path, mode, directory=False)
    return path


def secure_write_text(
    path: Path,
    content: str,
    *,
    mode: int = 0o600,
    encoding: str = "utf-8",
) -> Path:
    """Atomically replace *path* with owner-only text content.

    The temporary file is created in the destination directory so
    :func:`os.replace` remains atomic. Permissions are restricted before
    secret bytes are written, then re-applied after replacement for platforms
    whose replace semantics preserve the destination's prior mode.
    """
    secure_mkdir(path.parent, mode=0o700)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        if os.name == "nt":
            _set_owner_only_windows_acl(temporary_path, directory=False)
        elif hasattr(os, "fchmod"):
            os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _set_restrictive_permissions(path, mode, directory=False)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def secure_write_bytes(path: Path, content: bytes, *, mode: int = 0o600) -> Path:
    """Atomically replace *path* with owner-only binary content.

    The temporary file is created beside the destination so replacement stays
    atomic. Its permissions are restricted before writing any bytes.
    """
    secure_mkdir(path.parent, mode=0o700)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        if os.name == "nt":
            _set_owner_only_windows_acl(temporary_path, directory=False)
        elif hasattr(os, "fchmod"):
            os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _set_restrictive_permissions(path, mode, directory=False)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def secure_write_json(path: Path, payload: Any, *, mode: int = 0o600) -> Path:
    """Atomically persist JSON with restrictive permissions."""
    return secure_write_text(
        path,
        json.dumps(payload, indent=2),
        mode=mode,
        encoding="utf-8",
    )
