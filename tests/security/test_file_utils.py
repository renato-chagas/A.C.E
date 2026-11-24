"""Security invariants for atomic private-file persistence."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from openjarvis.security.file_utils import secure_write_bytes, secure_write_json


def test_secure_write_json_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"

    secure_write_json(path, {"access_token": "secret"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"access_token": "secret"}
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "nt", reason="Windows ACL check")
def test_secure_write_json_replaces_inherited_windows_acl(tmp_path: Path) -> None:
    path = tmp_path / "private.json"
    secure_write_json(path, {"access_token": "secret"})

    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    saved_acl = tmp_path / "private.json.acl"
    try:
        subprocess.run(
            [
                str(system_root / "System32" / "icacls.exe"),
                str(path),
                "/save",
                str(saved_acl),
                "/c",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        descriptor = saved_acl.read_text(encoding="utf-16-le")
    finally:
        saved_acl.unlink(missing_ok=True)

    access_entries = re.findall(r"\([^()]*\)", descriptor)
    assert len(access_entries) == 1
    assert ";FA;;;" in access_entries[0]


def test_secure_write_json_preserves_old_file_when_replace_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "credentials.json"
    secure_write_json(path, {"access_token": "old"})

    with (
        patch(
            "openjarvis.security.file_utils.os.replace",
            side_effect=OSError("simulated replace failure"),
        ),
        pytest.raises(OSError, match="simulated replace failure"),
    ):
        secure_write_json(path, {"access_token": "new"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"access_token": "old"}
    assert list(tmp_path.glob(".credentials.json.*.tmp")) == []


def test_secure_write_bytes_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "attachment.bin"

    secure_write_bytes(path, b"private binary content")

    assert path.read_bytes() == b"private binary content"
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "nt", reason="Windows ACL check")
def test_secure_write_bytes_replaces_inherited_windows_acl(tmp_path: Path) -> None:
    path = tmp_path / "private.bin"
    secure_write_bytes(path, b"secret")

    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    saved_acl = tmp_path / "private.bin.acl"
    try:
        subprocess.run(
            [
                str(system_root / "System32" / "icacls.exe"),
                str(path),
                "/save",
                str(saved_acl),
                "/c",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        descriptor = saved_acl.read_text(encoding="utf-16-le")
    finally:
        saved_acl.unlink(missing_ok=True)

    access_entries = re.findall(r"\([^()]*\)", descriptor)
    assert len(access_entries) == 1
    assert ";FA;;;" in access_entries[0]


def test_secure_write_bytes_preserves_old_file_when_replace_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attachment.bin"
    secure_write_bytes(path, b"old")

    with (
        patch(
            "openjarvis.security.file_utils.os.replace",
            side_effect=OSError("simulated replace failure"),
        ),
        pytest.raises(OSError, match="simulated replace failure"),
    ):
        secure_write_bytes(path, b"new")

    assert path.read_bytes() == b"old"
    assert list(tmp_path.glob(".attachment.bin.*.tmp")) == []
