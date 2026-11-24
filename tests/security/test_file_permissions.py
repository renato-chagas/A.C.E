"""Tests for secure file creation helpers (Section 4)."""

from __future__ import annotations

import csv
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest


def _assert_windows_owner_only_acl(path: Path, tmp_path: Path) -> None:
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    icacls = system_root / "System32" / "icacls.exe"
    whoami = system_root / "System32" / "whoami.exe"
    saved_acl = tmp_path / f"{path.name}.acl"
    try:
        subprocess.run(
            [str(icacls), str(path), "/save", str(saved_acl), "/c"],
            check=True,
            capture_output=True,
            text=True,
        )
        descriptor = saved_acl.read_text(encoding="utf-16-le")
    finally:
        saved_acl.unlink(missing_ok=True)

    identity = subprocess.run(
        [str(whoami), "/user", "/fo", "csv", "/nh"],
        check=True,
        capture_output=True,
        text=True,
    )
    current_sid = next(csv.reader(identity.stdout.splitlines()))[1]
    access_entries = re.findall(r"\([^()]*\)", descriptor)
    expected = (
        f"(A;OICI;FA;;;{current_sid})"
        if path.is_dir()
        else f"(A;;FA;;;{current_sid})"
    )
    assert access_entries == [expected]


class TestSecureMkdir:
    """secure_mkdir should create directories with 0o700."""

    def test_creates_directory_with_700(self) -> None:
        from openjarvis.security.file_utils import secure_mkdir

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "secure_dir"
            result = secure_mkdir(target)
            assert result.is_dir()
            if os.name == "nt":
                _assert_windows_owner_only_acl(target, Path(tmp))
            else:
                mode = stat.S_IMODE(os.stat(target).st_mode)
                assert mode == 0o700

    def test_creates_parent_directories(self) -> None:
        from openjarvis.security.file_utils import secure_mkdir

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a" / "b" / "c"
            result = secure_mkdir(target)
            assert result.is_dir()

    def test_existing_directory_gets_permission_fix(self) -> None:
        from openjarvis.security.file_utils import secure_mkdir

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "existing"
            target.mkdir(mode=0o755)
            secure_mkdir(target)
            if os.name == "nt":
                _assert_windows_owner_only_acl(target, Path(tmp))
            else:
                mode = stat.S_IMODE(os.stat(target).st_mode)
                assert mode == 0o700


class TestSecureCreate:
    """secure_create should create files with 0o600."""

    def test_creates_file_with_600(self) -> None:
        from openjarvis.security.file_utils import secure_create

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "secure_file.db"
            result = secure_create(target)
            assert result.exists()
            if os.name == "nt":
                _assert_windows_owner_only_acl(target, Path(tmp))
            else:
                mode = stat.S_IMODE(os.stat(target).st_mode)
                assert mode == 0o600

    def test_existing_file_gets_permission_fix(self) -> None:
        from openjarvis.security.file_utils import secure_create

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "existing.db"
            target.write_text("data")
            os.chmod(target, 0o644)
            secure_create(target)
            if os.name == "nt":
                _assert_windows_owner_only_acl(target, Path(tmp))
            else:
                mode = stat.S_IMODE(os.stat(target).st_mode)
                assert mode == 0o600

    def test_creates_parent_directory_with_700(self) -> None:
        from openjarvis.security.file_utils import secure_create

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sub" / "file.db"
            secure_create(target)
            if os.name == "nt":
                _assert_windows_owner_only_acl(target.parent, Path(tmp))
            else:
                parent_mode = stat.S_IMODE(os.stat(target.parent).st_mode)
                assert parent_mode == 0o700


@pytest.mark.skipif(os.name != "nt", reason="Windows ACL integration")
def test_ace_persistent_stores_use_owner_only_windows_acls(tmp_path: Path) -> None:
    from openjarvis.agents.manager import AgentManager
    from openjarvis.learning.code_improvement import CodeImprovementManager
    from openjarvis.memory.episodes import EpisodeStore
    from openjarvis.memory.store import LocalFactStore
    from openjarvis.proactive.local_monitor import LocalMonitorStore
    from openjarvis.scheduler.store import SchedulerStore
    from openjarvis.sessions.session import SessionStore
    from openjarvis.tools.approval_store import ApprovalStore
    from openjarvis.tools.storage.knowledge_graph import KnowledgeGraphMemory
    from openjarvis.tools.storage.sqlite import SQLiteMemory

    facts_path = tmp_path / "memory_facts.jsonl"
    facts = LocalFactStore(facts_path)
    facts.add("ACE facts remain local")

    episodes = EpisodeStore(tmp_path / "episodes.db")
    graph = KnowledgeGraphMemory(tmp_path / "knowledge_graph.db")
    documents = SQLiteMemory(tmp_path / "knowledge.db")
    LocalMonitorStore(tmp_path / "proactive_events.db")
    approvals = ApprovalStore(str(tmp_path / "approvals.db"))
    CodeImprovementManager(tmp_path / "improvements.db")
    agents = AgentManager(str(tmp_path / "agents.db"))
    scheduler = SchedulerStore(tmp_path / "scheduled_tasks.db")
    sessions = SessionStore(tmp_path / "sessions.db")

    try:
        for path in (
            facts_path,
            facts_path.with_suffix(".jsonl.lock"),
            tmp_path / "episodes.db",
            tmp_path / "knowledge_graph.db",
            tmp_path / "knowledge.db",
            tmp_path / "proactive_events.db",
            tmp_path / "approvals.db",
            tmp_path / "improvements.db",
            tmp_path / "agents.db",
            tmp_path / "scheduled_tasks.db",
            tmp_path / "sessions.db",
        ):
            _assert_windows_owner_only_acl(path, tmp_path)
    finally:
        episodes.close()
        graph.close()
        documents.close()
        approvals.close()
        agents.close()
        scheduler.close()
        sessions.close()
