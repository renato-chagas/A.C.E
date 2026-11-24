"""Tests for review-gated local source-code improvement proposals."""

from __future__ import annotations

import hashlib
import sqlite3
import stat
import subprocess
import sys
from contextlib import contextmanager
from unittest.mock import patch

import pytest

import openjarvis.learning.code_improvement as code_improvement
from openjarvis.learning.code_improvement import (
    CodeImprovementManager,
    _contained_path,
    _patch_paths,
    discover_related_python_files,
)


def _repository(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "src" / "value.py").write_text("value = 1\n", encoding="utf-8")
    return repo


def _patch(old: str = "value = 1", new: str = "value = 2") -> str:
    return f"--- a/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n-{old}\n+{new}\n"


def test_validation_uses_disposable_copy_and_apply_requires_approval(tmp_path):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    source_hash = hashlib.sha256((repo / "src" / "value.py").read_bytes()).hexdigest()
    proposal = manager.propose(
        repo,
        "Fix value",
        _patch(),
        expected_source_hashes={"src/value.py": source_hash},
    )

    assert (repo / "src" / "value.py").read_text(encoding="utf-8") == "value = 1\n"
    with pytest.raises(ValueError, match="approve"):
        manager.apply(proposal.proposal_id)

    validated = manager.validate(proposal.proposal_id)
    assert validated.status == "validated"
    assert "No project code was executed" in validated.validation
    assert (repo / "src" / "value.py").read_text(encoding="utf-8") == "value = 1\n"

    approved = manager.approve(proposal.proposal_id)
    assert approved.status == "approved"
    applied = manager.apply(proposal.proposal_id)
    assert applied.status == "applied"
    assert (repo / "src" / "value.py").read_text(encoding="utf-8") == "value = 2\n"


def test_validate_rejects_stale_source_without_modifying_it(tmp_path):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    proposal = manager.propose(repo, "Fix value", _patch())
    source = repo / "src" / "value.py"
    source.write_text("value = 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Source changed"):
        manager.validate(proposal.proposal_id)
    assert source.read_text(encoding="utf-8") == "value = 3\n"


def test_approval_rejects_a_new_validation_after_review(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    proposal = manager.propose(repo, "Fix value", _patch())
    reviewed = manager.validate(proposal.proposal_id)
    monkeypatch.setattr(manager, "_run_tests", lambda *_args, **_kw: "tests passed")
    latest = manager.validate(proposal.proposal_id, test_paths=("src",))

    assert latest.validation_receipt != reviewed.validation_receipt
    with pytest.raises(ValueError, match="changed since it was reviewed"):
        manager.approve(
            proposal.proposal_id, expected_receipt=reviewed.validation_receipt
        )
    assert manager.get(proposal.proposal_id).status == "validated"

    manager.approve(proposal.proposal_id, expected_receipt=latest.validation_receipt)
    with pytest.raises(ValueError, match="changed since it was reviewed"):
        manager.apply(
            proposal.proposal_id, expected_receipt=reviewed.validation_receipt
        )
    assert (repo / "src" / "value.py").read_text(encoding="utf-8") == "value = 1\n"


@pytest.mark.parametrize("changed_field", ["patch", "validation", "title"])
def test_approved_receipt_rejects_edited_saved_content(tmp_path, changed_field):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    proposal = manager.propose(repo, "Fix value", _patch())
    manager.validate(proposal.proposal_id)
    manager.approve(proposal.proposal_id)
    with sqlite3.connect(manager.db_path) as db:
        db.execute(
            f"UPDATE code_improvement_proposals SET {changed_field} = ? "
            "WHERE proposal_id = ?",
            (_patch(new="value = 9"), proposal.proposal_id),
        )

    with pytest.raises(ValueError, match="saved validation does not match"):
        manager.apply(proposal.proposal_id)
    assert (repo / "src" / "value.py").read_text(encoding="utf-8") == "value = 1\n"
    refreshed = manager.validate(proposal.proposal_id)
    assert refreshed.status == "validated"
    assert refreshed.validation_receipt


def test_apply_refuses_outputs_different_from_validated_copy(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    proposal = manager.propose(repo, "Fix value", _patch())
    manager.validate(proposal.proposal_id)
    manager.approve(proposal.proposal_id)
    monkeypatch.setattr(
        code_improvement,
        "_expected_postimages",
        lambda _patch, _snapshots: {"src/value.py": None},
    )

    with pytest.raises(ValueError, match="different files from validation"):
        manager.apply(proposal.proposal_id)
    assert manager.get(proposal.proposal_id).status == "approved"
    assert (repo / "src" / "value.py").read_text(encoding="utf-8") == "value = 1\n"


def test_validate_rejects_copy_that_captured_different_source(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    proposal = manager.propose(repo, "Fix value", _patch())
    original_copy = code_improvement._copy_repository

    def copy_with_source_edit(root, snapshot):
        original_copy(root, snapshot)
        (snapshot / "src" / "value.py").write_text("value = 9\n", encoding="utf-8")

    monkeypatch.setattr(code_improvement, "_copy_repository", copy_with_source_edit)
    with pytest.raises(ValueError, match="Source changed"):
        manager.validate(proposal.proposal_id)
    assert manager.get(proposal.proposal_id).status == "proposed"


def test_legacy_approval_requires_a_fresh_validation(tmp_path):
    repo = _repository(tmp_path)
    path = tmp_path / "improvements.db"
    manager = CodeImprovementManager(path)
    proposal = manager.propose(repo, "Fix value", _patch())
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE code_improvement_proposals SET status='approved', validation='old' "
            "WHERE proposal_id=?",
            (proposal.proposal_id,),
        )
        db.execute(
            "ALTER TABLE code_improvement_proposals DROP COLUMN validation_receipt"
        )
        db.execute(
            "ALTER TABLE code_improvement_proposals DROP COLUMN validation_files_json"
        )

    manager = CodeImprovementManager(path)
    with pytest.raises(ValueError, match="saved validation does not match"):
        manager.apply(proposal.proposal_id)
    assert manager.validate(proposal.proposal_id).status == "validated"


def test_propose_rejects_source_changed_after_model_context_was_read(tmp_path):
    repo = _repository(tmp_path)
    old_hash = hashlib.sha256(b"value = 0\n").hexdigest()
    manager = CodeImprovementManager(tmp_path / "improvements.db")

    with pytest.raises(ValueError, match="changed while the local model generated"):
        manager.propose(
            repo,
            "Fix value",
            _patch(),
            expected_source_hashes={"src/value.py": old_hash},
        )


def test_propose_rejects_files_outside_model_context(tmp_path):
    repo = _repository(tmp_path)
    (repo / "src" / "other.py").write_text("other = 1\n", encoding="utf-8")
    selected_hash = hashlib.sha256(b"other = 1\n").hexdigest()
    manager = CodeImprovementManager(tmp_path / "improvements.db")

    with pytest.raises(ValueError, match="not supplied to the local model"):
        manager.propose(
            repo,
            "Fix value",
            _patch(),
            expected_source_hashes={"src/other.py": selected_hash},
        )


def test_validate_does_not_resurrect_a_rejected_proposal(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    proposal = manager.propose(repo, "Fix value", _patch())
    original_syntax_check = code_improvement._syntax_check

    def reject_during_validation(snapshot, files):
        with manager._connect() as db:
            db.execute(
                "UPDATE code_improvement_proposals SET status='rejected' "
                "WHERE proposal_id=?",
                (proposal.proposal_id,),
            )
        return original_syntax_check(snapshot, files)

    monkeypatch.setattr(code_improvement, "_syntax_check", reject_during_validation)
    with pytest.raises(ValueError, match="state changed during validation"):
        manager.validate(proposal.proposal_id)

    assert manager.get(proposal.proposal_id).status == "rejected"


def test_approve_does_not_overwrite_a_concurrent_rejection(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    proposal = manager.propose(repo, "Fix value", _patch())
    manager.validate(proposal.proposal_id)
    original_connect = manager._connect
    connect_calls = 0

    @contextmanager
    def reject_between_status_check_and_update():
        nonlocal connect_calls
        connect_calls += 1
        if connect_calls == 2:
            with original_connect() as competing_db:
                competing_db.execute(
                    "UPDATE code_improvement_proposals SET status='rejected' "
                    "WHERE proposal_id=?",
                    (proposal.proposal_id,),
                )
        with original_connect() as db:
            yield db

    monkeypatch.setattr(manager, "_connect", reject_between_status_check_and_update)
    with pytest.raises(ValueError, match="state changed before approval"):
        manager.approve(proposal.proposal_id)

    assert manager.get(proposal.proposal_id).status == "rejected"


def test_apply_rolls_back_files_if_proposal_state_cannot_be_saved(tmp_path):
    repo = _repository(tmp_path)
    source = repo / "src" / "value.py"
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    proposal = manager.propose(repo, "Fix value", _patch())
    manager.validate(proposal.proposal_id)
    manager.approve(proposal.proposal_id)
    with sqlite3.connect(manager.db_path) as db:
        db.execute(
            """
            CREATE TRIGGER fail_applied_status
            BEFORE UPDATE OF status ON code_improvement_proposals
            WHEN NEW.status = 'applied'
            BEGIN
                SELECT RAISE(ABORT, 'simulated status-write failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="status-write failure"):
        manager.apply(proposal.proposal_id)

    assert source.read_text(encoding="utf-8") == "value = 1\n"
    assert manager.get(proposal.proposal_id).status == "approved"


def _apply_reversible_change(tmp_path):
    repo = _repository(tmp_path)
    (repo / "src" / "deleted.py").write_text("obsolete = True\n", encoding="utf-8")
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    patch_text = (
        _patch() + "--- /dev/null\n"
        "+++ b/src/added.py\n"
        "@@ -0,0 +1 @@\n"
        "+value = 3\n"
        "diff --git a/src/deleted.py b/src/deleted.py\n"
        "deleted file mode 100644\n"
        "--- a/src/deleted.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-obsolete = True\n"
    )
    proposal = manager.propose(repo, "Change, add, and delete files", patch_text)
    originals = {
        relative: ((repo / relative).read_bytes(), (repo / relative).stat().st_mode)
        for relative in ("src/value.py", "src/deleted.py")
    }
    manager.validate(proposal.proposal_id)
    manager.approve(proposal.proposal_id)
    applied = manager.apply(proposal.proposal_id)
    assert applied.status == "applied"
    assert (repo / "src" / "added.py").is_file()
    assert not (repo / "src" / "deleted.py").exists()
    return repo, manager, applied, originals


def test_rollback_restores_exact_originals_and_leaves_unrelated_edits(tmp_path):
    repo, manager, proposal, originals = _apply_reversible_change(tmp_path)
    unrelated = repo / "notes.txt"
    unrelated.write_bytes(b"user changes")
    restarted = CodeImprovementManager(manager.db_path)
    assert restarted.get(proposal.proposal_id).status == "applied"
    assert (repo / "src" / "added.py").is_file()

    rolled_back = restarted.rollback(
        proposal.proposal_id, expected_receipt=proposal.validation_receipt
    )

    assert rolled_back.status == "rolled_back"
    assert not (repo / "src" / "added.py").exists()
    assert unrelated.read_bytes() == b"user changes"
    for relative, (content, mode) in originals.items():
        path = repo / relative
        assert path.read_bytes() == content
        assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(mode)
    with sqlite3.connect(manager.db_path) as db:
        for table in (
            "code_improvement_apply_journals",
            "code_improvement_apply_journal_files",
        ):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)


def test_rollback_neither_accepts_a_stale_receipt_nor_clobbers_a_later_edit(tmp_path):
    repo, manager, proposal, _ = _apply_reversible_change(tmp_path)
    with pytest.raises(ValueError, match="changed since it was reviewed"):
        manager.rollback(proposal.proposal_id, expected_receipt="wrong")
    edited = repo / "src" / "value.py"
    edited.write_bytes(b"user_changes = True\n")
    added = repo / "src" / "added.py"
    original_addition = added.read_bytes()

    with pytest.raises(RuntimeError, match="changed since this proposal was applied"):
        manager.rollback(proposal.proposal_id)

    assert edited.read_bytes() == b"user_changes = True\n"
    assert added.read_bytes() == original_addition
    assert not (repo / "src" / "deleted.py").exists()
    assert manager.get(proposal.proposal_id).status == "applied"


@pytest.mark.parametrize("intervening_edit", [False, True])
def test_confirmed_interrupted_rollback_recovers_only_known_files(
    tmp_path, monkeypatch, intervening_edit
):
    repo, manager, proposal, originals = _apply_reversible_change(tmp_path)
    replace = CodeImprovementManager._replace_with_snapshot
    restored = []

    def interrupt(root, relative, snapshot, expected):
        replace(root, relative, snapshot, expected)
        restored.append(relative)
        raise KeyboardInterrupt

    with monkeypatch.context() as mocks:
        mocks.setattr(
            CodeImprovementManager, "_replace_with_snapshot", staticmethod(interrupt)
        )
        with pytest.raises(KeyboardInterrupt):
            manager.rollback(proposal.proposal_id)

    assert len(restored) == 1
    assert manager.get(proposal.proposal_id).status == "rolling_back"
    if intervening_edit:
        edited = repo / restored[0]
        edited.write_bytes(b"user_changes = True\n")
        before = {
            relative: (repo / relative).read_bytes()
            if (repo / relative).exists()
            else None
            for relative in proposal.files
        }
        with pytest.raises(RuntimeError, match="differ from both known states"):
            CodeImprovementManager(manager.db_path)
        after = {
            relative: (repo / relative).read_bytes()
            if (repo / relative).exists()
            else None
            for relative in proposal.files
        }
        assert after == before
        assert manager.get(proposal.proposal_id).status == "rolling_back"
    else:
        restarted = CodeImprovementManager(manager.db_path)
        assert restarted.get(proposal.proposal_id).status == "rolled_back"
        for relative, (content, _) in originals.items():
            assert (repo / relative).read_bytes() == content
        assert not (repo / "src" / "added.py").exists()


def test_legacy_applied_proposal_without_saved_files_stays_readable(tmp_path):
    repo, manager, proposal, _ = _apply_reversible_change(tmp_path)
    with sqlite3.connect(manager.db_path) as db:
        db.execute("DELETE FROM code_improvement_apply_journals")
        db.execute("DELETE FROM code_improvement_apply_journal_files")
    restarted = CodeImprovementManager(manager.db_path)

    assert restarted.get(proposal.proposal_id).status == "applied"
    with pytest.raises(ValueError, match="has no saved rollback files"):
        restarted.rollback(proposal.proposal_id)

    assert (repo / "src" / "added.py").exists()


def test_cli_rollback_requires_confirmation(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from openjarvis.cli import improve_cmd

    repo, manager, proposal, originals = _apply_reversible_change(tmp_path)
    monkeypatch.setattr(improve_cmd, "_open_manager", lambda: manager)
    runner = CliRunner()
    command = ["rollback", proposal.proposal_id]

    declined = runner.invoke(improve_cmd.improve, command, input="n\n")
    assert declined.exit_code == 0, declined.output
    assert manager.get(proposal.proposal_id).status == "applied"
    assert (repo / "src" / "added.py").exists()
    accepted = runner.invoke(improve_cmd.improve, command, input="y\n")
    assert accepted.exit_code == 0, accepted.output
    assert proposal.validation_receipt in accepted.output
    assert manager.get(proposal.proposal_id).status == "rolled_back"
    assert (repo / "src" / "value.py").read_bytes() == originals["src/value.py"][0]


def _interrupt_multi_file_apply(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    source = repo / "src" / "value.py"
    db_path = tmp_path / "improvements.db"
    manager = CodeImprovementManager(db_path)
    patch_text = (
        "--- a/src/value.py\n"
        "+++ b/src/value.py\n"
        "@@ -1 +1 @@\n"
        "-value = 1\n"
        "+value = 2\n"
        "--- /dev/null\n"
        "+++ b/src/added.py\n"
        "@@ -0,0 +1 @@\n"
        "+value = 3\n"
    )
    proposal = manager.propose(repo, "Update two files", patch_text)
    manager.validate(proposal.proposal_id)
    manager.approve(proposal.proposal_id)
    original_apply = code_improvement._run_git_apply

    def interrupt_after_patch(root, text, *, check):
        original_apply(root, text, check=check)
        if root == repo and not check:
            raise KeyboardInterrupt("simulated process interruption")

    monkeypatch.setattr(code_improvement, "_run_git_apply", interrupt_after_patch)
    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        manager.apply(proposal.proposal_id)

    assert source.read_text(encoding="utf-8") == "value = 2\n"
    assert (repo / "src" / "added.py").exists()
    return repo, db_path, proposal.proposal_id


def test_new_manager_recovers_interrupted_multi_file_apply(tmp_path, monkeypatch):
    repo, db_path, proposal_id = _interrupt_multi_file_apply(tmp_path, monkeypatch)
    source = repo / "src" / "value.py"
    restarted = CodeImprovementManager(db_path)

    assert restarted.get(proposal_id).status == "approved"
    assert source.read_text(encoding="utf-8") == "value = 1\n"
    assert not (repo / "src" / "added.py").exists()
    with sqlite3.connect(db_path) as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM code_improvement_apply_journals"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("relative", ["src/value.py", "src/added.py"])
def test_recovery_preserves_edits_made_after_interrupt(tmp_path, monkeypatch, relative):
    repo, db_path, proposal_id = _interrupt_multi_file_apply(tmp_path, monkeypatch)
    (repo / relative).write_bytes(b"user_edit = 4\n")
    paths = (repo / "src" / "value.py", repo / "src" / "added.py")
    before = {path: path.read_bytes() for path in paths}

    with pytest.raises(RuntimeError, match="Recovery paused.*differ from both"):
        CodeImprovementManager(db_path)

    assert {path: path.read_bytes() for path in paths} == before
    with sqlite3.connect(db_path) as db:
        assert db.execute(
            "SELECT status FROM code_improvement_proposals WHERE proposal_id=?",
            (proposal_id,),
        ).fetchone() == ("applying",)
        (postimages,) = db.execute(
            "SELECT postimages_json FROM code_improvement_apply_journals "
            "WHERE proposal_id=?",
            (proposal_id,),
        ).fetchone()
        assert postimages is not None


def test_old_journal_migration_preserves_unknown_changes(tmp_path, monkeypatch):
    repo, db_path, _ = _interrupt_multi_file_apply(tmp_path, monkeypatch)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "ALTER TABLE code_improvement_apply_journals DROP COLUMN postimages_json"
        )
    paths = (repo / "src" / "value.py", repo / "src" / "added.py")
    before = {path: path.read_bytes() for path in paths}

    with pytest.raises(RuntimeError, match="older recovery journal cannot verify"):
        CodeImprovementManager(db_path)

    assert {path: path.read_bytes() for path in paths} == before
    with sqlite3.connect(db_path) as db:
        assert db.execute(
            "SELECT postimages_json FROM code_improvement_apply_journals"
        ).fetchone() == (None,)


@pytest.mark.parametrize(
    "path",
    [
        "../../outside.py",
        "C:/outside.py",
        ".env",
        ".ENV",
        ".git/config",
        ".GIT/config",
    ],
)
def test_propose_rejects_unsafe_or_sensitive_paths(tmp_path, path):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    patch = f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-before\n+after\n"

    with pytest.raises(ValueError):
        manager.propose(repo, "Unsafe", patch)


@pytest.mark.parametrize(
    "mode_headers",
    [
        ("new file mode 120000",),
        ("old mode 100644", "new mode 120000"),
        ("index 1111111..2222222 160000",),
    ],
)
def test_propose_rejects_symlink_and_submodule_modes(tmp_path, mode_headers):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    patch = (
        "diff --git a/src/link.py b/src/link.py\n"
        + "".join(f"{header}\n" for header in mode_headers)
        + "--- a/src/link.py\n"
        "+++ b/src/link.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+../../outside.py\n"
    )

    with pytest.raises(ValueError, match="symbolic links or submodules"):
        manager.propose(repo, "Unsafe link patch", patch)


def test_contained_path_rejects_internal_symlink_parent(tmp_path):
    repo = _repository(tmp_path)
    real_directory = repo / "src" / "real"
    real_directory.mkdir()
    alias = repo / "src" / "alias"
    if sys.platform == "win32":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(real_directory)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            pytest.skip(f"Directory junction creation is unavailable: {result.stderr}")
    else:
        try:
            alias.symlink_to(real_directory, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"Directory symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link or junction"):
        _contained_path(repo, "src/alias/other.py")


def test_validation_checks_added_python_file_in_copy(tmp_path):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    patch = "--- /dev/null\n+++ b/src/added.py\n@@ -0,0 +1 @@\n+added = True\n"
    proposal = manager.propose(repo, "Add file", patch)

    validated = manager.validate(proposal.proposal_id)

    assert validated.status == "validated"
    assert not (repo / "src" / "added.py").exists()


def test_propose_rejects_patch_with_credential_like_content(tmp_path):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    patch = (
        '--- /dev/null\n+++ b/src/secrets.py\n@@ -0,0 +1 @@\n+password = "hunter2"\n'
    )

    with pytest.raises(ValueError, match="secret scanner"):
        manager.propose(repo, "Add credential", patch)


@pytest.mark.parametrize("control", ["\x1b[31m", "\u202e"])
def test_propose_rejects_terminal_and_bidirectional_controls(tmp_path, control):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    patch = _patch(new=f'value = "{control}2"')

    with pytest.raises(ValueError, match="control characters"):
        manager.propose(repo, "Review-safe patch", patch)


def test_patch_paths_handles_multiple_files_and_dash_prefixed_source_lines():
    patch_text = (
        "--- a/src/value.py\n"
        "+++ b/src/value.py\n"
        "@@ -1,2 +1,2 @@\n"
        "--- old comment\n"
        "+# new comment\n"
        " value = 1\n"
        "--- a/src/other.py\n"
        "+++ b/src/other.py\n"
        "@@ -1 +1 @@\n"
        "-other = 1\n"
        "+other = 2\n"
    )

    assert _patch_paths(patch_text) == ("src/other.py", "src/value.py")


def test_patch_paths_decodes_quoted_git_names_before_containment_checks(tmp_path):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    patch = (
        '--- "a/foo/\\056\\056/\\056\\056/outside.py"\n'
        '+++ "b/foo/\\056\\056/\\056\\056/outside.py"\n'
        "@@ -1 +1 @@\n-old\n+new\n"
    )

    with pytest.raises(ValueError, match="Unsafe patch path"):
        manager.propose(repo, "Reject escaped Git path", patch)


def test_patch_paths_preserves_safe_quoted_names_with_spaces():
    patch_text = "\n".join(
        (
            '--- "a/src/my file.py"',
            '+++ "b/src/my file.py"',
            "@@ -1 +1 @@",
            "-old",
            "+new",
        )
    )

    assert _patch_paths(patch_text) == ("src/my file.py",)

    quoted_utf8 = (
        '--- "a/src/caf\\303\\251.py"\n'
        '+++ "b/src/caf\\303\\251.py"\n'
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    assert _patch_paths(quoted_utf8) == ("src/café.py",)


def test_docker_test_runner_mounts_only_the_copy_and_disables_network(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    completed = subprocess.CompletedProcess([], 0, "2 passed", "")
    with (
        patch(
            "openjarvis.learning.code_improvement.shutil.which",
            return_value="docker",
        ),
        patch(
            "openjarvis.learning.code_improvement.subprocess.run",
            return_value=completed,
        ) as run,
    ):
        output = CodeImprovementManager._run_tests(
            snapshot,
            ("tests",),
            run_frontend=True,
            run_rust=True,
            image="openjarvis-sandbox:latest",
            runtime="docker",
            timeout=60,
        )

    command = run.call_args.args[0]
    assert command[command.index("--network") + 1] == "none"
    assert "--pull=never" in command
    mount = command[command.index("--mount") + 1]
    assert mount == f"type=bind,source={snapshot.resolve()},target=/source,readonly"
    assert "/workspace:rw,exec,nosuid,nodev,size=2g" in command
    assert "OPENAI_API_KEY" not in command
    assert any("cp -a /source/. /workspace/" in arg for arg in command)
    script = command[command.index("-c") + 1]
    assert "npm ci --offline" in script
    assert "npm test" in script
    assert "npm run build" in script
    assert "cargo fmt --all -- --check" in script
    assert (
        "cargo test --workspace --exclude openjarvis-python --locked --offline"
        in script
    )
    assert "/cache:rw,noexec,nosuid,size=2g" in command
    assert output == "2 passed"


def test_frontend_validation_is_recorded_without_touching_checkout(tmp_path):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    proposal = manager.propose(repo, "Fix value", _patch())

    with patch.object(
        CodeImprovementManager,
        "_run_tests",
        return_value="frontend checks passed",
    ) as run_tests:
        validated = manager.validate(proposal.proposal_id, run_frontend=True)

    assert validated.status == "validated"
    assert "frontend npm test and production build" in validated.validation
    assert "frontend checks passed" in validated.validation
    run_tests.assert_called_once()
    assert (repo / "src" / "value.py").read_text(encoding="utf-8") == "value = 1\n"


def test_rust_validation_is_recorded_without_touching_checkout(tmp_path):
    repo = _repository(tmp_path)
    manager = CodeImprovementManager(tmp_path / "improvements.db")
    proposal = manager.propose(repo, "Fix value", _patch())

    with patch.object(
        CodeImprovementManager,
        "_run_tests",
        return_value="Rust workspace checks passed",
    ) as run_tests:
        validated = manager.validate(proposal.proposal_id, run_rust=True)

    assert validated.status == "validated"
    assert "Rust formatting and workspace tests" in validated.validation
    assert "Rust workspace checks passed" in validated.validation
    assert run_tests.call_args.kwargs["run_rust"] is True
    assert (repo / "src" / "value.py").read_text(encoding="utf-8") == "value = 1\n"


def test_discovery_follows_local_imports_and_prioritizes_tests(tmp_path):
    repo = _repository(tmp_path)
    package = repo / "src" / "pkg"
    tests_dir = repo / "tests"
    package.mkdir()
    tests_dir.mkdir()
    (package / "parser.py").write_text(
        "from .helpers import normalize\ndef parse(text): return normalize(text)\n",
        encoding="utf-8",
    )
    (package / "helpers.py").write_text(
        "from .models import Model\ndef normalize(text): return text.strip()\n",
        encoding="utf-8",
    )
    (package / "models.py").write_text("class Model: pass\n", encoding="utf-8")
    (tests_dir / "test_parser.py").write_text(
        "from pkg.parser import parse\n", encoding="utf-8"
    )

    related = discover_related_python_files(repo, ("src/pkg/parser.py",))

    assert related[:2] == ("tests/test_parser.py", "src/pkg/helpers.py")
    assert "src/pkg/models.py" in related


def test_discovery_ignores_case_variant_dependency_directories(tmp_path):
    repo = _repository(tmp_path)
    source = repo / "src" / "main.py"
    source.write_text("import NODE_MODULES.pkg.helper\n", encoding="utf-8")
    dependency = repo / "NODE_MODULES" / "pkg"
    dependency.mkdir(parents=True)
    (dependency / "helper.py").write_text("value = 1\n", encoding="utf-8")

    related = discover_related_python_files(repo, ("src/main.py",))

    assert not any(path.startswith("NODE_MODULES/") for path in related)


def test_discovery_respects_remaining_model_context_budget(tmp_path):
    repo = _repository(tmp_path)
    package = repo / "src" / "pkg"
    package.mkdir()
    prefix = "import pkg.helper\n"
    source = prefix + "#" + "x" * (120_000 - len(prefix) - 2) + "\n"
    assert len(source.encode("utf-8")) == 120_000
    (package / "main.py").write_text(source, encoding="utf-8")
    (package / "helper.py").write_text("value = 1\n", encoding="utf-8")

    related = discover_related_python_files(repo, ("src/pkg/main.py",))

    assert related == ()


def test_discovery_rejects_unsafe_seed_paths(tmp_path):
    repo = _repository(tmp_path)

    with pytest.raises(ValueError, match="Unsafe source path"):
        discover_related_python_files(repo, ("../outside.py",))
