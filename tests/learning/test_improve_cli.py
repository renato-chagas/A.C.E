from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

from click.testing import CliRunner

from openjarvis.cli import improve_cmd
from openjarvis.learning.code_improvement import CodeImprovementProposal


def _proposal() -> CodeImprovementProposal:
    return CodeImprovementProposal(
        proposal_id="abc123def456",
        title="Fix parser boundary",
        repository="C:/work/OpenJarvis",
        patch="--- a/src/parser.py\n+++ b/src/parser.py\n@@ -1 +1 @@\n-old\n+new\n",
        status="validated",
        created_at=1.0,
        files=("src/parser.py",),
        validation="Python syntax checked; no project code executed.",
        validation_receipt="a" * 64,
    )


def test_apply_displays_validation_and_exact_diff_before_confirmation(
    monkeypatch,
) -> None:
    manager = MagicMock()
    manager.get.return_value = _proposal()
    monkeypatch.setattr(improve_cmd, "CodeImprovementManager", lambda: manager)

    result = CliRunner().invoke(
        improve_cmd.apply,
        ["abc123def456"],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    assert "Python syntax checked; no project code executed." in result.output
    assert "Validation receipt: " + "a" * 64 in result.output
    assert "Proposed diff:" in result.output
    assert "+new" in result.output
    assert "Approve and apply this exact diff?" in result.output
    manager.approve.assert_not_called()
    manager.apply.assert_not_called()


def test_apply_mutates_only_after_the_displayed_diff_is_approved(
    monkeypatch,
) -> None:
    manager = MagicMock()
    proposal = _proposal()
    manager.get.return_value = proposal
    manager.approve.return_value = replace(proposal, status="approved")
    manager.apply.return_value = replace(proposal, status="applied")
    monkeypatch.setattr(improve_cmd, "CodeImprovementManager", lambda: manager)

    result = CliRunner().invoke(
        improve_cmd.apply,
        [proposal.proposal_id],
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    assert result.output.index("Proposed diff:") < result.output.index(
        "Approve and apply this exact diff?"
    )
    manager.approve.assert_called_once_with(
        proposal.proposal_id, expected_receipt=proposal.validation_receipt
    )
    manager.apply.assert_called_once_with(
        proposal.proposal_id, expected_receipt=proposal.validation_receipt
    )


def test_apply_can_resume_a_recovered_proposal_after_new_confirmation(
    monkeypatch,
) -> None:
    manager = MagicMock()
    proposal = replace(_proposal(), status="approved")
    manager.get.return_value = proposal
    manager.apply.return_value = replace(proposal, status="applied")
    monkeypatch.setattr(improve_cmd, "CodeImprovementManager", lambda: manager)

    result = CliRunner().invoke(
        improve_cmd.apply,
        [proposal.proposal_id],
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    assert "Confirm retry of this previously approved diff?" in result.output
    assert result.output.index("Proposed diff:") < result.output.index(
        "Confirm retry of this previously approved diff?"
    )
    manager.approve.assert_not_called()
    manager.apply.assert_called_once_with(
        proposal.proposal_id, expected_receipt=proposal.validation_receipt
    )


def test_list_reports_a_preserved_recovery_conflict(monkeypatch) -> None:
    monkeypatch.setattr(
        improve_cmd,
        "CodeImprovementManager",
        MagicMock(
            side_effect=RuntimeError(
                "Recovery paused for proposal abc123def456. "
                "The recovery journal was kept."
            )
        ),
    )

    result = CliRunner().invoke(improve_cmd.list_proposals)

    assert result.exit_code == 1
    assert "Error: Recovery paused for proposal abc123def456." in result.output
    assert "The recovery journal was kept." in result.output


def test_validate_can_select_isolated_frontend_checks(monkeypatch) -> None:
    manager = MagicMock()
    manager.validate.return_value = replace(
        _proposal(),
        validation="Isolated checks passed: frontend npm test and production build.",
    )
    monkeypatch.setattr(improve_cmd, "CodeImprovementManager", lambda: manager)

    result = CliRunner().invoke(
        improve_cmd.validate,
        ["abc123def456", "--frontend"],
    )

    assert result.exit_code == 0, result.output
    assert "frontend npm test and production build" in result.output
    manager.validate.assert_called_once()
    assert manager.validate.call_args.kwargs["run_frontend"] is True


def test_validate_can_select_isolated_rust_checks(monkeypatch) -> None:
    manager = MagicMock()
    manager.validate.return_value = replace(
        _proposal(),
        validation="Isolated checks passed: Rust formatting and workspace crate tests.",
    )
    monkeypatch.setattr(improve_cmd, "CodeImprovementManager", lambda: manager)

    result = CliRunner().invoke(
        improve_cmd.validate,
        ["abc123def456", "--rust"],
    )

    assert result.exit_code == 0, result.output
    assert "Rust formatting and workspace crate tests" in result.output
    manager.validate.assert_called_once()
    assert manager.validate.call_args.kwargs["run_rust"] is True


def test_propose_adds_discovered_related_files_to_context(monkeypatch):
    manager = MagicMock()
    manager.propose.return_value = _proposal()
    patch_text = "--- a/src/parser.py\n+++ b/src/parser.py\n"
    source_hashes = {"src/parser.py": "a" * 64, "tests/test_parser.py": "b" * 64}
    local_patch = MagicMock(return_value=(patch_text, source_hashes))
    monkeypatch.setattr(improve_cmd, "CodeImprovementManager", lambda: manager)
    monkeypatch.setattr(
        improve_cmd,
        "discover_related_python_files",
        lambda _root, files: ("tests/test_parser.py",),
    )
    monkeypatch.setattr(improve_cmd, "_local_patch", local_patch)

    result = CliRunner().invoke(
        improve_cmd.propose,
        [
            "--task",
            "Fix parser",
            "--file",
            "src/parser.py",
            "--file",
            "src/parser.py",
            "--discover-related",
            "--repo",
            "C:/work",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Related files added: tests/test_parser.py" in result.output
    assert local_patch.call_args.args[0] == "Fix parser"
    assert local_patch.call_args.args[1] == (
        "src/parser.py",
        "tests/test_parser.py",
    )
    manager.propose.assert_called_once()
    assert manager.propose.call_args.args[0].name == "work"
    assert manager.propose.call_args.args[1:] == ("Fix parser", patch_text)
    assert manager.propose.call_args.kwargs == {"expected_source_hashes": source_hashes}
