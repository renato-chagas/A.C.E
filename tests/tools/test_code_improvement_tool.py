"""Conversational code improvement remains a review-only proposal flow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

from openjarvis.learning.code_improvement import CodeImprovementProposal
from openjarvis.tools import code_improvement as code_improvement_tool_module
from openjarvis.tools.code_improvement import CodeImprovementProposeTool


def _proposal(repository: Path) -> CodeImprovementProposal:
    return CodeImprovementProposal(
        proposal_id="abc123def456",
        title="Fix a parser edge case",
        repository=str(repository),
        patch=(
            "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n"
            "-old\n+new\n"
        ),
        status="proposed",
        created_at=1.0,
        files=("src/app.py",),
    )


def test_proposal_tool_is_confirmation_gated_and_cannot_apply() -> None:
    spec = CodeImprovementProposeTool().spec

    assert spec.requires_confirmation is True
    assert spec.required_capabilities == ["file:read", "memory:write"]
    assert "never changes the active checkout" in spec.description


def test_proposal_tool_creates_validated_review_without_applying(
    tmp_path, monkeypatch
) -> None:
    repository = tmp_path / "repo"
    (repository / "src").mkdir(parents=True)
    proposal = _proposal(repository)
    validated = replace(
        proposal,
        status="validated",
        validation="Patch applied in a disposable copy; Python syntax checked.",
    )
    manager = MagicMock()
    manager.propose.return_value = proposal
    manager.validate.return_value = validated
    generate = MagicMock(return_value=(proposal.patch, {"src/app.py": "a" * 64}))
    monkeypatch.setattr(
        code_improvement_tool_module,
        "generate_local_patch",
        generate,
    )

    result = CodeImprovementProposeTool(manager=manager).execute(
        task="Fix a parser edge case",
        repository=str(repository),
        files=["src/app.py"],
    )

    assert result.success is True
    assert result.metadata == {
        "proposal_id": "abc123def456",
        "status": "validated",
        "files": ["src/app.py"],
    }
    assert proposal.patch in result.content
    assert "jarvis improve apply abc123def456" in result.content
    generate.assert_called_once_with(
        "Fix a parser edge case",
        ("src/app.py",),
        repository.resolve(),
    )
    manager.validate.assert_called_once_with("abc123def456")
    manager.apply.assert_not_called()


def test_proposal_tool_rejects_unscoped_files_before_model_call(
    tmp_path, monkeypatch
) -> None:
    manager = MagicMock()
    generate = MagicMock()
    monkeypatch.setattr(
        code_improvement_tool_module,
        "generate_local_patch",
        generate,
    )

    result = CodeImprovementProposeTool(manager=manager).execute(
        task="Fix a parser edge case",
        repository=str(tmp_path),
        files="src/app.py",
    )

    assert result.success is False
    assert "repository-relative source file paths" in result.content
    generate.assert_not_called()
    manager.propose.assert_not_called()
