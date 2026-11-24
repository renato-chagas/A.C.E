"""Create validated, review-only proposals for changes to source code."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.learning.code_improvement import (
    CodeImprovementManager,
    discover_related_python_files,
    generate_local_patch,
)
from openjarvis.tools._stubs import BaseTool, ToolSpec

_MAX_REVIEW_DIFF_CHARS = 12_000


@ToolRegistry.register("code_improvement_propose")
class CodeImprovementProposeTool(BaseTool):
    """Generate and validate a local patch without applying it."""

    def __init__(self, manager: CodeImprovementManager | None = None) -> None:
        self._manager = manager

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="code_improvement_propose",
            description=(
                "Prepare a local-model patch for explicitly selected repository "
                "files. The patch is syntax-validated in a disposable copy and "
                "saved for review. This tool never changes the active checkout; "
                "the user must review and separately approve application."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Describe one concrete code improvement.",
                    },
                    "repository": {
                        "type": "string",
                        "description": "Absolute path to the target Git repository.",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 24,
                        "description": (
                            "Repository-relative source files the local model may "
                            "inspect. Repeat paths only when needed."
                        ),
                    },
                    "discover_related": {
                        "type": "boolean",
                        "description": (
                            "Include up to eight nearby Python modules by static "
                            "imports. Defaults to false."
                        ),
                    },
                },
                "required": ["task", "repository", "files"],
            },
            category="learning",
            cost_estimate=0.0,
            latency_estimate=45.0,
            requires_confirmation=True,
            timeout_seconds=300.0,
            required_capabilities=["file:read", "memory:write"],
        )

    def execute(self, **params: Any) -> ToolResult:
        started = time.monotonic()
        task = params.get("task")
        repository = params.get("repository")
        files = params.get("files")
        discover_related = params.get("discover_related", False)
        if not isinstance(task, str) or not isinstance(repository, str):
            return self._failure(
                "Provide a text task and an explicit repository path.", started
            )
        if not repository.strip() or not Path(repository).is_absolute():
            return self._failure(
                "Repository must be an explicit absolute path.", started
            )
        if (
            not isinstance(files, list)
            or not files
            or any(not isinstance(path, str) for path in files)
        ):
            return self._failure(
                "Provide a list of repository-relative source file paths.", started
            )
        if not isinstance(discover_related, bool):
            return self._failure("discover_related must be a boolean.", started)

        root = Path(repository).expanduser().resolve()
        selected_files = tuple(dict.fromkeys(files))
        if discover_related:
            try:
                related = discover_related_python_files(root, selected_files)
            except (OSError, RuntimeError, ValueError) as exc:
                return self._failure(f"Related-file discovery failed: {exc}", started)
            selected_files = tuple(dict.fromkeys((*selected_files, *related)))

        try:
            manager = self._manager or CodeImprovementManager()
            patch, source_hashes = generate_local_patch(task, selected_files, root)
            proposal = manager.propose(
                root,
                task,
                patch,
                expected_source_hashes=source_hashes,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return self._failure(f"Proposal rejected: {exc}", started)

        try:
            proposal = manager.validate(proposal.proposal_id)
        except (OSError, RuntimeError, ValueError) as exc:
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content=(
                    f"Proposal {proposal.proposal_id} was saved but did not pass "
                    f"validation: {exc}\nIt was not applied. Review with: "
                    f"jarvis improve show {proposal.proposal_id}"
                ),
                latency_seconds=time.monotonic() - started,
                metadata={
                    "proposal_id": proposal.proposal_id,
                    "status": proposal.status,
                    "files": list(proposal.files),
                },
            )

        detail = (
            f"Proposal {proposal.proposal_id} is {proposal.status}.\n"
            f"Files: {', '.join(proposal.files)}\n"
            f"Validation: {proposal.validation}\n"
            "The active repository was not changed. Review the exact diff with "
            f"`jarvis improve show {proposal.proposal_id}`. Apply it only after "
            "review with `jarvis improve apply "
            f"{proposal.proposal_id}`."
        )
        if len(proposal.patch) <= _MAX_REVIEW_DIFF_CHARS:
            detail += f"\n\nProposed diff:\n{proposal.patch}"
        else:
            detail += (
                "\nThe diff is too long to include here; the show command prints "
                "the full proposal."
            )
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            content=detail,
            latency_seconds=time.monotonic() - started,
            metadata={
                "proposal_id": proposal.proposal_id,
                "status": proposal.status,
                "files": list(proposal.files),
            },
        )

    def _failure(self, message: str, started: float) -> ToolResult:
        return ToolResult(
            tool_name=self.spec.name,
            success=False,
            content=message,
            latency_seconds=time.monotonic() - started,
        )


__all__ = ["CodeImprovementProposeTool"]
