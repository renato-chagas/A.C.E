"""SkillManageTool — create, list, load, or delete agent-authored skills."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List

from openjarvis.core.paths import get_config_dir
from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.security.file_utils import secure_write_text
from openjarvis.tools._stubs import BaseTool, ToolSpec

_VALID_SKILL_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_VALID_STEP_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,99}$")
_MAX_SKILL_STEPS = 25
_MAX_TEMPLATE_CHARS = 8_000


@ToolRegistry.register("skill_manage")
class SkillManageTool(BaseTool):
    """Manage agent-authored procedural skills."""

    def __init__(self, skills_dir: Path | str | None = None) -> None:
        if skills_dir is None:
            skills_dir = get_config_dir() / "skills"
        self._skills_dir = Path(skills_dir).expanduser()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="skill_manage",
            description="Create, list, load, or delete agent-authored skills.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "load", "delete"],
                        "description": "Action to perform.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Skill name (for create/load/delete).",
                    },
                    "description": {
                        "type": "string",
                        "description": "Skill description (for create).",
                    },
                    "steps": {
                        "type": "array",
                        "description": (
                            "List of workflow steps. Each step has exactly one "
                            "tool_name or skill_name, with optional "
                            "arguments_template and output_key."
                        ),
                    },
                },
                "required": ["action"],
            },
            category="skill",
            requires_confirmation=True,
        )

    def execute(self, **params: Any) -> ToolResult:
        action = params.get("action", "list")
        name = params.get("name", "")
        if action == "create":
            return self._create(
                name, params.get("description", ""), params.get("steps", [])
            )
        elif action == "list":
            return self._list()
        elif action == "load":
            return self._load(name)
        elif action == "delete":
            return self._delete(name)
        return ToolResult(
            tool_name=self.spec.name,
            success=False,
            content=f"Unknown action: {action}",
        )

    def _create(self, name: str, description: str, steps: List[dict]) -> ToolResult:
        if not isinstance(name, str) or not _VALID_SKILL_NAME.fullmatch(name):
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content=(
                    "Use a skill name with 1–64 lowercase letters, numbers, "
                    "hyphens, or underscores; it must start with a letter."
                ),
            )
        if not isinstance(description, str) or len(description) > 1_000:
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content="Skill description must be text of at most 1,000 characters.",
            )
        if not isinstance(steps, list) or not steps or len(steps) > _MAX_SKILL_STEPS:
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content=f"Provide between 1 and {_MAX_SKILL_STEPS} workflow steps.",
            )

        normalized_steps: list[dict[str, str]] = []
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                return ToolResult(
                    tool_name=self.spec.name,
                    success=False,
                    content=f"Workflow step {index} must be an object.",
                )
            tool_name = step.get("tool_name", "")
            skill_name = step.get("skill_name", "")
            if not isinstance(tool_name, str) or not isinstance(skill_name, str):
                return ToolResult(
                    tool_name=self.spec.name,
                    success=False,
                    content=f"Workflow step {index} has an invalid target.",
                )
            if bool(tool_name) == bool(skill_name):
                return ToolResult(
                    tool_name=self.spec.name,
                    success=False,
                    content=(
                        f"Workflow step {index} must specify exactly one "
                        "tool_name or skill_name."
                    ),
                )
            target = tool_name or skill_name
            if not _VALID_STEP_NAME.fullmatch(target):
                return ToolResult(
                    tool_name=self.spec.name,
                    success=False,
                    content=f"Workflow step {index} has an invalid target name.",
                )
            arguments = step.get("arguments_template", "{}")
            output_key = step.get("output_key", "")
            if not isinstance(arguments, str) or len(arguments) > _MAX_TEMPLATE_CHARS:
                return ToolResult(
                    tool_name=self.spec.name,
                    success=False,
                    content=(
                        f"Workflow step {index} arguments must be text of at most "
                        f"{_MAX_TEMPLATE_CHARS} characters."
                    ),
                )
            if not isinstance(output_key, str) or (
                output_key and not _VALID_STEP_NAME.fullmatch(output_key)
            ):
                return ToolResult(
                    tool_name=self.spec.name,
                    success=False,
                    content=f"Workflow step {index} has an invalid output_key.",
                )
            normalized_steps.append(
                {
                    "target_key": "tool_name" if tool_name else "skill_name",
                    "target": target,
                    "arguments_template": arguments,
                    "output_key": output_key,
                }
            )

        try:
            self._skills_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content=f"Could not open the configured skills directory: {exc}",
            )
        path = self._skills_dir / f"{name}.toml"
        if path.is_symlink() or path.exists():
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content=(
                    f"Skill '{name}' already exists. Choose another name; "
                    "existing skills are never overwritten."
                ),
            )
        try:
            path_is_scoped = path.parent.resolve() == self._skills_dir.resolve()
        except (OSError, RuntimeError):
            path_is_scoped = False
        if not path_is_scoped:
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content="Skill path is outside the configured skills directory.",
            )

        def quote(value: str) -> str:
            return json.dumps(value, ensure_ascii=False)

        lines = [
            "[skill]",
            f"name = {quote(name)}",
            f"description = {quote(description)}",
            "",
        ]
        for step in normalized_steps:
            lines.append("[[skill.steps]]")
            lines.append(f"{step['target_key']} = {quote(step['target'])}")
            lines.append(f"arguments_template = {quote(step['arguments_template'])}")
            if step["output_key"]:
                lines.append(f"output_key = {quote(step['output_key'])}")
            lines.append("")
        try:
            secure_write_text(path, "\n".join(lines) + "\n")
        except OSError as exc:
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content=f"Could not save skill '{name}': {exc}",
            )
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            content=f"Created skill: {name}",
        )

    def _list(self) -> ToolResult:
        if not self._skills_dir.exists():
            return ToolResult(
                tool_name=self.spec.name,
                success=True,
                content="No skills directory found.",
            )
        skills = []
        for f in sorted(self._skills_dir.glob("*.toml")):
            skills.append(f.stem)
        if not skills:
            return ToolResult(
                tool_name=self.spec.name,
                success=True,
                content="No skills found.",
            )
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            content="Available skills:\n" + "\n".join(f"- {s}" for s in skills),
        )

    def _load(self, name: str) -> ToolResult:
        path = self._skill_path(name)
        if path is None or not path.is_file():
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content=f"Skill not found: {name}",
            )
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            content=path.read_text(encoding="utf-8"),
        )

    def _delete(self, name: str) -> ToolResult:
        path = self._skill_path(name)
        if path is None or not path.is_file():
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content=f"Skill not found: {name}",
            )
        path.unlink()
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            content=f"Deleted skill: {name}",
        )

    def _skill_path(self, name: str) -> Path | None:
        if not isinstance(name, str) or not _VALID_SKILL_NAME.fullmatch(name):
            return None
        path = self._skills_dir / f"{name}.toml"
        if path.is_symlink():
            return None
        try:
            base = self._skills_dir.resolve()
            if path.resolve().parent != base:
                return None
        except (OSError, RuntimeError):
            return None
        return path
