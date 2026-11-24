from __future__ import annotations

from pathlib import Path

import pytest
import tomllib


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    return d


def test_skill_create(skills_dir: Path):
    from openjarvis.tools.skill_manage import SkillManageTool

    tool = SkillManageTool(skills_dir=skills_dir)
    result = tool.execute(
        action="create",
        name="api_health",
        description="Check API health",
        steps=[
            {
                "tool_name": "http_request",
                "arguments_template": '{"url": "{endpoint}/health"}',
            }
        ],
    )
    assert result.success
    assert (skills_dir / "api_health.toml").exists()


def test_skill_list(skills_dir: Path):
    from openjarvis.tools.skill_manage import SkillManageTool

    tool = SkillManageTool(skills_dir=skills_dir)
    tool.execute(
        action="create",
        name="skill_a",
        description="Skill A",
        steps=[{"tool_name": "calculator"}],
    )
    tool.execute(
        action="create",
        name="skill_b",
        description="Skill B",
        steps=[{"tool_name": "calculator"}],
    )
    result = tool.execute(action="list")
    assert "skill_a" in result.content
    assert "skill_b" in result.content


def test_skill_delete(skills_dir: Path):
    from openjarvis.tools.skill_manage import SkillManageTool

    tool = SkillManageTool(skills_dir=skills_dir)
    tool.execute(
        action="create",
        name="temp_skill",
        description="Temp",
        steps=[{"tool_name": "calculator"}],
    )
    assert (skills_dir / "temp_skill.toml").exists()
    result = tool.execute(action="delete", name="temp_skill")
    assert result.success
    assert not (skills_dir / "temp_skill.toml").exists()


def test_skill_load(skills_dir: Path):
    from openjarvis.tools.skill_manage import SkillManageTool

    tool = SkillManageTool(skills_dir=skills_dir)
    tool.execute(
        action="create",
        name="my_skill",
        description="My skill desc",
        steps=[
            {
                "tool_name": "web_search",
                "arguments_template": '{"q": "test"}',
            }
        ],
    )
    result = tool.execute(action="load", name="my_skill")
    assert "web_search" in result.content
    assert "My skill desc" in result.content


def test_skill_create_escapes_toml_and_requires_confirmation(skills_dir: Path):
    from openjarvis.tools.skill_manage import SkillManageTool

    tool = SkillManageTool(skills_dir=skills_dir)
    description = 'Say "hello"\nthen continue'
    result = tool.execute(
        action="create",
        name="escaped_skill",
        description=description,
        steps=[
            {
                "tool_name": "calculator",
                "arguments_template": '{"expression": "{value}"}',
                "output_key": "result",
            }
        ],
    )

    assert result.success
    manifest = tomllib.loads(
        (skills_dir / "escaped_skill.toml").read_text(encoding="utf-8")
    )
    assert manifest["skill"]["description"] == description
    assert manifest["skill"]["steps"][0]["output_key"] == "result"
    assert tool.spec.requires_confirmation is True


def test_skill_create_rejects_traversal_and_overwrite(skills_dir: Path):
    from openjarvis.tools.skill_manage import SkillManageTool

    tool = SkillManageTool(skills_dir=skills_dir)
    step = [{"tool_name": "calculator"}]

    traversal = tool.execute(
        action="create", name="../outside", description="bad", steps=step
    )
    assert not traversal.success
    assert not (skills_dir.parent / "outside.toml").exists()

    first = tool.execute(
        action="create", name="kept_skill", description="original", steps=step
    )
    assert first.success
    original = (skills_dir / "kept_skill.toml").read_text(encoding="utf-8")
    overwrite = tool.execute(
        action="create", name="kept_skill", description="replacement", steps=step
    )
    assert not overwrite.success
    assert (skills_dir / "kept_skill.toml").read_text(encoding="utf-8") == original
