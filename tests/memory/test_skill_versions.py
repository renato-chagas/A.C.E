from __future__ import annotations

import pytest

from openjarvis.memory.skill_versions import (
    list_skill_versions,
    load_skill_version,
    next_patch_version,
    record_skill_version,
    set_skill_version,
    skill_name_from_markdown,
    skill_version_from_markdown,
)

_SKILL_NAME = "ace-procedure-ab1234567890"


def _skill(version: str = "0.1.0", body: str = "First reviewed version.") -> str:
    return (
        "---\n"
        f"name: {_SKILL_NAME}\n"
        f"version: {version}\n"
        "---\n"
        f"\n{body}\n"
    )


def test_versions_are_immutable_ordered_and_reopenable(tmp_path):
    history = tmp_path / "skill_versions"
    first = _skill()
    second = _skill("0.1.1", "Updated reviewed version.")

    record_skill_version(history, _SKILL_NAME, first)
    record_skill_version(history, _SKILL_NAME, first)
    record_skill_version(history, _SKILL_NAME, second)

    versions = list_skill_versions(history, _SKILL_NAME)
    assert [item.version for item in versions] == ["0.1.0", "0.1.1"]
    assert versions[0].sha256 != versions[1].sha256
    assert load_skill_version(history, _SKILL_NAME, "0.1.0") == first
    assert load_skill_version(history, _SKILL_NAME, "0.1.1") == second


def test_version_collision_cannot_overwrite_an_approved_snapshot(tmp_path):
    history = tmp_path / "skill_versions"
    record_skill_version(history, _SKILL_NAME, _skill())

    with pytest.raises(FileExistsError, match="different content"):
        record_skill_version(
            history,
            _SKILL_NAME,
            _skill(body="Different content under the same version."),
        )


def test_semantic_patch_bump_changes_only_frontmatter():
    original = _skill("2.4.9", "Keep this body byte-for-byte.")
    next_version = next_patch_version(original)
    updated = set_skill_version(original, next_version)

    assert next_version == "2.4.10"
    assert skill_version_from_markdown(updated) == "2.4.10"
    assert skill_name_from_markdown(updated) == _SKILL_NAME
    assert updated.split("---\n", 2)[-1] == original.split("---\n", 2)[-1]


@pytest.mark.parametrize(
    "skill_name",
    ["../outside", "unrelated-skill", "ace-procedure-NOTHEX"],
)
def test_history_rejects_unsafe_or_non_ace_names(tmp_path, skill_name):
    with pytest.raises(ValueError, match="ACE procedure"):
        record_skill_version(
            tmp_path / "skill_versions",
            skill_name,
            _skill(),
        )
