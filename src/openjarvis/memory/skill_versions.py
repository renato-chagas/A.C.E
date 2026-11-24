"""Immutable, local history for explicitly approved procedure skills."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from openjarvis.security.file_utils import secure_create, secure_mkdir

_SKILL_NAME = re.compile(r"^ace-procedure-[a-f0-9]{8,16}$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_FRONTMATTER = re.compile(r"\A---\r?\n(?P<front>.*?)\r?\n---(?:\r?\n|$)", re.DOTALL)
_VERSION_LINE = re.compile(r"(?m)^version:[ \t]*[\"']?(\d+\.\d+\.\d+)[\"']?[ \t]*$")
_NAME_LINE = re.compile(r"(?m)^name:[ \t]*([a-z][a-z0-9_-]{0,63})[ \t]*$")


@dataclass(frozen=True, slots=True)
class SkillVersionSnapshot:
    """A validated archived version of one approved ACE procedure skill."""

    version: str
    path: Path
    sha256: str
    modified_at: float


def validate_skill_name(skill_name: str) -> str:
    """Return a safe ACE procedure skill name or reject it."""
    if not isinstance(skill_name, str) or not _SKILL_NAME.fullmatch(skill_name):
        raise ValueError("Only ACE procedure skill names can be versioned.")
    return skill_name


def skill_name_from_markdown(markdown: str) -> str:
    """Read and validate the name from a skill's YAML frontmatter."""
    frontmatter = _FRONTMATTER.match(markdown or "")
    if frontmatter is None:
        raise ValueError("Skill is missing valid YAML frontmatter.")
    match = _NAME_LINE.search(frontmatter.group("front"))
    if match is None:
        raise ValueError("Skill frontmatter has no valid name.")
    return validate_skill_name(match.group(1))


def skill_version_from_markdown(markdown: str) -> str:
    """Read a strict semantic version from skill frontmatter."""
    frontmatter = _FRONTMATTER.match(markdown or "")
    if frontmatter is None:
        raise ValueError("Skill is missing valid YAML frontmatter.")
    match = _VERSION_LINE.search(frontmatter.group("front"))
    if match is None or not _VERSION.fullmatch(match.group(1)):
        raise ValueError("Skill frontmatter has no semantic version.")
    return match.group(1)


def set_skill_version(markdown: str, version: str) -> str:
    """Replace only the version field in a skill's frontmatter."""
    if not _VERSION.fullmatch(version):
        raise ValueError("Skill version must use MAJOR.MINOR.PATCH.")
    skill_version_from_markdown(markdown)
    match = _VERSION_LINE.search(markdown)
    if match is None:
        raise ValueError("Skill frontmatter has no semantic version.")
    return markdown[: match.start()] + f"version: {version}" + markdown[match.end() :]


def next_patch_version(markdown: str) -> str:
    """Return the next patch version without changing the skill content."""
    major, minor, patch = (
        int(part) for part in skill_version_from_markdown(markdown).split(".")
    )
    return f"{major}.{minor}.{patch + 1}"


def _history_directory(history_root: str | Path, skill_name: str) -> Path:
    validate_skill_name(skill_name)
    root = Path(history_root).expanduser()
    if root.is_symlink():
        raise ValueError("Skill history root may not be a symbolic link.")
    secure_mkdir(root)
    resolved_root = root.resolve(strict=True)
    directory = resolved_root / skill_name
    if directory.is_symlink():
        raise ValueError("Skill history directory may not be a symbolic link.")
    secure_mkdir(directory)
    if directory.resolve(strict=True).parent != resolved_root:
        raise ValueError("Skill history path escaped its configured directory.")
    return directory


def record_skill_version(
    history_root: str | Path,
    skill_name: str,
    markdown: str,
) -> SkillVersionSnapshot:
    """Store an immutable version snapshot; identical retries are idempotent."""
    validate_skill_name(skill_name)
    if skill_name_from_markdown(markdown) != skill_name:
        raise ValueError("Skill name does not match the version history key.")
    version = skill_version_from_markdown(markdown)
    directory = _history_directory(history_root, skill_name)
    path = directory / f"{version}.md"
    if path.is_symlink():
        raise ValueError("Skill version snapshots may not be symbolic links.")
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(markdown)
            handle.flush()
            os.fsync(handle.fileno())
        secure_create(path, mode=0o600)
    except FileExistsError:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Existing skill version snapshot is not a regular file.")
        if path.read_text(encoding="utf-8") != markdown:
            raise FileExistsError(
                f"Version {version} already exists with different content."
            )
    digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    return SkillVersionSnapshot(
        version=version,
        path=path,
        sha256=digest,
        modified_at=path.stat().st_mtime,
    )


def list_skill_versions(
    history_root: str | Path,
    skill_name: str,
) -> list[SkillVersionSnapshot]:
    """List validated immutable snapshots from oldest to newest."""
    validate_skill_name(skill_name)
    root = Path(history_root).expanduser()
    if root.is_symlink():
        raise ValueError("Skill history root may not be a symbolic link.")
    if not root.exists():
        return []
    directory = root.resolve(strict=True) / skill_name
    if directory.is_symlink():
        raise ValueError("Skill history directory may not be a symbolic link.")
    if not directory.exists():
        return []
    if directory.resolve(strict=True).parent != root.resolve(strict=True):
        raise ValueError("Skill history path escaped its configured directory.")
    snapshots: list[SkillVersionSnapshot] = []
    for path in sorted(directory.iterdir()):
        if path.suffix != ".md":
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("Skill history contains a non-regular snapshot.")
        version = path.stem
        if not _VERSION.fullmatch(version):
            raise ValueError("Skill history contains an invalid version filename.")
        markdown = path.read_text(encoding="utf-8")
        if skill_name_from_markdown(markdown) != skill_name:
            raise ValueError("Skill history snapshot has a mismatched skill name.")
        if skill_version_from_markdown(markdown) != version:
            raise ValueError("Skill history filename does not match its version.")
        snapshots.append(
            SkillVersionSnapshot(
                version=version,
                path=path,
                sha256=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                modified_at=path.stat().st_mtime,
            )
        )
    return sorted(
        snapshots,
        key=lambda item: tuple(int(part) for part in item.version.split(".")),
    )


def load_skill_version(
    history_root: str | Path,
    skill_name: str,
    version: str,
) -> str:
    """Read a validated archived snapshot by exact semantic version."""
    if not _VERSION.fullmatch(version):
        raise ValueError("Skill version must use MAJOR.MINOR.PATCH.")
    for snapshot in list_skill_versions(history_root, skill_name):
        if snapshot.version == version:
            return snapshot.path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Version {version} was not found for {skill_name}.")


__all__ = [
    "SkillVersionSnapshot",
    "list_skill_versions",
    "load_skill_version",
    "next_patch_version",
    "record_skill_version",
    "set_skill_version",
    "skill_name_from_markdown",
    "skill_version_from_markdown",
    "validate_skill_name",
]
