"""Portable snapshots of the local ACE memory stores."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing, contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from openjarvis.core.paths import _reject_source_tree
from openjarvis.memory.store import _cross_process_lock
from openjarvis.security.file_utils import (
    secure_create,
    secure_mkdir,
    secure_write_json,
    secure_write_text,
)
from openjarvis.sessions.working_chat import working_chat_path

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openjarvis.core.config import StorageConfig

_DATABASES = {
    "memory.db": {"documents"},
    "knowledge.db": {"knowledge_chunks"},
    "knowledge_graph.db": {"entities", "relations"},
    "episodes.db": {"episodes"},
    "working_chats.db": {"working_chats"},
}
_FACTS = "memory_facts.jsonl"
_MANIFEST = "manifest.json"
_MEMBERS = {*_DATABASES, _FACTS}
_SETTINGS = ("max_facts", "episodes_max_count", "episodes_retention_days")
_CHUNK_BYTES = 1024 * 1024
# Python 3.10 does not export these two SQLite result code names.
_BUSY_CODES = (5, 6)  # SQLITE_BUSY, SQLITE_LOCKED


@contextmanager
def _new_private_directory(path: Path) -> Iterator[Path]:
    path.mkdir(mode=0o700, exist_ok=False)
    try:
        yield secure_mkdir(path)
    except BaseException:
        # Only remove entries the commands can create in their new directory.
        if path.resolve() == path:
            names = {*_MEMBERS, _MANIFEST, "config.toml"}
            names.update(
                name + suffix
                for name in _DATABASES
                for suffix in ("-journal", "-wal", "-shm")
            )
            for name in names:
                with suppress(OSError):
                    (path / name).unlink(missing_ok=True)
            with suppress(OSError):
                path.rmdir()
        raise


def _destination(path: Path) -> Path:
    path = path.expanduser()
    if not path.name or path.name == "..":
        raise ValueError("Specify a new directory inside an existing parent.")
    return path.parent.resolve(strict=True) / path.name


def _verify_database(path: Path, tables: set[str]) -> None:
    # Only the copied database is inspected, without honoring external journals.
    uri = path.as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.execute("PRAGMA trusted_schema=OFF")
        actual = {
            name
            for (name,) in conn.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            )
        }
        if not tables.issubset(actual):
            raise ValueError(f"{path.name} is missing required memory tables.")
        if conn.execute("PRAGMA quick_check(1)").fetchone() != ("ok",):
            raise ValueError(f"{path.name} failed SQLite integrity checks.")


def _snapshot_database(source: Path, destination: Path) -> None:
    def check_progress(status: int, _remaining: int, _total: int) -> None:
        if status in _BUSY_CODES and time.monotonic() > busy_deadline:
            raise TimeoutError(f"Timed out waiting for {destination.name}.")

    uri = source.resolve().as_uri() + "?mode=ro"
    secure_create(destination)
    busy_deadline = time.monotonic() + 15
    with (
        closing(sqlite3.connect(uri, uri=True)) as origin,
        closing(sqlite3.connect(destination)) as target,
    ):
        # Pin a read snapshot so concurrent commits do not restart the copy.
        origin.execute("BEGIN")
        origin.execute("SELECT count(*) FROM sqlite_schema").fetchone()
        origin.backup(target, pages=128, progress=check_progress)
        target.execute("PRAGMA journal_mode=DELETE")
    _verify_database(destination, _DATABASES[destination.name])


def _stream(source: Path, output: BinaryIO | None = None) -> dict[str, int | str]:
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as stream:
        while chunk := stream.read(_CHUNK_BYTES):
            if output is not None:
                output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def _retention(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or value.keys() != set(_SETTINGS):
        raise ValueError("Invalid backup retention settings.")
    for name in _SETTINGS:
        minimum = 0 if name == "episodes_retention_days" else 1
        if type(value[name]) is not int or not minimum <= value[name] < 2**63:
            raise ValueError(f"Invalid backup retention setting {name}.")
    return value


def backup_memory(config: StorageConfig, destination: Path) -> int:
    """Copy the available local stores into a new owner-only directory."""
    sources = {
        "knowledge.db": Path(config.knowledge_db_path),
        "knowledge_graph.db": Path(config.graph_db_path),
        "episodes.db": Path(config.episodes_path),
        "working_chats.db": working_chat_path(),
    }
    if config.default_backend == "sqlite":
        sources["memory.db"] = Path(config.db_path)
    if config.backend == "local":
        sources[_FACTS] = Path(config.facts_path)
    sources = {name: path.expanduser() for name, path in sources.items()}
    sources = {name: path for name, path in sources.items() if path.is_file()}
    if not sources:
        raise ValueError("No local memory stores were found to back up.")

    retention = _retention({name: getattr(config, name) for name in _SETTINGS})
    with _new_private_directory(_destination(destination)) as directory:
        files = {}
        for name, source in sources.items():
            target = directory / name
            if name == _FACTS:
                secure_create(target)
                with (
                    _cross_process_lock(source.with_suffix(source.suffix + ".lock")),
                    target.open("wb") as output,
                ):
                    files[name] = _stream(source, output)
            else:
                _snapshot_database(source, target)
                files[name] = _stream(target)
        secure_write_json(
            directory / _MANIFEST,
            {"format": 1, "files": files, "retention": retention},
        )
    return len(files)


def _bundle_file(directory: Path, name: str) -> Path:
    path = directory / name
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError(f"Backup member {name} must be a regular file.")
    return path


def _load_manifest(directory: Path) -> tuple[dict, dict[str, int]]:
    with _bundle_file(directory, _MANIFEST).open("rb") as stream:
        content = stream.read(16_385)
    if len(content) > 16_384:
        raise ValueError("Backup manifest is too large.")
    try:
        manifest = json.loads(content)
    except (ValueError, RecursionError) as exc:
        raise ValueError("Backup manifest is invalid.") from exc
    if not isinstance(manifest, dict) or type(manifest.get("format")) is not int:
        raise ValueError("Backup manifest is invalid.")
    if manifest["format"] != 1:
        raise ValueError("Unsupported backup format.")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or not files.keys() <= _MEMBERS:
        raise ValueError("Backup manifest includes invalid memory files.")
    return files, _retention(manifest.get("retention"))


def restore_memory(source: Path, destination: Path) -> int:
    """Verify a backup and restore it as a new local OpenJarvis home."""
    source = source.expanduser().resolve(strict=True)
    destination = _reject_source_tree(_destination(destination))
    files, retention = _load_manifest(source)
    with _new_private_directory(destination) as directory:
        for name, metadata in files.items():
            original = _bundle_file(source, name)
            if (
                not isinstance(metadata, dict)
                or type(metadata.get("size")) is not int
                or original.stat().st_size != metadata["size"]
            ):
                raise ValueError(f"Backup integrity verification failed for {name}.")
            target = secure_create(directory / name)
            with target.open("wb") as output:
                actual = _stream(original, output)
            if actual != metadata:
                raise ValueError(f"Backup integrity verification failed for {name}.")
            if name in _DATABASES:
                _verify_database(target, _DATABASES[name])
        # Paths come from the new home, even when the old profile customized them.
        episodes = "true" if "episodes.db" in files else "false"
        secure_write_text(
            directory / "config.toml",
            '[intelligence]\nprovider = "local"\n\n'
            f"[memory]\nenabled = true\nepisodes_enabled = {episodes}\n"
            + "".join(f"{name} = {value}\n" for name, value in retention.items()),
        )
    return len(files)
