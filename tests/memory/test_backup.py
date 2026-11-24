"""Exercise backup and restore with isolated memory data."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pytest
from click.testing import CliRunner

from openjarvis.cli import cli
from openjarvis.connectors.store import KnowledgeStore
from openjarvis.core.config import StorageConfig, load_config
from openjarvis.memory.backup import backup_memory, restore_memory
from openjarvis.memory.episodes import EpisodeStore
from openjarvis.memory.store import LocalFactStore
from openjarvis.sessions.working_chat import WorkingChatStore
from openjarvis.tools.storage.knowledge_graph import Entity, KnowledgeGraphMemory
from openjarvis.tools.storage.sqlite import SQLiteMemory


@pytest.fixture
def local_profile(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("OPENJARVIS_HOME", str(home))
    monkeypatch.delenv("OPENJARVIS_CONFIG", raising=False)
    return home


def test_backup_restore_loads_all_local_stores(tmp_path, monkeypatch, local_profile):
    facts_path = json.dumps(str(local_profile / "custom facts.jsonl"))
    (local_profile / "config.toml").write_text(
        f"[memory]\nfacts_path = {facts_path}\n"
        "episodes_max_count = 9000\nepisodes_retention_days = 0\n"
        "\n[example]\napi_key = 'test-credential'\n",
        encoding="utf-8",
    )
    config = load_config().memory
    assert LocalFactStore(config.facts_path).add("The blue binder holds notes.")
    question = "Our decision was to keep the reading group handouts in the blue binder."
    answer = "The finished handouts are in the blue binder by the shared desk."
    with (
        closing(EpisodeStore(config.episodes_path)) as episodes,
        closing(KnowledgeGraphMemory(config.graph_db_path)) as graph,
        closing(KnowledgeStore(config.knowledge_db_path)) as knowledge,
        closing(SQLiteMemory(config.db_path)) as documents,
    ):
        assert episodes.record(question, answer)
        graph.add_entity(Entity("binder", "object", "Blue Binder"))
        knowledge.store("The blue binder contains research notes.")
        documents.store("The blue binder contains reading group handouts.")
        chats = WorkingChatStore()
        chats.append(chats.open("Reading group"), question, answer)
        bundle = tmp_path / "snapshot"
        runner = CliRunner()
        copied = runner.invoke(cli, ["memory", "backup", str(bundle)])
        assert copied.exit_code == 0, (copied.output, copied.exception)

    assert {item.name for item in bundle.iterdir()} == {
        "manifest.json",
        "memory.db",
        "knowledge.db",
        "knowledge_graph.db",
        "episodes.db",
        "working_chats.db",
        "memory_facts.jsonl",
    }
    recovered = tmp_path / "recovered"
    arguments = ["memory", "restore", str(bundle), "--into", str(recovered)]
    restored = runner.invoke(cli, arguments)
    assert restored.exit_code == 0, (restored.output, restored.exception)
    duplicate = runner.invoke(cli, arguments)
    assert duplicate.exit_code == 1

    monkeypatch.setenv("OPENJARVIS_HOME", str(recovered))
    load_config.cache_clear()
    restored_config = load_config()
    assert restored_config.intelligence.provider == "local"
    assert restored_config.memory.episodes_enabled
    assert restored_config.memory.episodes_max_count == 9_000
    assert restored_config.memory.episodes_retention_days == 0
    assert "test-credential" not in (recovered / "config.toml").read_text()
    facts = LocalFactStore(restored_config.memory.facts_path)
    assert [fact.text for fact in facts.list()] == ["The blue binder holds notes."]
    assert WorkingChatStore().open("Reading group").turns == ((question, answer),)
    with closing(EpisodeStore(restored_config.memory.episodes_path)) as episodes:
        assert answer in episodes.search("handouts", top_k=1)[0].content
    with closing(KnowledgeGraphMemory(restored_config.memory.graph_db_path)) as graph:
        assert graph.get_entity("binder").name == "Blue Binder"
    for name, table in (
        ("memory.db", "documents"),
        ("knowledge.db", "knowledge_chunks"),
    ):
        with closing(sqlite3.connect(recovered / name)) as conn:
            assert (
                "blue binder"
                in conn.execute(f"SELECT content FROM {table}").fetchone()[0]
            )


def test_snapshot_includes_uncheckpointed_commits(tmp_path, local_profile):
    config = StorageConfig()
    with closing(sqlite3.connect(config.db_path)) as source:
        source.execute("PRAGMA journal_mode=WAL")
        source.execute("PRAGMA wal_autocheckpoint=0")
        source.execute("CREATE TABLE documents (content TEXT)")
        source.commit()
        source.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        source.execute("INSERT INTO documents VALUES ('in the WAL')")
        source.commit()
        assert (local_profile / "memory.db-wal").stat().st_size > 0
        backup = tmp_path / "bundle"
        assert backup_memory(config, backup) == 1
        assert {item.name for item in backup.iterdir()} == {
            "memory.db",
            "manifest.json",
        }
        destination = tmp_path / "destination"
        assert restore_memory(backup, destination) == 1
        with closing(sqlite3.connect(destination / "memory.db")) as conn:
            assert conn.execute("SELECT content FROM documents").fetchone() == (
                "in the WAL",
            )


def test_tampered_backup_never_creates_a_ready_profile(tmp_path, local_profile):
    config = StorageConfig()
    facts = LocalFactStore(config.facts_path)
    assert facts.add("The binder is blue.")
    bundle = tmp_path / "bundle"
    backup_memory(config, bundle)
    bundled_facts = bundle / "memory_facts.jsonl"
    bundled_facts.write_bytes(bundled_facts.read_bytes().replace(b"blue", b"pink"))

    destination = tmp_path / "reject"
    with pytest.raises(ValueError, match="integrity verification"):
        restore_memory(bundle, destination)
    assert not destination.exists()


def test_backup_rejects_unknown_manifest_path_and_existing_folder(
    tmp_path, local_profile
):
    config = StorageConfig()
    assert LocalFactStore(config.facts_path).add("The binder is blue.")
    bundle = tmp_path / "bundle"
    backup_memory(config, bundle)
    manifest_path = bundle / "manifest.json"
    before = manifest_path.read_bytes()
    with pytest.raises(FileExistsError):
        backup_memory(config, bundle)
    assert manifest_path.read_bytes() == before
    manifest = json.loads(before)
    facts = manifest["files"].pop("memory_facts.jsonl")
    manifest["files"]["../config.toml"] = facts
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid memory files"):
        restore_memory(bundle, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_empty_or_wrong_memory_database_leaves_no_bundle(tmp_path, local_profile):
    config = StorageConfig()
    bundle = tmp_path / "bundle"

    with pytest.raises(ValueError, match="No local memory stores"):
        backup_memory(config, bundle)
    assert not bundle.exists()

    with closing(sqlite3.connect(config.db_path)) as unrelated:
        unrelated.execute("CREATE TABLE unrelated (content TEXT)")
    with pytest.raises(ValueError, match="missing required memory tables"):
        backup_memory(config, bundle)
    assert not bundle.exists()
