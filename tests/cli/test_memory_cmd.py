"""Tests for ``jarvis memory`` CLI commands."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from openjarvis.cli import cli
from openjarvis.core.registry import MemoryRegistry
from openjarvis.memory.episodes import EpisodeStore
from openjarvis.memory.store import LocalFactStore
from openjarvis.tools.storage.sqlite import SQLiteMemory


def _register_sqlite():
    """Re-register sqlite backend (conftest clears registries)."""
    if not MemoryRegistry.contains("sqlite"):
        MemoryRegistry.register_value("sqlite", SQLiteMemory)


def test_memory_index_file(tmp_path: Path, monkeypatch):
    """Index a single text file and check success message."""
    _register_sqlite()
    db_path = str(tmp_path / "mem.db")

    # Create a text file with enough content
    doc = tmp_path / "doc.txt"
    doc.write_text(" ".join(f"word{i}" for i in range(100)))

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "_get_backend",
        lambda b=None: SQLiteMemory(db_path=db_path),
    )

    result = CliRunner().invoke(cli, ["memory", "index", str(doc)])
    assert result.exit_code == 0
    assert "Indexed" in result.output or "chunk" in result.output


def test_memory_index_ocr_flag_is_forwarded(tmp_path: Path, monkeypatch):
    _register_sqlite()
    image = tmp_path / "scan.png"
    image.write_bytes(b"fake image")
    db_path = str(tmp_path / "memory.db")
    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "_get_backend",
        lambda _backend=None: SQLiteMemory(db_path=db_path),
    )
    options = {}

    def fake_ingest(path, *, config=None, ocr=False):
        options["ocr"] = ocr
        from openjarvis.tools.storage.chunking import chunk_text

        return chunk_text(" ".join(f"word{i}" for i in range(100)), source=str(path))

    monkeypatch.setattr(mod, "ingest_path", fake_ingest)

    result = CliRunner().invoke(cli, ["memory", "index", str(image), "--ocr"])

    assert result.exit_code == 0, result.output
    assert options["ocr"] is True


def test_memory_index_local_media_flag_is_forwarded(tmp_path: Path, monkeypatch):
    media = tmp_path / "recording.wav"
    media.write_bytes(b"fake audio")
    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    speech_config = SimpleNamespace(model="base")
    monkeypatch.setattr(
        mod,
        "_get_backend",
        lambda _backend=None: SimpleNamespace(store=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        mod,
        "load_config",
        lambda: SimpleNamespace(speech=speech_config),
    )
    options = {}

    def fake_ingest(
        path,
        *,
        config=None,
        ocr=False,
        transcribe_media=False,
        speech_config=None,
    ):
        options.update(
            ocr=ocr,
            transcribe_media=transcribe_media,
            speech_config=speech_config,
        )
        from openjarvis.tools.storage.chunking import chunk_text

        return chunk_text(" ".join(f"spoken{i}" for i in range(100)), source=str(path))

    monkeypatch.setattr(mod, "ingest_path", fake_ingest)

    result = CliRunner().invoke(
        cli, ["memory", "index", str(media), "--transcribe-media"]
    )

    assert result.exit_code == 0, result.output
    assert options == {
        "ocr": False,
        "transcribe_media": True,
        "speech_config": speech_config,
    }


def test_memory_index_replaces_existing_source(tmp_path: Path, monkeypatch):
    """Re-indexing a file replaces its previous chunks."""
    _register_sqlite()
    db_path = str(tmp_path / "mem.db")
    doc = tmp_path / "doc.txt"
    doc.write_text(" ".join(["legacy"] * 100), encoding="utf-8")

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "_get_backend",
        lambda b=None: SQLiteMemory(db_path=db_path),
    )

    first = CliRunner().invoke(cli, ["memory", "index", str(doc)])
    assert first.exit_code == 0

    doc.write_text(" ".join(["updated"] * 100), encoding="utf-8")
    second = CliRunner().invoke(cli, ["memory", "index", str(doc)])
    assert second.exit_code == 0

    backend = SQLiteMemory(db_path=db_path)
    assert backend.count() == 1
    assert backend.retrieve("legacy") == []
    updated = backend.retrieve("updated")
    assert len(updated) == 1
    assert updated[0].source == str(doc)


def test_memory_index_nonexistent(tmp_path: Path):
    """Indexing a nonexistent path should fail."""
    _register_sqlite()
    result = CliRunner().invoke(cli, ["memory", "index", str(tmp_path / "nope")])
    assert result.exit_code != 0


def test_memory_index_rejects_non_progress_chunk_values(tmp_path: Path):
    doc = tmp_path / "doc.txt"
    doc.write_text("a b", encoding="utf-8")

    zero_size = CliRunner().invoke(
        cli,
        ["memory", "index", str(doc), "--chunk-size", "0"],
    )
    negative_overlap = CliRunner().invoke(
        cli,
        ["memory", "index", str(doc), "--chunk-overlap", "-1"],
    )

    assert zero_size.exit_code != 0
    assert "0 is not in the range x>=1" in zero_size.output
    assert negative_overlap.exit_code != 0
    assert "-1 is not in the range x>=0" in negative_overlap.output


def test_memory_search_returns_results(tmp_path: Path, monkeypatch):
    """Search returns results from pre-populated backend."""
    _register_sqlite()
    db_path = str(tmp_path / "mem.db")
    backend = SQLiteMemory(db_path=db_path)
    backend.store(
        "Python programming language guide",
        source="guide.md",
    )

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "_get_backend",
        lambda b=None: SQLiteMemory(db_path=db_path),
    )

    result = CliRunner().invoke(cli, ["memory", "search", "Python"])
    assert result.exit_code == 0
    assert "Python" in result.output
    backend.close()


def test_memory_search_no_results(tmp_path: Path, monkeypatch):
    """Search with no matches shows appropriate message."""
    _register_sqlite()
    db_path = str(tmp_path / "mem.db")
    backend = SQLiteMemory(db_path=db_path)
    backend.store("some unrelated content about cats")

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "_get_backend",
        lambda b=None: SQLiteMemory(db_path=db_path),
    )

    result = CliRunner().invoke(cli, ["memory", "search", "quantum supercollider"])
    assert result.exit_code == 0
    assert "No results" in result.output
    backend.close()


def test_memory_stats_shows_count(tmp_path: Path, monkeypatch):
    """Stats command shows document count."""
    _register_sqlite()
    db_path = str(tmp_path / "mem.db")
    backend = SQLiteMemory(db_path=db_path)
    backend.store("doc one")
    backend.store("doc two")

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "_get_backend",
        lambda b=None: SQLiteMemory(db_path=db_path),
    )

    result = CliRunner().invoke(cli, ["memory", "stats"])
    assert result.exit_code == 0
    assert "2" in result.output
    backend.close()


def _patch_fact_store(monkeypatch, tmp_path: Path) -> LocalFactStore:
    """Point ``jarvis memory list/clear`` at a temp fact store."""
    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    store = LocalFactStore(tmp_path / "facts.jsonl")
    monkeypatch.setattr(mod, "_get_fact_store", lambda: store)
    return store


def test_memory_list_empty(tmp_path: Path, monkeypatch):
    _patch_fact_store(monkeypatch, tmp_path)
    result = CliRunner().invoke(cli, ["memory", "list"])
    assert result.exit_code == 0
    assert "No memory facts" in result.output


def test_memory_list_shows_facts(tmp_path: Path, monkeypatch):
    store = _patch_fact_store(monkeypatch, tmp_path)
    store.add("User prefers dark mode")
    store.add("User lives in Berlin")

    result = CliRunner().invoke(cli, ["memory", "list"])
    assert result.exit_code == 0
    assert "dark mode" in result.output
    assert "Berlin" in result.output


def test_memory_list_marks_only_quarantined_facts(tmp_path: Path, monkeypatch):
    store = _patch_fact_store(monkeypatch, tmp_path)
    store.add("clean auto note", source="auto", trust="auto")
    store.add("flagged note", source="auto", trust="untrusted")

    result = CliRunner().invoke(cli, ["memory", "list"])
    assert result.exit_code == 0
    assert "quarantined" in result.output.lower()
    assert "excluded from model recall" in result.output.lower()


def test_memory_list_without_quarantine_shows_no_warning(tmp_path: Path, monkeypatch):
    store = _patch_fact_store(monkeypatch, tmp_path)
    store.add("clean auto note", source="auto", trust="auto")

    result = CliRunner().invoke(cli, ["memory", "list"])
    assert result.exit_code == 0
    assert "quarantined" not in result.output.lower()


def test_memory_facts_reindex_uses_configured_local_index(tmp_path: Path, monkeypatch):
    import openjarvis.memory.fact_embeddings as fact_embeddings

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    store = _patch_fact_store(monkeypatch, tmp_path)
    store.add("The desktop has an RX graphics card", trust="auto")
    facts_path = tmp_path / "facts.jsonl"
    calls = {}

    class _FakeIndex:
        def __init__(self, path, *, embedding_model):
            calls["path"] = path
            calls["model"] = embedding_model

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def index_missing_embeddings(self, facts, *, limit):
            calls["facts"] = facts
            calls["limit"] = limit
            return 1, 0

    monkeypatch.setattr(fact_embeddings, "FactEmbeddingIndex", _FakeIndex)
    monkeypatch.setattr(
        mod,
        "load_config",
        lambda: SimpleNamespace(
            memory=SimpleNamespace(
                facts_path=str(facts_path),
                facts_embedding_model="nomic-embed-text",
            )
        ),
    )

    result = CliRunner().invoke(cli, ["memory", "facts", "reindex", "--limit", "10"])

    assert result.exit_code == 0, result.output
    assert "Indexed 1 fact" in result.output
    assert calls["path"] == facts_path.with_name("facts.vectors.db")
    assert calls["model"] == "nomic-embed-text"
    assert calls["limit"] == 10
    assert len(calls["facts"]) == 1


def test_memory_trust_promotes_a_quarantined_fact(tmp_path: Path, monkeypatch):
    store = _patch_fact_store(monkeypatch, tmp_path)
    store.add("flagged note", source="auto", trust="untrusted")
    assert store.list()[0].trusted_for_recall is False

    result = CliRunner().invoke(cli, ["memory", "trust", "1"])

    assert result.exit_code == 0
    assert store.list()[0].trust == "trusted"
    assert store.list()[0].trusted_for_recall is True


def test_memory_trust_rejects_an_unknown_index(tmp_path: Path, monkeypatch):
    _patch_fact_store(monkeypatch, tmp_path).add("only note", trust="untrusted")

    result = CliRunner().invoke(cli, ["memory", "trust", "4"])

    assert result.exit_code != 0
    assert "no fact #4" in result.output.lower()


def test_memory_clear_with_confirmation(tmp_path: Path, monkeypatch):
    store = _patch_fact_store(monkeypatch, tmp_path)
    store.add("fact one")
    store.add("fact two")

    result = CliRunner().invoke(cli, ["memory", "clear"], input="y\n")
    assert result.exit_code == 0
    assert "Cleared 2" in result.output
    assert store.count() == 0


def test_memory_clear_aborted(tmp_path: Path, monkeypatch):
    store = _patch_fact_store(monkeypatch, tmp_path)
    store.add("keep me")

    result = CliRunner().invoke(cli, ["memory", "clear"], input="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output
    assert store.count() == 1


def test_memory_clear_yes_flag(tmp_path: Path, monkeypatch):
    store = _patch_fact_store(monkeypatch, tmp_path)
    store.add("fact")

    result = CliRunner().invoke(cli, ["memory", "clear", "--yes"])
    assert result.exit_code == 0
    assert "Cleared 1" in result.output
    assert store.count() == 0


def test_memory_clear_empty(tmp_path: Path, monkeypatch):
    _patch_fact_store(monkeypatch, tmp_path)
    result = CliRunner().invoke(cli, ["memory", "clear"])
    assert result.exit_code == 0
    assert "No memory facts to clear" in result.output


def test_memory_episodes_show_by_id(tmp_path: Path, monkeypatch):
    episode_path = tmp_path / "episodes.db"
    store = EpisodeStore(episode_path, semantic_enabled=False)
    episode_id = store.episode_id_for(
        source="chat",
        session_id="session-1",
        idempotency_key="turn-1",
    )
    try:
        assert store.record(
            "This project keeps a local note.",
            "The full assistant response is shown by the episode ID.",
            source="chat",
            session_id="session-1",
            idempotency_key="turn-1",
        )
    finally:
        store.close()

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "_get_episode_store",
        lambda: EpisodeStore(episode_path, semantic_enabled=False),
    )

    result = CliRunner().invoke(cli, ["memory", "episodes", "show", episode_id])

    assert result.exit_code == 0
    assert episode_id in result.output
    assert "This project keeps a local note." in result.output
    assert "The full assistant response is shown by the episode ID." in result.output


def test_memory_episodes_show_rejects_malformed_id():
    result = CliRunner().invoke(cli, ["memory", "episodes", "show", "short-id"])

    assert result.exit_code != 0
    assert "full 32-character ID" in result.output


def test_memory_episodes_forget_purges_episode_graph_evidence_and_keeps_web_sources(
    tmp_path: Path, monkeypatch
):
    from openjarvis.tools.storage.knowledge_graph import (
        Entity,
        KnowledgeGraphMemory,
        Relation,
    )

    episode_path = tmp_path / "episodes.db"
    graph_path = tmp_path / "knowledge_graph.db"
    seed = EpisodeStore(episode_path, semantic_enabled=False)
    episode_id = seed.episode_id_for(
        source="chat",
        session_id="session-1",
        idempotency_key="turn-1",
    )
    assert seed.record(
        "This project keeps an important local decision.",
        "The decision is retained in a separate episode.",
        source="chat",
        session_id="session-1",
        idempotency_key="turn-1",
    )
    seed.close()
    facts = LocalFactStore(tmp_path / "facts.jsonl")
    other_episode = "b" * 32
    facts.add_many_with_provenance(
        ["episode-only fact", "shared fact"],
        source="auto",
        trust="auto",
        source_id=episode_id,
    )
    facts.add_many_with_provenance(
        ["shared fact"],
        source="auto",
        trust="auto",
        source_id=other_episode,
    )
    facts.add("legacy unlinked fact", source="auto", trust="auto")

    graph = KnowledgeGraphMemory(graph_path)
    graph.add_entity(
        Entity(
            "project",
            "project",
            "ACE",
            properties={
                "evidence_quote": "ACE has a private conversation-only detail.",
                "evidence_sources": [
                    {
                        "source": "conversation:session-1:turn-1",
                        "source_kind": "conversation",
                        "episode_id": episode_id,
                        "quote": "ACE has a private conversation-only detail.",
                    }
                ],
            },
            source="conversation",
        )
    )
    graph.add_entity(Entity("base", "software", "OpenJarvis"))
    graph.add_relation(
        Relation(
            "project",
            "base",
            "built_on",
            properties={"episode_ids": [episode_id, "b" * 32]},
            source="conversation",
            evidence=[
                {
                    "source": "conversation:session-1:turn-1",
                    "quote": "ACE is built on OpenJarvis.",
                    "episode_id": episode_id,
                },
                {
                    "source": "web",
                    "source_url": "https://example.com/ace",
                    "quote": "OpenJarvis is a software framework.",
                },
            ],
        )
    )
    graph.close()

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "load_config",
        lambda: SimpleNamespace(
            memory=SimpleNamespace(
                graph_db_path=str(graph_path), facts_path=str(tmp_path / "facts.jsonl")
            )
        ),
    )
    monkeypatch.setattr(
        mod,
        "_get_episode_store",
        lambda: EpisodeStore(episode_path, semantic_enabled=False),
    )

    result = CliRunner().invoke(cli, ["memory", "episodes", "forget", episode_id])

    assert result.exit_code == 0, result.output
    assert "Forgot episode" in result.output
    assert "evidence item(s)" in result.output
    assert "Removed 1 fact(s) supported only by this episode" in result.output
    assert "Kept 1 fact(s) with another known origin" in result.output
    assert "no complete episode provenance" in result.output
    remaining = EpisodeStore(episode_path, semantic_enabled=False)
    try:
        assert remaining.get_by_id(episode_id) is None
    finally:
        remaining.close()
    remaining_facts = LocalFactStore(tmp_path / "facts.jsonl").list()
    assert {fact.text for fact in remaining_facts} == {
        "shared fact",
        "legacy unlinked fact",
    }
    shared = next(fact for fact in remaining_facts if fact.text == "shared fact")
    assert shared.source_ids == [other_episode]
    assert shared.source_tracking_complete is True
    graph = KnowledgeGraphMemory(graph_path)
    try:
        assert graph.count_episode_links(episode_id) == 0
        assert graph.count_episode_links("b" * 32) == 1
        assert graph.evidence_count() == 1
        relation = graph.query_pattern(relation_type="built_on").relations[0]
        assert relation.properties["episode_ids"] == ["b" * 32]
        assert relation.evidence[0]["source"] == "web"
        project = graph.get_entity("project")
        assert project is not None
        assert "evidence_quote" not in project.properties
        assert "private conversation-only detail" not in str(project.properties)
        events = json.dumps(graph.recent_events(limit=50), ensure_ascii=False)
        assert "private conversation-only detail" not in events
        assert "ACE is built on OpenJarvis." not in events
        assert graph.recent_events(limit=1)[0]["event_type"] == "memory_data_purged"
    finally:
        graph.close()


def test_memory_episodes_forget_reports_and_retries_partial_cleanup(
    tmp_path: Path, monkeypatch
):
    episode_path = tmp_path / "episodes.db"
    seed = EpisodeStore(episode_path, semantic_enabled=False)
    episode_id = seed.episode_id_for(
        source="chat",
        session_id="session-partial",
        idempotency_key="turn-partial",
    )
    assert seed.record(
        "The project decision is to keep the archive on this computer.",
        "This decision is saved so the archive remains local.",
        source="chat",
        session_id="session-partial",
        idempotency_key="turn-partial",
    )
    seed.close()

    fact_path = tmp_path / "facts.jsonl"
    LocalFactStore(fact_path).add_many_with_provenance(
        ["episode-only fact"],
        source="auto",
        trust="auto",
        source_id=episode_id,
    )
    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "load_config",
        lambda: SimpleNamespace(
            memory=SimpleNamespace(
                graph_db_path=str(tmp_path / "graph.db"),
                facts_path=str(fact_path),
            )
        ),
    )
    monkeypatch.setattr(
        mod,
        "_get_episode_store",
        lambda: EpisodeStore(episode_path, semantic_enabled=False),
    )
    monkeypatch.setattr(mod, "_get_fact_store", lambda: LocalFactStore(fact_path))
    monkeypatch.setattr(mod, "_prune_fact_embeddings", lambda *_args, **_kwargs: None)

    def fail_graph_purge(_episode_ids):
        raise RuntimeError("graph store locked")

    monkeypatch.setattr(mod, "_purge_episode_graph_data", fail_graph_purge)
    failed = CliRunner().invoke(cli, ["memory", "episodes", "forget", episode_id])

    assert failed.exit_code != 0
    assert "episode may still be present" in failed.output
    assert "may already have been removed" in failed.output
    assert "recover" in failed.output
    retained = EpisodeStore(episode_path, semantic_enabled=False)
    try:
        assert retained.get_by_id(episode_id) is not None
        assert retained.pending_deletion_intents() == [episode_id]
    finally:
        retained.close()
    assert LocalFactStore(fact_path).list() == []

    monkeypatch.setattr(
        mod,
        "_purge_episode_graph_data",
        lambda _episode_ids: {
            "entities_removed": 0,
            "entities_updated": 0,
            "relations_removed": 0,
            "relations_updated": 0,
            "evidence_removed": 0,
        },
    )
    retried = CliRunner().invoke(cli, ["memory", "episodes", "recover"])

    assert retried.exit_code == 0, retried.output
    assert "Recovered cleanup for 1 episode(s)" in retried.output
    removed = EpisodeStore(episode_path, semantic_enabled=False)
    try:
        assert removed.get_by_id(episode_id) is None
        assert removed.pending_deletion_intents() == []
    finally:
        removed.close()


def test_memory_episodes_clear_recovery_keeps_newer_memory(tmp_path: Path, monkeypatch):
    from openjarvis.tools.storage.knowledge_graph import (
        Entity,
        KnowledgeGraphMemory,
        Relation,
    )

    episode_path = tmp_path / "episodes.db"
    seed = EpisodeStore(episode_path, semantic_enabled=False)
    episode_id = seed.episode_id_for(
        source="chat",
        session_id="session-clear-partial",
        idempotency_key="turn-clear-partial",
    )
    assert seed.record(
        "The project decision is to keep this archive local.",
        "The decision is saved so the project history remains on this computer.",
        source="chat",
        session_id="session-clear-partial",
        idempotency_key="turn-clear-partial",
    )
    seed.close()

    fact_path = tmp_path / "facts.jsonl"
    LocalFactStore(fact_path).add_many_with_provenance(
        ["episode-only fact"],
        source="auto",
        trust="auto",
        source_id=episode_id,
    )
    graph_path = tmp_path / "graph.db"
    with KnowledgeGraphMemory(graph_path) as graph:
        graph.add_entity(Entity("project", "project", "ACE"))
        graph.add_entity(Entity("framework", "software", "OpenJarvis"))
        graph.add_relation(
            Relation(
                "project",
                "framework",
                "built_on",
                evidence=[
                    {"source": "conversation:chat:old", "quote": "Old graph proof."}
                ],
            )
        )
    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "load_config",
        lambda: SimpleNamespace(
            memory=SimpleNamespace(
                graph_db_path=str(graph_path),
                facts_path=str(fact_path),
            )
        ),
    )
    monkeypatch.setattr(
        mod,
        "_get_episode_store",
        lambda: EpisodeStore(episode_path, semantic_enabled=False),
    )
    monkeypatch.setattr(mod, "_get_fact_store", lambda: LocalFactStore(fact_path))
    monkeypatch.setattr(mod, "_prune_fact_embeddings", lambda *_args, **_kwargs: None)

    purge_graph = mod._purge_episode_graph_data

    def fail_graph_purge(*_args, **_kwargs):
        raise RuntimeError("graph store locked")

    monkeypatch.setattr(mod, "_purge_episode_graph_data", fail_graph_purge)
    result = CliRunner().invoke(cli, ["memory", "episodes", "clear", "--yes"])

    assert result.exit_code != 0
    assert "Some episodes may remain" in result.output
    assert "may already have been removed" in result.output
    assert "jarvis memory episodes recover" in result.output
    retained = EpisodeStore(episode_path, semantic_enabled=False)
    try:
        assert retained.get_by_id(episode_id) is not None
        pending = retained.pending_archive_clear()
        assert pending is not None
        assert pending.episode_ids == (episode_id,)
        assert pending.conversation_sources == ("conversation:chat:old",)
        assert retained.record(
            "The next project decision came after the cleanup was interrupted.",
            "The new project history should still be available.",
            source="chat",
            session_id="session-clear-partial",
            idempotency_key="turn-new",
        )
        new_id = retained.episode_id_for(
            source="chat",
            session_id="session-clear-partial",
            idempotency_key="turn-new",
        )
    finally:
        retained.close()
    assert LocalFactStore(fact_path).list() == []

    LocalFactStore(fact_path).add_many_with_provenance(
        ["new episode fact"], source="auto", trust="auto", source_id=new_id
    )
    with KnowledgeGraphMemory(graph_path) as graph:
        graph.add_relation(
            Relation(
                "project",
                "framework",
                "built_on",
                evidence=[
                    {"source": "conversation:chat:new", "quote": "New graph proof."}
                ],
            )
        )
    monkeypatch.setattr(mod, "_purge_episode_graph_data", purge_graph)
    recovered = CliRunner().invoke(cli, ["memory", "episodes", "recover"])
    assert recovered.exit_code == 0, recovered.output
    assert "Recovered the interrupted archive clear" in recovered.output
    cleared = EpisodeStore(episode_path, semantic_enabled=False)
    try:
        assert cleared.get_by_id(episode_id) is None
        assert cleared.get_by_id(new_id) is not None
        assert cleared.pending_archive_clear() is None
    finally:
        cleared.close()
    assert [fact.text for fact in LocalFactStore(fact_path).list()] == [
        "new episode fact"
    ]
    with KnowledgeGraphMemory(graph_path) as graph:
        relation = graph.query_pattern(relation_type="built_on").relations[0]
        assert [item["quote"] for item in relation.evidence] == ["New graph proof."]


def test_memory_episodes_recover_finishes_cleanup_after_episode_record_was_deleted(
    tmp_path: Path, monkeypatch
):
    episode_path = tmp_path / "episodes.db"
    seed = EpisodeStore(episode_path, semantic_enabled=False)
    assert seed.record(
        "The project decision is to preserve local research notes.",
        "The decision is retained as part of the local research history.",
        source="chat",
        session_id="session-crash",
        idempotency_key="turn-crash",
    )
    episode_id = seed.list(limit=1)[0].episode_id
    seed.record_deletion_intent(episode_id)
    assert seed.delete(episode_id)
    seed.close()

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "_get_episode_store",
        lambda: EpisodeStore(episode_path, semantic_enabled=False),
    )
    fact_calls = []
    graph_calls = []
    monkeypatch.setattr(
        mod, "_purge_episode_fact_data", lambda ids: fact_calls.append(ids) or {}
    )
    monkeypatch.setattr(
        mod, "_purge_episode_graph_data", lambda ids: graph_calls.append(ids) or {}
    )

    result = CliRunner().invoke(cli, ["memory", "episodes", "recover"])

    assert result.exit_code == 0, result.output
    assert "Recovered cleanup for 1 episode(s)" in result.output
    assert fact_calls == [[episode_id]]
    assert graph_calls == [[episode_id]]
    reopened = EpisodeStore(episode_path, semantic_enabled=False)
    try:
        assert reopened.get_by_id(episode_id) is None
        assert reopened.pending_deletion_intents() == []
    finally:
        reopened.close()


def test_memory_episodes_forget_fails_closed_when_fact_store_lacks_provenance(
    tmp_path: Path, monkeypatch
):
    episode_path = tmp_path / "episodes.db"
    seed = EpisodeStore(episode_path, semantic_enabled=False)
    episode_id = seed.episode_id_for(
        source="chat",
        session_id="session-1",
        idempotency_key="turn-unsupported",
    )
    assert seed.record(
        "The project decision is to keep ACE local.",
        "The decision is saved for the project.",
        source="chat",
        session_id="session-1",
        idempotency_key="turn-unsupported",
    )
    seed.close()

    class _UnsupportedFactStore:
        def purge_episode_sources(self, _episode_ids):
            return {"unsupported": 1}

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "load_config",
        lambda: SimpleNamespace(
            memory=SimpleNamespace(
                graph_db_path=str(tmp_path / "graph.db"),
                facts_path=str(tmp_path / "facts.jsonl"),
            )
        ),
    )
    monkeypatch.setattr(
        mod,
        "_get_episode_store",
        lambda: EpisodeStore(episode_path, semantic_enabled=False),
    )
    monkeypatch.setattr(
        mod,
        "_get_fact_store",
        lambda: _UnsupportedFactStore(),
    )

    result = CliRunner().invoke(cli, ["memory", "episodes", "forget", episode_id])

    assert result.exit_code != 0
    assert "does not support episode-aware cleanup" in result.output
    remaining = EpisodeStore(episode_path, semantic_enabled=False)
    try:
        assert remaining.get_by_id(episode_id) is not None
    finally:
        remaining.close()


def test_memory_episodes_clear_purges_conversation_graph_data_but_keeps_web(
    tmp_path: Path, monkeypatch
):
    from openjarvis.tools.storage.knowledge_graph import (
        Entity,
        KnowledgeGraphMemory,
        Relation,
    )

    episode_path = tmp_path / "episodes.db"
    graph_path = tmp_path / "knowledge_graph.db"
    seed = EpisodeStore(episode_path, semantic_enabled=False)
    episode_id = seed.episode_id_for(
        source="chat",
        session_id="session-1",
        idempotency_key="turn-1",
    )
    assert seed.record(
        "ACE remembers a local design decision.",
        "The assistant response stays inside the episode.",
        source="chat",
        session_id="session-1",
        idempotency_key="turn-1",
    )
    seed.close()
    facts = LocalFactStore(tmp_path / "facts.jsonl")
    facts.add_many_with_provenance(
        ["episode-only fact", "another episode fact"],
        source="auto",
        trust="auto",
        source_id=episode_id,
    )
    facts.add("legacy unlinked fact", source="auto", trust="auto")
    facts.add_many_with_provenance(
        ["fact without a retained episode"],
        source="auto",
        trust="auto",
        source_id="f" * 32,
    )

    graph = KnowledgeGraphMemory(graph_path)
    graph.add_entity(Entity("project", "project", "ACE"))
    graph.add_entity(Entity("framework", "software", "OpenJarvis"))
    graph.add_relation(
        Relation(
            "project",
            "framework",
            "built_on",
            properties={"episode_ids": [episode_id]},
            evidence=[
                {
                    "source": "conversation:session-1:turn-1",
                    "episode_id": episode_id,
                    "quote": "ACE is built on OpenJarvis.",
                },
                {
                    "source": "web",
                    "source_url": "https://example.com/ace",
                    "quote": "OpenJarvis is a software framework.",
                },
            ],
        )
    )
    graph.close()

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "load_config",
        lambda: SimpleNamespace(
            memory=SimpleNamespace(
                graph_db_path=str(graph_path), facts_path=str(tmp_path / "facts.jsonl")
            )
        ),
    )
    monkeypatch.setattr(
        mod,
        "_get_episode_store",
        lambda: EpisodeStore(episode_path, semantic_enabled=False),
    )

    result = CliRunner().invoke(cli, ["memory", "episodes", "clear", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Cleared 1 conversation episode(s)" in result.output
    assert "Removed 2 fact(s) supported only by cleared episodes" in result.output
    remaining = EpisodeStore(episode_path, semantic_enabled=False)
    try:
        assert remaining.count() == 0
    finally:
        remaining.close()
    remaining_facts = LocalFactStore(tmp_path / "facts.jsonl").list()
    assert {fact.text for fact in remaining_facts} == {
        "legacy unlinked fact",
        "fact without a retained episode",
    }
    graph = KnowledgeGraphMemory(graph_path)
    try:
        relation = graph.query_pattern(relation_type="built_on").relations[0]
        assert "episode_ids" not in relation.properties
        assert [item["source"] for item in relation.evidence] == ["web"]
        events = json.dumps(graph.recent_events(limit=50), ensure_ascii=False)
        assert "ACE is built on OpenJarvis." not in events
        assert graph.recent_events(limit=1)[0]["event_type"] == "memory_data_purged"
    finally:
        graph.close()


def test_memory_episodes_clear_scrubs_legacy_graph_data_when_archive_is_empty(
    tmp_path: Path, monkeypatch
):
    from openjarvis.tools.storage.knowledge_graph import (
        Entity,
        KnowledgeGraphMemory,
        Relation,
    )

    episode_path = tmp_path / "episodes.db"
    graph_path = tmp_path / "knowledge_graph.db"
    seed = EpisodeStore(episode_path, semantic_enabled=False)
    with seed._conn:
        seed._conn.execute(
            "INSERT INTO episode_deletion_journal "
            "(scope_id, created_at) VALUES ('*', 1)"
        )
    seed.close()

    graph = KnowledgeGraphMemory(graph_path)
    graph.add_entity(
        Entity(
            "project",
            "project",
            "ACE",
            properties={"evidence_quote": "Legacy conversation quote."},
            source="conversation",
        )
    )
    graph.add_entity(
        Entity("framework", "software", "OpenJarvis", source="conversation")
    )
    graph.add_relation(
        Relation(
            "project",
            "framework",
            "built_on",
            evidence=[
                {
                    "source": "conversation:legacy:turn",
                    "quote": "Legacy conversation quote.",
                }
            ],
        )
    )
    graph.close()

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "load_config",
        lambda: SimpleNamespace(
            memory=SimpleNamespace(
                graph_db_path=str(graph_path), facts_path=str(tmp_path / "facts.jsonl")
            )
        ),
    )
    monkeypatch.setattr(
        mod,
        "_get_episode_store",
        lambda: EpisodeStore(episode_path, semantic_enabled=False),
    )

    recovery = CliRunner().invoke(cli, ["memory", "episodes", "recover"])
    assert recovery.exit_code != 0
    assert "has no saved scope" in recovery.output
    with KnowledgeGraphMemory(graph_path) as current_graph:
        assert current_graph.relation_count() == 1

    result = CliRunner().invoke(cli, ["memory", "episodes", "clear", "--yes"])

    assert result.exit_code == 0, result.output
    assert "No conversation episodes to clear" in result.output
    graph = KnowledgeGraphMemory(graph_path)
    try:
        assert graph.entity_count() == 0
        assert graph.relation_count() == 0
        events = json.dumps(graph.recent_events(limit=50), ensure_ascii=False)
        assert "Legacy conversation quote" not in events
        assert graph.recent_events(limit=1)[0]["event_type"] == "memory_data_purged"
    finally:
        graph.close()


class _EpisodeStoreFinalDeleteFailure:
    def get_by_id(self, _episode_id: str):
        return object()

    def delete(self, _episode_id: str) -> bool:
        raise RuntimeError("episode database locked")

    def pending_deletion_intents(self):
        return []

    def record_deletion_intent(self, _episode_id):
        pass

    def finish_deletion_intent(self, _episode_id):
        pass

    def pending_archive_clear(self):
        return None

    def archive_episode_ids(self):
        return ("a" * 32,)

    def record_archive_clear(self, _intent) -> None:
        pass

    def finish_archive_clear(self, _intent) -> int:
        raise RuntimeError("episode database locked")

    def close(self) -> None:
        pass


def test_memory_episodes_forget_reports_final_episode_store_failure(monkeypatch):
    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(mod, "_get_episode_store", _EpisodeStoreFinalDeleteFailure)
    monkeypatch.setattr(mod, "_purge_episode_fact_data", lambda _ids: {})
    monkeypatch.setattr(mod, "_purge_episode_graph_data", lambda _ids: {})

    result = CliRunner().invoke(cli, ["memory", "episodes", "forget", "a" * 32])

    assert result.exit_code != 0
    assert "episode may still be present" in result.output
    assert "related memory may already have been removed" in result.output
    assert "retry" in result.output
    assert "episode database locked" in result.output


def test_memory_episodes_clear_reports_final_episode_store_failure(
    tmp_path, monkeypatch
):
    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(mod, "_get_episode_store", _EpisodeStoreFinalDeleteFailure)
    monkeypatch.setattr(
        mod,
        "load_config",
        lambda: SimpleNamespace(
            memory=SimpleNamespace(graph_db_path=str(tmp_path / "graph.db"))
        ),
    )
    monkeypatch.setattr(mod, "_purge_episode_fact_data", lambda _ids: {})
    monkeypatch.setattr(mod, "_purge_episode_graph_data", lambda *_args, **_kwargs: {})

    result = CliRunner().invoke(cli, ["memory", "episodes", "clear", "--yes"])

    assert result.exit_code != 0
    assert "Some episodes may remain" in result.output
    assert "related memory may already have been removed" in result.output
    assert "retry" in result.output
    assert "episode database locked" in result.output


class _EpisodeStoreConcurrentDelete(_EpisodeStoreFinalDeleteFailure):
    def delete(self, _episode_id: str) -> bool:
        return False


def test_memory_episodes_forget_reports_episode_removed_before_final_delete(
    monkeypatch,
):
    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(mod, "_get_episode_store", _EpisodeStoreConcurrentDelete)
    monkeypatch.setattr(mod, "_purge_episode_fact_data", lambda _ids: {})
    monkeypatch.setattr(mod, "_purge_episode_graph_data", lambda _ids: {})

    result = CliRunner().invoke(cli, ["memory", "episodes", "forget", "b" * 32])

    assert result.exit_code != 0
    assert "Related memory cleanup completed" in result.output
    assert "removed by another process" in result.output


def test_memory_facts_revise_updates_recall_and_shows_history(tmp_path, monkeypatch):
    from openjarvis.tools.storage.context import build_context_message

    store = _patch_fact_store(monkeypatch, tmp_path)
    old = "The desktop has an RX 7600."
    new = "The desktop uses Ryzen integrated graphics."
    assert store.add(old, source="auto", trust="auto")
    runner = CliRunner()

    revised = runner.invoke(cli, ["memory", "facts", "revise", old, new])
    assert revised.exit_code == 0, revised.output
    listing = runner.invoke(cli, ["memory", "list"])
    assert "superseded" in listing.output
    assert "quarantined" not in listing.output
    recalled = build_context_message([], facts=store.list()).content
    assert new + " [user-approved]" in recalled
    assert old not in recalled
    trust = runner.invoke(cli, ["memory", "trust", "1"])
    assert trust.exit_code != 0
    assert "superseded" in trust.output


def test_memory_facts_revise_rejects_when_local_scanners_cannot_clear(
    tmp_path, monkeypatch
):
    store = _patch_fact_store(monkeypatch, tmp_path)
    assert store.add("Existing fact")
    monkeypatch.setattr("openjarvis.memory.service._build_secret_scanner", lambda: None)
    result = CliRunner().invoke(
        cli, ["memory", "facts", "revise", "Existing fact", "Corrected fact"]
    )
    assert result.exit_code != 0
    assert "could not clear" in result.output
    assert [fact.text for fact in store.list()] == ["Existing fact"]


def test_memory_facts_revise_rejects_detected_credentials(tmp_path, monkeypatch):
    store = _patch_fact_store(monkeypatch, tmp_path)
    assert store.add("Existing fact")
    result = CliRunner().invoke(
        cli,
        ["memory", "facts", "revise", "Existing fact", "password='fake-test-value'"],
    )
    assert result.exit_code != 0
    assert "could not clear" in result.output
    assert [(fact.text, fact.superseded) for fact in store.list()] == [
        ("Existing fact", False)
    ]
