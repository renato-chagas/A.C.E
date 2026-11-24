"""Tests for the SQLite/FTS5 memory backend."""

from __future__ import annotations

from pathlib import Path

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import MemoryRegistry
from openjarvis.tools.storage.sqlite import SQLiteMemory


def _make_backend(tmp_path: Path) -> SQLiteMemory:
    """Create an SQLiteMemory instance using a temp database."""
    # Register manually since conftest clears registries
    if not MemoryRegistry.contains("sqlite"):
        MemoryRegistry.register_value("sqlite", SQLiteMemory)
    return SQLiteMemory(db_path=tmp_path / "test_memory.db")


def test_registration_in_memory_registry():
    """Importing the module registers 'sqlite' in MemoryRegistry."""
    MemoryRegistry.register_value("sqlite", SQLiteMemory)
    assert MemoryRegistry.contains("sqlite")


def test_creates_tables_on_init(tmp_path: Path):
    backend = _make_backend(tmp_path)
    # The Rust implementation and Python fallback both expose the same API.
    assert backend.count() == 0
    backend.close()


def test_store_returns_uuid(tmp_path: Path):
    backend = _make_backend(tmp_path)
    doc_id = backend.store("hello world")
    assert isinstance(doc_id, str)
    assert len(doc_id) == 36  # UUID4 strings include hyphens.
    backend.close()


def test_store_and_retrieve(tmp_path: Path):
    backend = _make_backend(tmp_path)
    backend.store("Python is a programming language", source="wiki.md")
    backend.store("The weather is sunny today", source="diary.md")
    results = backend.retrieve("programming language")
    assert len(results) >= 1
    assert "Python" in results[0].content
    backend.close()


def test_retrieve_ranking_by_relevance(tmp_path: Path):
    backend = _make_backend(tmp_path)
    backend.store("machine learning and deep learning")
    backend.store("cooking recipes for dinner")
    backend.store("advanced machine learning techniques")
    results = backend.retrieve("machine learning")
    assert len(results) >= 2
    # The ML-related docs should be first
    assert "machine" in results[0].content.lower()
    backend.close()


def test_retrieve_top_k_limit(tmp_path: Path):
    backend = _make_backend(tmp_path)
    for i in range(10):
        backend.store(f"document number {i} about testing")
    results = backend.retrieve("testing", top_k=3)
    assert len(results) <= 3
    backend.close()


def test_retrieve_no_results(tmp_path: Path):
    backend = _make_backend(tmp_path)
    backend.store("hello world")
    results = backend.retrieve("quantum physics supercollider")
    assert len(results) == 0
    backend.close()


def test_retrieve_query_with_apostrophe(tmp_path: Path):
    """Regression: an internal apostrophe (e.g. "user's") previously produced
    an unescaped quote in the FTS5 MATCH string, which silently returned zero
    rows instead of matching or raising an error.
    """
    backend = _make_backend(tmp_path)
    backend.store("The user's name is Trev.", source="identity.md")
    results = backend.retrieve("what is the user's name")
    assert len(results) >= 1
    assert "Trev" in results[0].content
    backend.close()


def test_delete_existing(tmp_path: Path):
    backend = _make_backend(tmp_path)
    doc_id = backend.store("deletable content")
    assert backend.count() == 1
    assert backend.delete(doc_id) is True
    assert backend.count() == 0
    backend.close()


def test_delete_nonexistent(tmp_path: Path):
    backend = _make_backend(tmp_path)
    assert backend.delete("nonexistent_id") is False
    backend.close()


def test_clear(tmp_path: Path):
    backend = _make_backend(tmp_path)
    backend.store("doc one")
    backend.store("doc two")
    assert backend.count() == 2
    backend.clear()
    assert backend.count() == 0
    backend.close()


def test_count(tmp_path: Path):
    backend = _make_backend(tmp_path)
    assert backend.count() == 0
    backend.store("first")
    assert backend.count() == 1
    backend.store("second")
    assert backend.count() == 2
    backend.close()


def test_source_and_metadata_roundtrip(tmp_path: Path):
    backend = _make_backend(tmp_path)
    meta = {"author": "test", "page": 42}
    backend.store(
        "content with metadata",
        source="paper.pdf",
        metadata=meta,
    )
    results = backend.retrieve("content metadata")
    assert len(results) == 1
    assert results[0].source == "paper.pdf"
    assert results[0].metadata["author"] == "test"
    assert results[0].metadata["page"] == 42
    backend.close()


def test_event_bus_integration_store(tmp_path: Path):
    bus = EventBus(record_history=True)
    backend = _make_backend(tmp_path)
    # Monkey-patch the global bus for this test
    import openjarvis.tools.storage.sqlite as mod

    original = mod.get_event_bus
    mod.get_event_bus = lambda: bus
    try:
        backend.store("test event emission")
        events = [e for e in bus.history if e.event_type == EventType.MEMORY_STORE]
        assert len(events) == 1
        assert events[0].data["backend"] == "sqlite"
    finally:
        mod.get_event_bus = original
    backend.close()


def test_event_bus_integration_retrieve(tmp_path: Path):
    bus = EventBus(record_history=True)
    backend = _make_backend(tmp_path)
    backend.store("searchable content for events")
    import openjarvis.tools.storage.sqlite as mod

    original = mod.get_event_bus
    mod.get_event_bus = lambda: bus
    try:
        backend.retrieve("searchable")
        events = [e for e in bus.history if e.event_type == EventType.MEMORY_RETRIEVE]
        assert len(events) == 1
        assert events[0].data["backend"] == "sqlite"
    finally:
        mod.get_event_bus = original
    backend.close()


def test_python_fallback_persists_and_manages_documents(tmp_path, monkeypatch):
    """The standard-library path remains fully usable without the Rust wheel."""
    import openjarvis._rust_bridge as bridge

    def missing_rust():
        raise ImportError("No module named 'openjarvis_rust'")

    monkeypatch.setattr(bridge, "get_rust_module", missing_rust)
    path = tmp_path / "nested" / "memory.db"
    backend = SQLiteMemory(path)
    assert backend._rust_impl is None

    first_id = backend.store(
        "The user's favorite editor is VS Code",
        source="profile.md",
        metadata={"kind": "preference", "rank": 2},
    )
    replaced_ids = backend.replace_source(
        "project.md",
        [
            ("ACE uses local inference", {"topic": "runtime"}),
            ("ACE stores knowledge in SQLite", None),
        ],
    )
    assert len(replaced_ids) == 2
    assert backend.count() == 3

    results = backend.retrieve("user's favorite editor", top_k=2)
    assert results
    assert results[0].source == "profile.md"
    assert results[0].metadata == {"kind": "preference", "rank": 2}
    assert "VS Code" in results[0].content

    assert backend.delete(first_id) is True
    assert backend.delete(first_id) is False
    backend.close()

    reopened = SQLiteMemory(path)
    assert reopened.count() == 2
    assert reopened.retrieve("knowledge SQLite")[0].source == "project.md"
    reopened.clear()
    assert reopened.count() == 0
    reopened.close()


def test_python_fallback_handles_reserved_words_and_punctuation(tmp_path, monkeypatch):
    import openjarvis._rust_bridge as bridge

    monkeypatch.setattr(
        bridge,
        "get_rust_module",
        lambda: (_ for _ in ()).throw(ImportError("extension missing")),
    )
    backend = SQLiteMemory(tmp_path / "memory.db")
    backend.store("The OR operator and NEAR keyword are documented")
    results = backend.retrieve('"OR" NEAR operator')
    assert results
    assert "operator" in results[0].content
    backend.close()
