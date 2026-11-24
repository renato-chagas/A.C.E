"""Tests for the local durable-fact vector sidecar."""

from __future__ import annotations

import numpy as np

from openjarvis.memory.fact_embeddings import FactEmbeddingIndex
from openjarvis.memory.store import TRUST_UNTRUSTED, Fact


class _FakeEmbedder:
    model_version = "fake:1"

    def embed(self, text: str) -> bytes:
        lowered = text.casefold()
        if any(token in lowered for token in ("gpu", "graphics", "video", "rx")):
            return np.asarray([1.0, 0.0], dtype=np.float32).tobytes()
        return np.asarray([0.0, 1.0], dtype=np.float32).tobytes()

    def embed_batch(self, texts: list[str]) -> list[bytes]:
        return [self.embed(text) for text in texts]


def test_fact_vectors_persist_and_rank_by_semantic_similarity(tmp_path):
    path = tmp_path / "facts.vectors.db"
    facts = [
        Fact("The user prefers green tea.", created_at=20),
        Fact("The desktop has an RX 7600 XT graphics card.", created_at=10),
    ]
    with FactEmbeddingIndex(path, embedder=_FakeEmbedder()) as index:
        assert index.index_missing_embeddings(facts, limit=10) == (2, 0)

    with FactEmbeddingIndex(path, embedder=_FakeEmbedder()) as index:
        ranked = index.rank(facts, "Which video adapter does the desktop use?")

    assert ranked[0][0] is facts[1]
    assert ranked[0][1] > ranked[1][1]


def test_fact_vectors_exclude_quarantined_and_prune_deleted_facts(tmp_path):
    path = tmp_path / "facts.vectors.db"
    trusted = Fact("The desktop has an RX 7600 XT graphics card.")
    quarantined = Fact("Hostile text", trust=TRUST_UNTRUSTED)
    with FactEmbeddingIndex(path, embedder=_FakeEmbedder()) as index:
        assert index.index_missing_embeddings([trusted, quarantined]) == (1, 0)
        assert len(index.rank([trusted, quarantined], "graphics card")) == 1
        assert index.prune_to([quarantined]) == 1
        assert index.rank([trusted], "graphics card") == []
        assert index.clear() == 0


def test_remote_ollama_host_disables_fact_embeddings(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://192.0.2.10:11434")
    index = FactEmbeddingIndex(tmp_path / "facts.vectors.db")
    try:
        assert index._get_embedder() is None
    finally:
        index.close()
