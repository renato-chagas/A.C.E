"""Local semantic index for durable memory facts.

The human-readable JSONL fact store remains authoritative. This sidecar keeps
only model-versioned float vectors keyed by a stable hash of fact text, so
semantic recall can be rebuilt or deleted without changing the fact format.
All embedding requests are restricted to a loopback Ollama service.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Sequence

from openjarvis.connectors.embeddings import decode_embedding
from openjarvis.security.file_utils import secure_create

if TYPE_CHECKING:
    from openjarvis.connectors.embeddings import OllamaEmbedder
    from openjarvis.memory.store import Fact

logger = logging.getLogger(__name__)
_DEFAULT_MODEL = "nomic-embed-text"


def fact_embedding_index_path(facts_path: str | Path) -> Path:
    """Return the private SQLite sidecar path for a JSONL fact store."""
    path = Path(facts_path).expanduser()
    return path.with_name(f"{path.stem}.vectors.db")


def _fact_key(text: str) -> str:
    normalized = (text or "").strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _recallable(fact: Fact) -> bool:
    return bool(getattr(fact, "trusted_for_recall", True))


class FactEmbeddingIndex:
    """SQLite vector sidecar for query-aware durable-fact retrieval."""

    def __init__(
        self,
        path: str | Path,
        *,
        embedding_model: str = _DEFAULT_MODEL,
        embedder: OllamaEmbedder | None = None,
    ) -> None:
        self._path = Path(path).expanduser()
        self._embedding_model = (embedding_model or _DEFAULT_MODEL).strip()
        self._embedder = embedder
        self._lock = RLock()
        if str(self._path) == ":memory:":
            db_path = ":memory:"
        else:
            secure_create(self._path)
            db_path = str(self._path)
        self._conn = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS fact_embeddings (
                fact_key TEXT PRIMARY KEY,
                model_version TEXT NOT NULL,
                vector BLOB NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS fact_embeddings_model_idx "
            "ON fact_embeddings(model_version)"
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> FactEmbeddingIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get_embedder(self) -> OllamaEmbedder | None:
        if self._embedder is None:
            # Reuse the memory subsystem's loopback-only resolver. It refuses
            # remote OLLAMA_HOST values and disables environment proxies.
            from openjarvis.memory.episodes import _local_embedder

            self._embedder = _local_embedder(self._embedding_model)
        return self._embedder

    def index_missing_embeddings(
        self,
        facts: Sequence[Fact],
        *,
        limit: int = 100,
    ) -> tuple[int, int]:
        """Embed up to *limit* recallable facts; return (indexed, remaining)."""
        current = {
            _fact_key(fact.text): fact
            for fact in facts
            if _recallable(fact) and str(getattr(fact, "text", "")).strip()
        }
        if not current:
            with self._lock:
                self._prune_locked(set())
            return 0, 0
        embedder = self._get_embedder()
        if embedder is None:
            raise RuntimeError(
                "Local fact embeddings are unavailable. Start loopback Ollama "
                f"and install {self._embedding_model!r}."
            )

        model_version = embedder.model_version
        with self._lock:
            self._prune_locked(set(current))
            rows = self._conn.execute(
                "SELECT fact_key FROM fact_embeddings WHERE model_version=?",
                (model_version,),
            ).fetchall()
            indexed_keys = {str(row[0]) for row in rows}
        missing = [key for key in current if key not in indexed_keys]
        selected = missing[: max(0, min(int(limit), 1_000))]
        indexed = 0
        try:
            vectors = embedder.embed_batch([current[key].text for key in selected])
        except Exception:
            logger.debug("Local durable-fact batch embedding failed", exc_info=True)
            vectors = []
        for key, vector in zip(selected, vectors):
            if not vector:
                continue
            with self._lock:
                self._conn.execute(
                    "INSERT INTO fact_embeddings "
                    "(fact_key, model_version, vector, updated_at) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(fact_key) DO UPDATE SET "
                    "model_version=excluded.model_version, vector=excluded.vector, "
                    "updated_at=excluded.updated_at",
                    (key, model_version, vector, time.time()),
                )
                self._conn.commit()
            indexed += 1
        with self._lock:
            remaining_rows = self._conn.execute(
                "SELECT fact_key FROM fact_embeddings WHERE model_version=?",
                (model_version,),
            ).fetchall()
        remaining_keys = {str(row[0]) for row in remaining_rows}
        return indexed, sum(key not in remaining_keys for key in current)

    def rank(
        self,
        facts: Sequence[Fact],
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[tuple[Fact, float]]:
        """Return indexed facts ordered by local cosine similarity."""
        if not query or not query.strip():
            return []
        try:
            current = {
                _fact_key(fact.text): fact
                for fact in facts
                if _recallable(fact) and str(getattr(fact, "text", "")).strip()
            }
            with self._lock:
                self._prune_locked(set(current))
                model_versions = self._conn.execute(
                    "SELECT DISTINCT model_version FROM fact_embeddings"
                ).fetchall()
            if not current:
                return []
            embedder = self._get_embedder()
            if embedder is None:
                return []
            model_version = embedder.model_version
            if not any(str(row[0]) == model_version for row in model_versions):
                return []
            query_blob = embedder.embed(query[:6_000])
            query_vector = decode_embedding(query_blob)
            if query_vector is None or not query_vector.size:
                return []
            import numpy as np

            query_norm = float(np.linalg.norm(query_vector))
            if not query_norm:
                return []
            with self._lock:
                rows = self._conn.execute(
                    "SELECT fact_key, vector FROM fact_embeddings "
                    "WHERE model_version=?",
                    (model_version,),
                ).fetchall()
            ranked: list[tuple[Fact, float]] = []
            for row in rows:
                fact = current.get(str(row["fact_key"]))
                vector = decode_embedding(row["vector"])
                if fact is None or vector is None or vector.shape != query_vector.shape:
                    continue
                denominator = query_norm * float(np.linalg.norm(vector))
                if denominator <= 0:
                    continue
                score = float(np.dot(query_vector, vector) / denominator)
                if not np.isfinite(score):
                    continue
                ranked.append((fact, score))
            ranked.sort(
                key=lambda item: (item[1], float(getattr(item[0], "created_at", 0))),
                reverse=True,
            )
            return ranked[:top_k] if top_k is not None else ranked
        except Exception:
            logger.debug("Local durable-fact semantic search failed", exc_info=True)
            return []

    def prune_to(self, facts: Sequence[Fact]) -> int:
        """Remove vectors for deleted or quarantined facts; return rows removed."""
        allowed = {
            _fact_key(fact.text)
            for fact in facts
            if _recallable(fact) and str(getattr(fact, "text", "")).strip()
        }
        with self._lock:
            return self._prune_locked(allowed)

    def clear(self) -> int:
        """Remove every durable-fact vector from this sidecar."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM fact_embeddings")
            self._conn.commit()
            return max(0, cursor.rowcount)

    def _prune_locked(self, allowed: set[str]) -> int:
        rows = self._conn.execute(
            "SELECT fact_key FROM fact_embeddings"
        ).fetchall()
        stale = [str(row[0]) for row in rows if str(row[0]) not in allowed]
        if not stale:
            return 0
        self._conn.executemany(
            "DELETE FROM fact_embeddings WHERE fact_key=?",
            ((key,) for key in stale),
        )
        self._conn.commit()
        return len(stale)


__all__ = ["FactEmbeddingIndex", "fact_embedding_index_path"]
