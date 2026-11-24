"""SQLite/FTS5 memory backend with an optional Rust acceleration path."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from openjarvis.core.events import EventType, get_event_bus
from openjarvis.core.registry import MemoryRegistry
from openjarvis.tools.storage._stubs import (
    MemoryBackend,
    MemoryBackendUnavailable,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

_FTS5_UNAVAILABLE_HINT = (
    "The built-in SQLite runtime does not provide FTS5, which the SQLite "
    "memory backend requires. Install a Python build with SQLite FTS5 support "
    "or build the optional `openjarvis_rust` extension."
)


def _check_fts5(conn: sqlite3.Connection) -> bool:
    """Return whether this SQLite runtime can create and query FTS5 tables."""
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE temp.openjarvis_fts5_probe USING fts5(content)"
        )
        conn.execute("DROP TABLE temp.openjarvis_fts5_probe")
        return True
    except sqlite3.Error:
        try:
            conn.execute("DROP TABLE IF EXISTS temp.openjarvis_fts5_probe")
        except sqlite3.Error:
            pass
        return False


@MemoryRegistry.register("sqlite")
class SQLiteMemory(MemoryBackend):
    """Full-text search memory backend using SQLite FTS5.

    The native Rust implementation is used when installed. Python's standard
    library ``sqlite3`` implementation provides the same persistent behavior
    when the optional extension is absent.
    """

    backend_id: str = "sqlite"

    def __init__(self, db_path: str | Path = "") -> None:
        if not db_path:
            from openjarvis.core.config import DEFAULT_CONFIG_DIR

            db_path = DEFAULT_CONFIG_DIR / "memory.db"

        path = (
            Path(db_path).expanduser()
            if str(db_path) != ":memory:"
            else Path(":memory:")
        )
        self._db_path = str(path)
        self._lock = threading.RLock()
        self._rust_impl: Any | None = None
        self._conn: sqlite3.Connection | None = None

        if self._db_path != ":memory:":
            from openjarvis.security.file_utils import secure_create

            secure_create(path)

        try:
            from openjarvis._rust_bridge import get_rust_module

            rust = get_rust_module()
        except ImportError:
            rust = None

        if rust is not None:
            self._rust_impl = rust.SQLiteMemory(self._db_path)
            return

        logger.warning(
            "openjarvis_rust is unavailable; using the built-in Python SQLite/FTS5 "
            "memory backend."
        )
        conn = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False)
        conn.execute("PRAGMA busy_timeout = 10000")
        if str(path) != ":memory:":
            try:
                conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.Error:
                logger.warning("Could not enable SQLite WAL mode for memory storage")
        if not _check_fts5(conn):
            conn.close()
            raise MemoryBackendUnavailable(_FTS5_UNAVAILABLE_HINT)
        self._conn = conn
        try:
            self._create_tables()
        except Exception:
            self._conn.close()
            self._conn = None
            raise

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SQLite memory backend is closed")
        return self._conn

    def _create_tables(self) -> None:
        """Create or migrate the document and FTS tables."""
        conn = self._connection()
        with self._lock, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL DEFAULT (julianday('now'))
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
                USING fts5(content, source, tokenize='porter unicode61');
                """
            )
            fts_sql_row = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND name='documents_fts'"
            ).fetchone()
            fts_sql = str(fts_sql_row[0] or "") if fts_sql_row else ""
            fts_columns = [
                str(row[1]) for row in conn.execute("PRAGMA table_info(documents_fts)")
            ]
            if fts_columns[:2] != ["content", "source"] or not all(
                marker in fts_sql.lower() for marker in ("porter", "unicode61")
            ):
                conn.execute("DROP TABLE documents_fts")
                conn.execute(
                    "CREATE VIRTUAL TABLE documents_fts USING fts5("
                    "content, source, tokenize='porter unicode61')"
                )
                conn.execute(
                    "INSERT INTO documents_fts(rowid, content, source) "
                    "SELECT rowid, content, source FROM documents"
                )

    @staticmethod
    def _metadata_json(metadata: Optional[Dict[str, Any]]) -> str:
        return json.dumps(metadata if metadata is not None else {}, ensure_ascii=False)

    @staticmethod
    def _publish_store_event(doc_id: str, source: str) -> None:
        get_event_bus().publish(
            EventType.MEMORY_STORE,
            {"backend": "sqlite", "doc_id": doc_id, "source": source},
        )

    def store(
        self,
        content: str,
        *,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist *content* and return a unique document id."""
        if self._rust_impl is not None:
            doc_id = self._rust_impl.store(
                content, source, json.dumps(metadata) if metadata else None
            )
        else:
            doc_id = str(uuid.uuid4())
            conn = self._connection()
            with self._lock, conn:
                cursor = conn.execute(
                    "INSERT INTO documents(id, content, source, metadata) "
                    "VALUES (?, ?, ?, ?)",
                    (doc_id, content, source, self._metadata_json(metadata)),
                )
                conn.execute(
                    "INSERT INTO documents_fts(rowid, content, source) "
                    "VALUES (?, ?, ?)",
                    (cursor.lastrowid, content, source),
                )
        self._publish_store_event(doc_id, source)
        return doc_id

    def replace_source(
        self,
        source: str,
        documents: List[tuple[str, Optional[Dict[str, Any]]]],
    ) -> List[str]:
        """Atomically replace all documents associated with *source*."""
        if self._rust_impl is not None:
            payload = [
                (content, json.dumps(metadata) if metadata else None)
                for content, metadata in documents
            ]
            doc_ids = self._rust_impl.replace_source(source, payload)
        else:
            doc_ids = []
            conn = self._connection()
            with self._lock, conn:
                conn.execute(
                    "DELETE FROM documents_fts WHERE rowid IN "
                    "(SELECT rowid FROM documents WHERE source = ?)",
                    (source,),
                )
                conn.execute("DELETE FROM documents WHERE source = ?", (source,))
                for content, metadata in documents:
                    doc_id = str(uuid.uuid4())
                    cursor = conn.execute(
                        "INSERT INTO documents(id, content, source, metadata) "
                        "VALUES (?, ?, ?, ?)",
                        (doc_id, content, source, self._metadata_json(metadata)),
                    )
                    conn.execute(
                        "INSERT INTO documents_fts(rowid, content, source) "
                        "VALUES (?, ?, ?)",
                        (cursor.lastrowid, content, source),
                    )
                    doc_ids.append(doc_id)
        for doc_id in doc_ids:
            self._publish_store_event(doc_id, source)
        return doc_ids

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Search by token using FTS5 and return best BM25 matches."""
        if not query.strip() or top_k <= 0:
            return []

        if self._rust_impl is not None:
            from openjarvis._rust_bridge import retrieval_results_from_json

            results = retrieval_results_from_json(
                self._rust_impl.retrieve(query, top_k)
            )
        else:
            words = _query_words(query)[:128]
            if not words:
                return []
            fts_query = " OR ".join(f'"{word}"' for word in words)
            conn = self._connection()
            with self._lock:
                rows = conn.execute(
                    "SELECT d.content, d.source, d.metadata, "
                    "-bm25(documents_fts, 1.0, 0.5) AS score "
                    "FROM documents_fts "
                    "JOIN documents AS d ON d.rowid = documents_fts.rowid "
                    "WHERE documents_fts MATCH ? "
                    "ORDER BY bm25(documents_fts, 1.0, 0.5) LIMIT ?",
                    (fts_query, int(top_k)),
                ).fetchall()
            results = []
            for content, source, metadata, score in rows:
                try:
                    parsed_metadata = json.loads(metadata or "{}")
                except (TypeError, json.JSONDecodeError):
                    parsed_metadata = {}
                results.append(
                    RetrievalResult(
                        content=content,
                        source=source or "",
                        metadata=(
                            parsed_metadata if isinstance(parsed_metadata, dict) else {}
                        ),
                        score=float(score or 0.0),
                    )
                )

        get_event_bus().publish(
            EventType.MEMORY_RETRIEVE,
            {"backend": self.backend_id, "query": query, "num_results": len(results)},
        )
        return results

    def delete(self, doc_id: str) -> bool:
        """Delete a document by id."""
        if self._rust_impl is not None:
            return bool(self._rust_impl.delete(doc_id))
        conn = self._connection()
        with self._lock, conn:
            conn.execute(
                "DELETE FROM documents_fts WHERE rowid = "
                "(SELECT rowid FROM documents WHERE id = ?)",
                (doc_id,),
            )
            cursor = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Remove all stored documents."""
        if self._rust_impl is not None:
            self._rust_impl.clear()
            return
        conn = self._connection()
        with self._lock, conn:
            conn.execute("DELETE FROM documents_fts")
            conn.execute("DELETE FROM documents")

    def count(self) -> int:
        """Return the number of stored documents."""
        if self._rust_impl is not None:
            return int(self._rust_impl.count())
        conn = self._connection()
        with self._lock:
            return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def close(self) -> None:
        """Close the Python database connection when using the fallback."""
        if self._rust_impl is not None:
            close = getattr(self._rust_impl, "close", None)
            if callable(close):
                close()
            self._rust_impl = None
        if self._conn is not None:
            with self._lock:
                self._conn.close()
                self._conn = None


def _query_words(query: str) -> List[str]:
    """Split punctuation safely and preserve the Rust backend's token rules."""
    words: List[str] = []
    current: List[str] = []
    for char in query:
        if char.isalnum():
            current.append(char)
        elif current:
            words.append("".join(current))
            current.clear()
    if current:
        words.append("".join(current))
    return list(dict.fromkeys(words))


__all__ = ["SQLiteMemory"]
