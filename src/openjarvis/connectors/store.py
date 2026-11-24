"""KnowledgeStore — source-aware SQLite/FTS5 memory backend for Deep Research.

Extends ``MemoryBackend`` with per-document provenance columns so that the
IngestionPipeline and the ``knowledge_search`` tool can filter results by
source, doc_type, author, and timestamp ranges.

Pure Python ``sqlite3`` (no Rust extension required).
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from openjarvis.core.events import EventType, get_event_bus
from openjarvis.core.registry import MemoryRegistry
from openjarvis.tools.storage._stubs import MemoryBackend, RetrievalResult

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_MAIN_TABLE = """
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id            TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    content       TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT '',
    source_id     TEXT NOT NULL DEFAULT '',
    doc_type      TEXT NOT NULL DEFAULT '',
    title         TEXT NOT NULL DEFAULT '',
    author        TEXT NOT NULL DEFAULT '',
    participants  TEXT NOT NULL DEFAULT '[]',
    participants_raw TEXT NOT NULL DEFAULT '[]',
    timestamp     TEXT NOT NULL DEFAULT '',
    thread_id     TEXT NOT NULL DEFAULT '',
    channel       TEXT NOT NULL DEFAULT '',
    url           TEXT NOT NULL DEFAULT '',
    metadata      TEXT NOT NULL DEFAULT '{}',
    chunk_index   INTEGER NOT NULL DEFAULT 0,
    embedding     BLOB,
    embedding_model_version TEXT NOT NULL DEFAULT '',
    content_hash  TEXT NOT NULL DEFAULT '',
    deleted_at    REAL,
    last_synced   REAL NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL
);
"""

# Additive migrations: existing installs get new columns via ALTER TABLE.
# Order doesn't matter — each entry is independent and idempotent.
_V1_COLUMNS: List[tuple[str, str]] = [
    ("source_id", "TEXT NOT NULL DEFAULT ''"),
    ("participants_raw", "TEXT NOT NULL DEFAULT '[]'"),
    ("channel", "TEXT NOT NULL DEFAULT ''"),
    ("embedding", "BLOB"),
    ("embedding_model_version", "TEXT NOT NULL DEFAULT ''"),
    ("content_hash", "TEXT NOT NULL DEFAULT ''"),
    ("deleted_at", "REAL"),
    ("last_synced", "REAL NOT NULL DEFAULT 0"),
]

_CREATE_FTS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts
USING fts5(
    content,
    title,
    author,
    content='knowledge_chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
"""

_CREATE_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge_chunks BEGIN
    INSERT INTO knowledge_fts(rowid, content, title, author)
    VALUES (new.rowid, new.content, new.title, new.author);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge_chunks BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, content, title, author)
    VALUES ('delete', old.rowid, old.content, old.title, old.author);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge_chunks BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, content, title, author)
    VALUES ('delete', old.rowid, old.content, old.title, old.author);
    INSERT INTO knowledge_fts(rowid, content, title, author)
    VALUES (new.rowid, new.content, new.title, new.author);
END;
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_kc_source        ON knowledge_chunks(source);
CREATE INDEX IF NOT EXISTS idx_kc_doc_type      ON knowledge_chunks(doc_type);
CREATE INDEX IF NOT EXISTS idx_kc_author        ON knowledge_chunks(author);
CREATE INDEX IF NOT EXISTS idx_kc_timestamp     ON knowledge_chunks(timestamp);
CREATE INDEX IF NOT EXISTS idx_kc_thread_id     ON knowledge_chunks(thread_id);
CREATE INDEX IF NOT EXISTS idx_kc_doc_id        ON knowledge_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_kc_source_id     ON knowledge_chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_kc_content_hash  ON knowledge_chunks(content_hash);
CREATE UNIQUE INDEX IF NOT EXISTS idx_kc_natural_key
    ON knowledge_chunks(source, source_id, chunk_index)
    WHERE source_id != '';
"""

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _plain_fts_query(text: str) -> str:
    return " OR ".join(f'"{term}"' for term in re.findall(r"\w+", text))


def _to_iso(ts: Optional[Union[datetime, str]]) -> str:
    """Normalise a timestamp to ISO 8601 string (UTC)."""
    if ts is None:
        return ""
    if isinstance(ts, str):
        return ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def _to_epoch(ts: Union[datetime, str, float, int]) -> float:
    """Normalise a timestamp to Unix epoch seconds (float)."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        # Treat empty/zero-ish strings as epoch 0; otherwise parse ISO 8601.
        if not ts:
            return 0.0
        return datetime.fromisoformat(ts).timestamp()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.timestamp()


# ---------------------------------------------------------------------------
# KnowledgeStore
# ---------------------------------------------------------------------------


@MemoryRegistry.register("knowledge")
class KnowledgeStore(MemoryBackend):
    """Source-aware SQLite/FTS5 knowledge store for Deep Research.

    Stores document chunks with rich provenance metadata and supports
    filtered BM25 retrieval by source, doc_type, author, and timestamp.
    """

    backend_id: str = "knowledge"

    def __init__(self, db_path: Union[str, Path] = "") -> None:
        if not db_path:
            from openjarvis.core.config import DEFAULT_CONFIG_DIR

            db_path = DEFAULT_CONFIG_DIR / "knowledge.db"

        self._db_path = str(Path(db_path).expanduser())
        # Ensure the parent directory exists (skip for :memory:)
        if self._db_path != ":memory:":
            from openjarvis.security.file_utils import secure_create

            secure_create(Path(self._db_path))

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Create tables, FTS virtual table, triggers and indexes."""
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.executescript(
            _CREATE_MAIN_TABLE + _CREATE_FTS_TABLE + _CREATE_TRIGGERS
        )
        self._migrate_v1_columns()
        # Indexes reference the new columns, so they run after migrations.
        self._conn.executescript(_CREATE_INDEXES)
        self._conn.commit()

    def _migrate_v1_columns(self) -> None:
        """Add v1 schema columns to pre-existing tables (idempotent)."""
        existing = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(knowledge_chunks)"
            ).fetchall()
        }
        for col_name, col_def in _V1_COLUMNS:
            if col_name not in existing:
                self._conn.execute(
                    f"ALTER TABLE knowledge_chunks ADD COLUMN {col_name} {col_def}"
                )

    # ------------------------------------------------------------------
    # MemoryBackend interface
    # ------------------------------------------------------------------

    def store(  # type: ignore[override]
        self,
        content: str,
        *,
        source: str = "",
        doc_type: str = "",
        doc_id: Optional[str] = None,
        title: str = "",
        author: str = "",
        participants: Optional[List[str]] = None,
        timestamp: Optional[Union[datetime, str]] = None,
        thread_id: Optional[str] = None,
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_index: int = 0,
        # v1 schema additions
        source_id: str = "",
        participants_raw: Optional[List[str]] = None,
        channel: Optional[str] = None,
        content_hash: str = "",
        embedding: Optional[bytes] = None,
        embedding_model_version: str = "",
        last_synced: Optional[Union[datetime, str, float]] = None,
    ) -> str:
        """Persist a content chunk and return its unique chunk id.

        All source-level fields are merged into the stored metadata so that
        ``retrieve()`` results carry full provenance.
        """
        chunk_id = str(uuid.uuid4())
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        ts_str = _to_iso(timestamp)
        participants_json = json.dumps(participants or [])
        participants_raw_json = json.dumps(participants_raw or [])
        last_synced_epoch = (
            _to_epoch(last_synced) if last_synced is not None else time.time()
        )

        # Merge provenance fields into metadata for easy access in results
        combined_meta: Dict[str, Any] = dict(metadata or {})
        combined_meta["chunk_id"] = chunk_id
        combined_meta["source"] = source
        combined_meta["source_id"] = source_id
        combined_meta["doc_type"] = doc_type
        combined_meta["doc_id"] = doc_id
        combined_meta["title"] = title
        combined_meta["author"] = author
        combined_meta["participants"] = participants or []
        combined_meta["participants_raw"] = participants_raw or []
        combined_meta["timestamp"] = ts_str
        combined_meta["thread_id"] = thread_id or ""
        combined_meta["channel"] = channel or ""
        combined_meta["url"] = url or ""
        combined_meta["chunk_index"] = chunk_index
        combined_meta["content_hash"] = content_hash
        combined_meta["embedding_model_version"] = embedding_model_version

        meta_json = json.dumps(combined_meta)

        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO knowledge_chunks
                (id, doc_id, content, source, source_id, doc_type, title, author,
                 participants, participants_raw, timestamp, thread_id, channel,
                 url, metadata, chunk_index, embedding, embedding_model_version,
                 content_hash, last_synced, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                doc_id,
                content,
                source,
                source_id,
                doc_type,
                title,
                author,
                participants_json,
                participants_raw_json,
                ts_str,
                thread_id or "",
                channel or "",
                url or "",
                meta_json,
                chunk_index,
                embedding,
                embedding_model_version,
                content_hash,
                last_synced_epoch,
                time.time(),
            ),
        )
        self._conn.commit()

        # If the natural-key (source, source_id, chunk_index) already existed
        # SQLite skipped the insert. Re-runs of the dogfood script or a
        # SyncEngine that re-emits a known document hit this path; return
        # the existing chunk id so callers don't see two different identities
        # for the same row, and suppress the MEMORY_STORE event (nothing new
        # was actually written).
        if cur.rowcount == 0:
            existing = self._conn.execute(
                "SELECT id FROM knowledge_chunks "
                "WHERE source=? AND source_id=? AND chunk_index=?",
                (source, source_id, chunk_index),
            ).fetchone()
            if existing is not None:
                return existing["id"]
            return chunk_id

        get_event_bus().publish(
            EventType.MEMORY_STORE,
            {
                "backend": self.backend_id,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "source": source,
                "doc_type": doc_type,
            },
        )
        return chunk_id

    def store_web_capture(
        self,
        content: str,
        *,
        url: str,
        doc_id: str,
        title: str,
        timestamp: str,
        content_hash: str,
        search_query: str,
        search_engine: str,
        is_page_text: bool,
    ) -> bool:
        """Atomically refresh a web record. A snippet cannot replace page text."""
        chunk_id = str(uuid.uuid4())
        now = time.time()
        metadata = json.dumps(
            {
                "chunk_id": chunk_id,
                "source": "web",
                "source_id": url,
                "doc_type": "web_page",
                "doc_id": doc_id,
                "title": title,
                "timestamp": timestamp,
                "url": url,
                "chunk_index": 0,
                "content_hash": content_hash,
                "search_query": search_query,
                "search_engine": search_engine,
                "capture_kind": "page" if is_page_text else "snippet",
            }
        )
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO knowledge_chunks (
                    id, doc_id, content, source, source_id, doc_type, title,
                    timestamp, url, metadata, content_hash, last_synced, created_at
                ) VALUES (?, ?, ?, 'web', ?, 'web_page', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, source_id, chunk_index) WHERE source_id != ''
                DO UPDATE SET id=excluded.id, doc_id=excluded.doc_id,
                    content=excluded.content, title=excluded.title,
                    timestamp=excluded.timestamp, url=excluded.url,
                    metadata=excluded.metadata, content_hash=excluded.content_hash,
                    last_synced=excluded.last_synced, created_at=excluded.created_at,
                    embedding=NULL, embedding_model_version='', deleted_at=NULL
                WHERE knowledge_chunks.doc_type = 'web_page' AND (
                    knowledge_chunks.deleted_at IS NOT NULL OR ? OR
                    CASE WHEN json_valid(knowledge_chunks.metadata) THEN
                        json_extract(knowledge_chunks.metadata, '$.capture_kind')
                            = 'snippet'
                    ELSE 0 END
                )
                """,
                (
                    chunk_id,
                    doc_id,
                    content,
                    url,
                    title,
                    timestamp,
                    url,
                    metadata,
                    content_hash,
                    now,
                    now,
                    is_page_text,
                ),
            )
        if cursor.rowcount == 0:
            return False
        get_event_bus().publish(
            EventType.MEMORY_STORE,
            {
                "backend": self.backend_id,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "source": "web",
                "doc_type": "web_page",
            },
        )
        return True

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        source: Optional[str] = None,
        doc_type: Optional[str] = None,
        author: Optional[str] = None,
        since: Optional[Union[datetime, str]] = None,
        until: Optional[Union[datetime, str]] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Search using FTS5 BM25 with optional column filters.

        Parameters
        ----------
        query:    Plain-text search query (FTS5 punctuation is handled internally).
        top_k:    Maximum number of results.
        source:   Restrict to chunks from this source (e.g. "gmail").
        doc_type: Restrict to chunks of this type (e.g. "email").
        author:   Restrict to chunks authored by this person.
        since:    Exclude chunks whose timestamp is earlier than this value.
        until:    Exclude chunks whose timestamp is later than this value.
        """
        # Callers pass ordinary text, not FTS5 expressions. Punctuation such
        # as apostrophes, hyphens and question marks is otherwise parsed as
        # query syntax and can turn a matching search into an empty result.
        fts_query = _plain_fts_query(query)
        if not fts_query:
            return []

        def _clean_str(val: Optional[str]) -> Optional[str]:
            if isinstance(val, str):
                return val.strip() or None
            return val

        source_val = _clean_str(source)
        doc_type_val = _clean_str(doc_type)
        author_val = _clean_str(author)

        def _clean_ts(val: Union[datetime, str, None]) -> Union[datetime, str, None]:
            if isinstance(val, str):
                s = val.strip()
                return s if s else None
            return val

        since_val = _clean_ts(since)
        until_val = _clean_ts(until)

        since_str = _to_iso(since_val) if since_val is not None else None
        until_str = _to_iso(until_val) if until_val is not None else None

        # Build the WHERE clause for filter columns
        filters: List[str] = []
        params: List[Any] = []

        if source_val is not None:
            filters.append("kc.source = ?")
            params.append(source_val)
        if doc_type_val is not None:
            filters.append("kc.doc_type = ?")
            params.append(doc_type_val)
        if author_val is not None:
            filters.append("kc.author = ?")
            params.append(author_val)
        if since_str:
            filters.append("kc.timestamp >= ?")
            params.append(since_str)
        if until_str:
            filters.append("kc.timestamp <= ?")
            params.append(until_str)

        # Always exclude tombstoned rows.
        filters.append("kc.deleted_at IS NULL")

        where_clause = "AND " + " AND ".join(filters)

        # FTS5 bm25() returns negative scores; abs() gives a positive rank
        sql = f"""
            SELECT
                kc.id,
                kc.content,
                kc.source,
                kc.metadata,
                abs(bm25(knowledge_fts)) AS score
            FROM knowledge_fts
            JOIN knowledge_chunks kc ON knowledge_fts.rowid = kc.rowid
            WHERE knowledge_fts MATCH ?
            {where_clause}
            ORDER BY score DESC
            LIMIT ?
        """

        try:
            rows = self._conn.execute(sql, [fts_query] + params + [top_k]).fetchall()
        except sqlite3.OperationalError as exc:
            # Only malformed FTS expressions are empty searches. Other SQLite
            # failures must reach callers rather than masquerade as no matches.
            if str(exc).lower().startswith("fts5: syntax error"):
                return []
            raise

        results: List[RetrievalResult] = []
        for row in rows:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            # Ensure chunk_id is always present in metadata (backfill for
            # rows stored before this field was added to combined_meta).
            if "chunk_id" not in meta:
                meta["chunk_id"] = row["id"]
            results.append(
                RetrievalResult(
                    content=row["content"],
                    score=float(row["score"]),
                    source=row["source"],
                    metadata=meta,
                )
            )

        get_event_bus().publish(
            EventType.MEMORY_RETRIEVE,
            {
                "backend": self.backend_id,
                "query": query,
                "num_results": len(results),
                "filters": {
                    "source": source,
                    "doc_type": doc_type,
                    "author": author,
                    "since": since_str,
                    "until": until_str,
                },
            },
        )
        return results

    @staticmethod
    def read_web_captures(
        db_path: str | Path,
        *,
        query: str = "",
        urls: tuple[str, ...] = (),
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """List or search saved web captures without opening a writable store."""
        path = Path(db_path).expanduser()
        if not path.exists():
            return []
        fts_query = _plain_fts_query(query)
        if query and not fts_query and not urls:
            return []

        with closing(
            sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
        ) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(knowledge_chunks)")
            }
            if not columns:
                return []
            where = [
                "kc.source = 'web'",
                "kc.doc_type = 'web_page'",
                "kc.url <> ''",
            ]
            if "deleted_at" in columns:
                where.append("kc.deleted_at IS NULL")
            params: list[str | int] = []
            table = "knowledge_chunks AS kc"
            order = "kc.created_at DESC, kc.id"
            if urls:
                placeholders = ", ".join("?" for _ in urls)
                where.append(f"kc.url IN ({placeholders})")
                params.extend(urls)
            elif fts_query:
                table += " JOIN knowledge_fts ON knowledge_fts.rowid = kc.rowid"
                where.append("knowledge_fts MATCH ?")
                params.append(fts_query)
                order = "bm25(knowledge_fts), kc.created_at DESC, kc.id"
            params.append(limit)
            rows = conn.execute(
                "SELECT kc.id AS revision, kc.url, substr(kc.title, 1, 400) AS title, "
                "kc.timestamp AS saved_at, substr(kc.content, 1, 9000) AS text, "
                "CASE WHEN json_valid(kc.metadata) "
                "THEN json_extract(kc.metadata, '$.capture_kind') END AS kind "
                f"FROM {table} WHERE {' AND '.join(where)} ORDER BY {order} LIMIT ?",
                params,
            ).fetchall()

        return [
            {
                "url": row["url"],
                "revision": row["revision"],
                "title": row["title"] or row["url"],
                "saved_at": row["saved_at"],
                "text": row["text"],
                "capture": row["kind"]
                if row["kind"] in ("page", "snippet")
                else "unknown",
            }
            for row in rows
        ]

    def retrieve_by_url(
        self,
        url: str,
        *,
        top_k: int = 5,
        source: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Return saved chunks for an exact URL without requiring FTS terms."""
        url_value = url.strip()
        limit = max(0, int(top_k))
        if not url_value or limit == 0:
            return []

        filters = ["url = ?", "deleted_at IS NULL"]
        params: List[Any] = [url_value]
        source_value = source.strip() if isinstance(source, str) else None
        doc_type_value = doc_type.strip() if isinstance(doc_type, str) else None
        if source_value:
            filters.append("source = ?")
            params.append(source_value)
        if doc_type_value:
            filters.append("doc_type = ?")
            params.append(doc_type_value)

        rows = self._conn.execute(
            "SELECT id, content, source, url, metadata FROM knowledge_chunks "
            f"WHERE {' AND '.join(filters)} "
            "ORDER BY created_at DESC, chunk_index ASC LIMIT ?",
            [*params, limit],
        ).fetchall()
        results: List[RetrievalResult] = []
        for row in rows:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            metadata.setdefault("chunk_id", row["id"])
            metadata.setdefault("url", row["url"])
            results.append(
                RetrievalResult(
                    content=row["content"],
                    score=1.0,
                    source=row["source"],
                    metadata=metadata,
                )
            )

        get_event_bus().publish(
            EventType.MEMORY_RETRIEVE,
            {
                "backend": self.backend_id,
                "query": url_value,
                "num_results": len(results),
                "filters": {
                    "url": url_value,
                    "source": source_value,
                    "doc_type": doc_type_value,
                },
            },
        )
        return results

    def chunk_hashes(self, doc_id: str, source_id: str) -> Dict[int, str]:
        """Return ``{chunk_index: content_hash}`` for one document's body chunks.

        Scoped by *source_id* as well as *doc_id* because attachment chunks
        share the parent ``doc_id`` under a synthetic ``source_id`` and
        restart ``chunk_index`` at 0. Empty when nothing is stored. The
        ingestion pipeline compares this against a freshly chunked document
        to decide whether a re-sync needs to rewrite it.
        """
        rows = self._conn.execute(
            "SELECT chunk_index, content_hash FROM knowledge_chunks "
            "WHERE doc_id = ? AND source_id = ?",
            (doc_id, source_id),
        ).fetchall()
        return {row[0]: row[1] or "" for row in rows}

    def embedding_versions(self, doc_id: str, source_id: str) -> set[str]:
        """Return the distinct ``embedding_model_version`` values on a document's body.

        Rows that were ingested without an embedder carry ``""``. The
        ingestion pipeline uses this to notice that an embedder has since
        become available (or changed model) and re-embed an otherwise
        unchanged document.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT embedding_model_version FROM knowledge_chunks "
            "WHERE doc_id = ? AND source_id = ?",
            (doc_id, source_id),
        ).fetchall()
        return {row[0] or "" for row in rows}

    def delete(self, doc_id: str) -> bool:
        """Delete all chunks with the given *doc_id*. Returns True if any existed."""
        cur = self._conn.execute(
            "DELETE FROM knowledge_chunks WHERE doc_id = ?", (doc_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks with the given *source*. Returns the count removed.

        Must be called when a connector is disconnected so a later
        reconnect -- to the same or a different underlying data source --
        starts clean. Without this, orphaned chunks from the old source
        remain indexed forever, and if the new source produces a doc_id
        that happens to collide with one of them (e.g. two vaults each
        containing a file at the same relative path), the ingestion
        pipeline's duplicate-doc_id dedup silently discards the new
        content in favor of the stale orphan.
        """
        return self.delete_by_sources((source,))

    def delete_by_sources(self, sources: Iterable[str]) -> int:
        """Atomically delete chunks belonging to any of *sources*."""
        unique_sources = tuple(dict.fromkeys(sources))
        if not unique_sources:
            return 0
        placeholders = ", ".join("?" for _ in unique_sources)
        try:
            cur = self._conn.execute(
                f"DELETE FROM knowledge_chunks WHERE source IN ({placeholders})",
                unique_sources,
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return cur.rowcount

    def clear(self) -> None:
        """Remove all stored chunks."""
        self._conn.executescript(
            "DELETE FROM knowledge_chunks; DELETE FROM knowledge_fts;"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Extra helpers
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of stored chunks."""
        row = self._conn.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()
        return row[0] if row else 0

    def distinct_sources(self) -> List[str]:
        """Return the sorted list of distinct ``source`` values currently indexed.

        Used by the research agent to populate the system prompt with the
        sources the user actually has connected — so the model doesn't
        mention "Notion" or "Apple Notes" when nothing from those sources
        is in the corpus.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT source FROM knowledge_chunks "
            "WHERE source IS NOT NULL AND source != '' "
            "ORDER BY source"
        ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass


__all__ = ["KnowledgeStore"]
