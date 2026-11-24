"""Persistent, searchable records of substantial completed conversations."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock, RLock
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from openjarvis.core.paths import get_config_dir
from openjarvis.memory.procedures import (
    ProcedureCandidate,
    ProcedureOutcome,
    ProcedureProposal,
    RelatedProcedureOutcome,
    RelatedProcedureProposal,
    is_explicit_failure,
    is_explicit_success,
    procedure_title_similarity,
)

if TYPE_CHECKING:
    from openjarvis.connectors.embeddings import OllamaEmbedder
    from openjarvis.tools.storage._stubs import RetrievalResult

logger = logging.getLogger(__name__)

_EPISODE_CUES = re.compile(
    r"\b(?:decidi|decidimos|decisão|resolvi|resolvemos|corrigi|corrigimos|"
    r"funcionou|falhou|aprendi|concluí|conclusão|problema|solução|meta|"
    r"projeto|bug|fix|fixed|resolved|decision|learned|outcome|goal|project|"
    r"issue|error|solution|worked|failed)\b",
    re.IGNORECASE,
)
_MAX_EXCHANGE_CHARS = 12_000
_MAX_EMBED_CHARS = 6_000
_DERIVATION_KINDS = frozenset({"facts", "graph"})
_DERIVATION_LEASE_SECONDS = 600
_DERIVATION_NOT_DELETING = (
    "AND NOT EXISTS (SELECT 1 FROM episode_deletion_journal "
    "WHERE scope_id IN ('*', job.episode_id)) "
    "AND NOT EXISTS (SELECT 1 FROM episode_clear_journal) "
)
_EMBEDDER_CACHE_LOCK = Lock()
_EMBEDDER_CACHE: dict[tuple[str, str], tuple[float, OllamaEmbedder | None]] = {}


def _is_loopback_host(host: str) -> bool:
    parsed = urlsplit(host if "://" in host else f"http://{host}")
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return False
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        return False
    if port == 0:
        return False
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _local_embedder(model: str) -> OllamaEmbedder | None:
    """Return a cached Ollama embedder only for a loopback service."""
    from openjarvis.connectors.embeddings import (
        DEFAULT_OLLAMA_HOST,
        OllamaEmbedder,
    )

    host = os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
    if not _is_loopback_host(host):
        return None
    parsed_host = urlsplit(host if "://" in host else f"http://{host}")
    hostname = parsed_host.hostname or "localhost"
    port = parsed_host.port or 11434
    authority_host = f"[{hostname}]" if ":" in hostname else hostname
    normalized_host = f"{parsed_host.scheme}://{authority_host}:{port}"
    key = (normalized_host.casefold(), model)
    now = time.monotonic()
    with _EMBEDDER_CACHE_LOCK:
        cached = _EMBEDDER_CACHE.get(key)
        if cached is not None and now < cached[0]:
            return cached[1]
        embedder = OllamaEmbedder(
            model=model,
            host=normalized_host,
            timeout=4.0,
            allow_redirects=False,
            trust_env=False,
        )
        available = embedder if embedder.is_available() else None
        ttl = 300.0 if available is not None else 30.0
        _EMBEDDER_CACHE[key] = (now + ttl, available)
        return available


@dataclass(frozen=True, slots=True)
class Episode:
    """One completed exchange retained as a historical, unverified record."""

    episode_id: str
    user_text: str
    assistant_text: str
    source: str
    created_at: float
    session_id: str = ""


@dataclass(frozen=True, slots=True)
class EpisodeDerivationClaim:
    """Exclusive, time-limited right to publish one episode derivation."""

    episode: Episode
    kind: str
    token: str
    attempts: int


@dataclass(frozen=True, slots=True)
class ArchiveClearIntent:
    """Snapshot of deletion targets across the local memory stores."""

    token: str
    episode_ids: tuple[str, ...]
    graph_episode_ids: tuple[str, ...]
    conversation_sources: tuple[str, ...]


class EpisodeStore:
    """Local SQLite/FTS5 store for substantial conversation episodes.

    Episodes preserve the original exchange rather than turning the assistant's
    interpretation into a fact. They are searchable, bounded by age and count,
    and intended to be supplied to the model only as untrusted history.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_episodes: int = 5_000,
        retention_days: int = 365,
        semantic_enabled: bool = False,
        embedding_model: str = "nomic-embed-text",
        embedder: OllamaEmbedder | None = None,
    ) -> None:
        self._path = (
            Path(path).expanduser() if path else get_config_dir() / "episodes.db"
        )
        self._max_episodes = max(1, min(int(max_episodes), 100_000))
        self._retention_days = max(0, min(int(retention_days), 3_650))
        self._semantic_enabled = bool(semantic_enabled)
        self._embedding_model = embedding_model.strip() or "nomic-embed-text"
        self._embedder = embedder
        self._lock = RLock()
        if str(self._path) != ":memory:":
            from openjarvis.security.file_utils import secure_create

            secure_create(self._path)
        self._conn = sqlite3.connect(
            str(self._path), timeout=5.0, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    row_id INTEGER PRIMARY KEY,
                    episode_id TEXT NOT NULL UNIQUE,
                    user_text TEXT NOT NULL,
                    assistant_text TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    session_id TEXT NOT NULL DEFAULT ''
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
                USING fts5(
                    user_text,
                    assistant_text,
                    source,
                    content='episodes',
                    content_rowid='row_id',
                    tokenize='porter unicode61'
                );
                CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
                    INSERT INTO episodes_fts(rowid, user_text, assistant_text, source)
                    VALUES (new.row_id, new.user_text, new.assistant_text, new.source);
                END;
                CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
                    INSERT INTO episodes_fts(
                        episodes_fts, rowid, user_text, assistant_text, source
                    ) VALUES (
                        'delete', old.row_id, old.user_text, old.assistant_text,
                        old.source
                    );
                END;
                CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
                    INSERT INTO episodes_fts(
                        episodes_fts, rowid, user_text, assistant_text, source
                    ) VALUES (
                        'delete', old.row_id, old.user_text, old.assistant_text,
                        old.source
                    );
                    INSERT INTO episodes_fts(rowid, user_text, assistant_text, source)
                    VALUES (new.row_id, new.user_text, new.assistant_text, new.source);
                END;
                CREATE INDEX IF NOT EXISTS episodes_created_idx
                ON episodes(created_at DESC);
                CREATE TABLE IF NOT EXISTS episode_deletion_journal (
                    scope_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS episode_clear_journal (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    snapshot TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS episode_derivation_jobs (
                    episode_id TEXT NOT NULL
                        REFERENCES episodes(episode_id) ON DELETE CASCADE,
                    kind TEXT NOT NULL CHECK (kind IN ('facts', 'graph')),
                    available_at REAL NOT NULL,
                    token TEXT NOT NULL DEFAULT '',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (episode_id, kind)
                );
                CREATE INDEX IF NOT EXISTS episode_derivation_ready_idx
                ON episode_derivation_jobs(kind, available_at);
                CREATE TABLE IF NOT EXISTS episode_embeddings (
                    episode_id TEXT PRIMARY KEY
                        REFERENCES episodes(episode_id) ON DELETE CASCADE,
                    model_version TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS episode_embeddings_model_idx
                ON episode_embeddings(model_version);
                CREATE TABLE IF NOT EXISTS procedure_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    evidence_episode_id TEXT NOT NULL UNIQUE
                        REFERENCES episodes(episode_id) ON DELETE CASCADE,
                    session_id TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    draft_markdown TEXT NOT NULL,
                    success_signal TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'proposed'
                        CHECK(status IN ('proposed', 'approved', 'rejected')),
                    skill_path TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS procedure_proposals_status_idx
                ON procedure_proposals(status, created_at DESC);
                CREATE TABLE IF NOT EXISTS procedure_outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    evidence_episode_id TEXT NOT NULL UNIQUE
                        REFERENCES episodes(episode_id) ON DELETE CASCADE,
                    session_id TEXT NOT NULL,
                    task_title TEXT NOT NULL,
                    outcome TEXT NOT NULL CHECK(outcome='failure'),
                    signal TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS procedure_outcomes_session_idx
                ON procedure_outcomes(session_id, created_at DESC);
                """
            )
            # Existing episode databases predate session grouping.
            episode_columns = {
                str(row["name"])
                for row in self._conn.execute("PRAGMA table_info(episodes)")
            }
            if "session_id" not in episode_columns:
                self._conn.execute(
                    "ALTER TABLE episodes ADD COLUMN session_id "
                    "TEXT NOT NULL DEFAULT ''"
                )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS episodes_session_idx "
                "ON episodes(session_id, created_at DESC)"
            )
            self._conn.commit()

    @staticmethod
    def is_worthy(user_text: str, assistant_text: str) -> bool:
        """Keep meaningful tasks and outcomes; skip greetings and small talk."""
        user = (user_text or "").strip()
        assistant = (assistant_text or "").strip()
        if not user or not assistant:
            return False
        if (
            _EPISODE_CUES.search(user)
            or is_explicit_success(user)
            or is_explicit_failure(user)
        ):
            return True
        if (
            len(user) >= 60
            and len(assistant) >= 120
            and _EPISODE_CUES.search(assistant)
        ):
            return True
        return len(user) >= 120 and len(assistant) >= 120

    @staticmethod
    def _episode(row: sqlite3.Row) -> Episode:
        return Episode(
            episode_id=str(row["episode_id"]),
            user_text=str(row["user_text"]),
            assistant_text=str(row["assistant_text"]),
            source=str(row["source"]),
            created_at=float(row["created_at"]),
            session_id=str(row["session_id"] or ""),
        )

    @staticmethod
    def episode_id_for(
        *, source: str = "", session_id: str = "", idempotency_key: str = ""
    ) -> str:
        """Return the stable ID used to store an exchange, or empty without a key."""
        safe_source = re.sub(r"[^a-zA-Z0-9_.-]", "_", (source or "")[:80])
        safe_session_id = (session_id or "").strip()[:128]
        safe_key = (idempotency_key or "").strip()[:256]
        if not safe_key:
            return ""
        identity = f"{safe_source}\0{safe_session_id}\0{safe_key}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]

    def record(
        self,
        user_text: str,
        assistant_text: str,
        *,
        source: str = "",
        session_id: str = "",
        idempotency_key: str = "",
        derivations: tuple[str, ...] = (),
    ) -> bool:
        """Persist a substantial exchange and queue opt-in work on first insertion."""
        derivations = self._derivation_kinds(derivations)
        user = (user_text or "").strip()
        assistant = (assistant_text or "").strip()
        if not self.is_worthy(user, assistant):
            return False
        user = user[:_MAX_EXCHANGE_CHARS]
        assistant = assistant[:_MAX_EXCHANGE_CHARS]
        safe_source = re.sub(r"[^a-zA-Z0-9_.-]", "_", (source or "")[:80])
        safe_session_id = (session_id or "").strip()[:128]
        safe_key = (idempotency_key or "").strip()[:256]
        episode_id = (
            self.episode_id_for(
                source=source, session_id=session_id, idempotency_key=safe_key
            )
            or uuid.uuid4().hex
        )
        if safe_key:
            with self._lock:
                existing = self._conn.execute(
                    "SELECT user_text, assistant_text, source, session_id "
                    "FROM episodes WHERE episode_id=?",
                    (episode_id,),
                ).fetchone()
            if existing is not None:
                return (
                    str(existing["user_text"]) == user
                    and str(existing["assistant_text"]) == assistant
                    and str(existing["source"]) == safe_source
                    and str(existing["session_id"]) == safe_session_id
                )

        now = time.time()
        embedder = self._get_embedder()
        embedding = None
        model_version = ""
        if embedder is not None:
            embedding = self._embed_exchange(embedder, user, assistant)
            if embedding:
                model_version = embedder.model_version
        with self._lock, self._conn:
            insert = (
                "INSERT OR IGNORE INTO episodes "
                if safe_key
                else "INSERT INTO episodes "
            )
            cursor = self._conn.execute(
                insert + "(episode_id, user_text, assistant_text, source, created_at, "
                "session_id) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    episode_id,
                    user,
                    assistant,
                    safe_source,
                    now,
                    safe_session_id,
                ),
            )
            if safe_key and cursor.rowcount == 0:
                existing = self._conn.execute(
                    "SELECT user_text, assistant_text, source, session_id "
                    "FROM episodes WHERE episode_id=?",
                    (episode_id,),
                ).fetchone()
                return bool(
                    existing is not None
                    and str(existing["user_text"]) == user
                    and str(existing["assistant_text"]) == assistant
                    and str(existing["source"]) == safe_source
                    and str(existing["session_id"]) == safe_session_id
                )
            if embedding and model_version:
                self._conn.execute(
                    "INSERT INTO episode_embeddings "
                    "(episode_id, model_version, vector, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (episode_id, model_version, embedding, now),
                )
            if derivations:
                self._conn.executemany(
                    "INSERT INTO episode_derivation_jobs "
                    "(episode_id, kind, available_at) VALUES (?, ?, ?)",
                    ((episode_id, kind, now) for kind in derivations),
                )
            self._prune_locked(now)
        return True

    @staticmethod
    def _derivation_kinds(kinds: tuple[str, ...]) -> tuple[str, ...]:
        if set(kinds) - _DERIVATION_KINDS:
            raise ValueError("Episode derivation kinds must be facts or graph.")
        return tuple(dict.fromkeys(kinds))

    def claim_derivation(self, kinds: tuple[str, ...]) -> EpisodeDerivationClaim | None:
        """Lease one ready job for an allowed kind across processes."""
        kinds = self._derivation_kinds(kinds)
        if not kinds:
            return None
        placeholders = ", ".join("?" for _ in kinds)
        select = (
            "SELECT episode.*, job.kind, job.attempts "
            "FROM episode_derivation_jobs AS job "
            "JOIN episodes AS episode ON episode.episode_id = job.episode_id "
            f"WHERE job.kind IN ({placeholders}) AND job.available_at <= ? "
            + _DERIVATION_NOT_DELETING
            + "ORDER BY job.available_at, episode.created_at, job.episode_id, "
            "job.kind LIMIT 1"
        )
        with self._lock:
            # Idle workers only read. Recheck eligibility after taking the writer lock.
            if self._conn.execute(select, (*kinds, time.time())).fetchone() is None:
                return None
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                now = time.time()
                self._prune_locked(now)
                row = self._conn.execute(select, (*kinds, now)).fetchone()
                if row is None:
                    return None
                token = uuid.uuid4().hex
                attempts = row["attempts"] + 1
                self._conn.execute(
                    "UPDATE episode_derivation_jobs "
                    "SET available_at = ?, token = ?, attempts = ? "
                    "WHERE episode_id = ? AND kind = ?",
                    (
                        now + _DERIVATION_LEASE_SECONDS,
                        token,
                        attempts,
                        row["episode_id"],
                        row["kind"],
                    ),
                )
                return EpisodeDerivationClaim(
                    episode=self._episode(row),
                    kind=row["kind"],
                    token=token,
                    attempts=attempts,
                )

    def complete_derivation(
        self, claim: EpisodeDerivationClaim, apply: Callable[[], None]
    ) -> bool:
        """Publish a live claim and retire it before any deletion can begin.

        The callback must only perform brief writes to the derived stores.
        Run inference before this call and do not reenter the episode store.
        """
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            self._prune_locked(now)
            job = self._conn.execute(
                "SELECT 1 FROM episode_derivation_jobs AS job "
                "WHERE job.episode_id = ? AND job.kind = ? AND job.token = ? "
                "AND job.token <> '' AND job.available_at > ? "
                + _DERIVATION_NOT_DELETING,
                (claim.episode.episode_id, claim.kind, claim.token, now),
            ).fetchone()
            if job is None:
                return False
            apply()
            self._conn.execute(
                "DELETE FROM episode_derivation_jobs WHERE episode_id = ? AND kind = ?",
                (claim.episode.episode_id, claim.kind),
            )
            return True

    def fail_derivation(self, claim: EpisodeDerivationClaim) -> bool:
        """Reschedule a live claim with exponential backoff, capped at one hour."""
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            cursor = self._conn.execute(
                "UPDATE episode_derivation_jobs SET available_at = ? + "
                "MIN(3600, 15 * (1 << MIN(attempts - 1, 8))), token = '' "
                "WHERE episode_id = ? AND kind = ? AND token = ? "
                "AND token <> '' AND available_at > ?",
                (now, claim.episode.episode_id, claim.kind, claim.token, now),
            )
            return cursor.rowcount == 1

    def list_for_session(self, session_id: str, *, limit: int = 20) -> list[Episode]:
        """Return recent episodes from one explicitly identified chat session."""
        session_id = (session_id or "").strip()[:128]
        if not session_id:
            return []
        limit = max(1, min(int(limit), 100))
        with self._lock:
            self._prune_locked(time.time())
            self._conn.commit()
            rows = self._conn.execute(
                "SELECT * FROM episodes WHERE session_id=? "
                "ORDER BY created_at DESC, row_id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [self._episode(row) for row in rows]

    @staticmethod
    def _procedure_proposal(row: sqlite3.Row) -> ProcedureProposal:
        return ProcedureProposal(
            proposal_id=str(row["proposal_id"]),
            evidence_episode_id=str(row["evidence_episode_id"]),
            session_id=str(row["session_id"]),
            skill_name=str(row["skill_name"]),
            title=str(row["title"]),
            description=str(row["description"]),
            draft_markdown=str(row["draft_markdown"]),
            success_signal=str(row["success_signal"]),
            status=str(row["status"]),
            skill_path=str(row["skill_path"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def create_procedure_proposal(
        self,
        candidate: ProcedureCandidate,
        *,
        evidence_episode_id: str,
        session_id: str,
    ) -> ProcedureProposal | None:
        """Persist one review-only proposal, deduplicated by source episode."""
        episode_id = (evidence_episode_id or "").strip()
        session = (session_id or "").strip()[:128]
        if not episode_id or not session:
            return None
        now = time.time()
        proposal_id = uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO procedure_proposals "
                "(proposal_id, evidence_episode_id, session_id, skill_name, title, "
                "description, draft_markdown, success_signal, status, created_at, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, ?)",
                (
                    proposal_id,
                    episode_id,
                    session,
                    candidate.skill_name,
                    candidate.title,
                    candidate.description,
                    candidate.draft_markdown,
                    candidate.success_signal,
                    now,
                    now,
                ),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM procedure_proposals WHERE evidence_episode_id=?",
                (episode_id,),
            ).fetchone()
        return self._procedure_proposal(row) if row is not None else None

    def get_procedure_proposal(self, proposal_id: str) -> ProcedureProposal | None:
        """Fetch one proposal by its complete ID or an unambiguous prefix."""
        key = (proposal_id or "").strip().lower()
        if not key or len(key) < 8 or not re.fullmatch(r"[a-f0-9]+", key):
            return None
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM procedure_proposals WHERE proposal_id LIKE ? LIMIT 2",
                (f"{key}%",),
            ).fetchall()
        if len(rows) != 1:
            return None
        return self._procedure_proposal(rows[0])

    def similar_procedure_proposals(
        self,
        proposal_id: str,
        *,
        limit: int = 5,
        minimum_similarity: float = 0.6,
    ) -> list[RelatedProcedureProposal]:
        """Find lexical matches from other sessions without merging proposals."""
        proposal = self.get_procedure_proposal(proposal_id)
        if proposal is None:
            return []
        limit = max(1, min(int(limit), 20))
        threshold = float(minimum_similarity)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_similarity must be between 0 and 1")
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM procedure_proposals WHERE session_id<>? "
                "AND evidence_episode_id<>? ORDER BY created_at DESC LIMIT 5000",
                (proposal.session_id, proposal.evidence_episode_id),
            ).fetchall()
        matches: list[RelatedProcedureProposal] = []
        for row in rows:
            related = self._procedure_proposal(row)
            score = procedure_title_similarity(proposal.title, related.title)
            if score >= threshold:
                matches.append(RelatedProcedureProposal(related, score))
        matches.sort(
            key=lambda item: (item.similarity, item.proposal.created_at),
            reverse=True,
        )
        return matches[:limit]

    @staticmethod
    def _procedure_outcome(row: sqlite3.Row) -> ProcedureOutcome:
        return ProcedureOutcome(
            outcome_id=str(row["outcome_id"]),
            evidence_episode_id=str(row["evidence_episode_id"]),
            session_id=str(row["session_id"]),
            task_title=str(row["task_title"]),
            outcome=str(row["outcome"]),
            signal=str(row["signal"]),
            created_at=float(row["created_at"]),
        )

    def record_procedure_outcome(
        self,
        *,
        evidence_episode_id: str,
        session_id: str,
        task_title: str,
        outcome: str,
        signal: str,
    ) -> ProcedureOutcome | None:
        """Store explicit user feedback against an episode in the same session."""
        episode_id = (evidence_episode_id or "").strip()
        session = (session_id or "").strip()[:128]
        title = re.sub(r"\s+", " ", (task_title or "").strip())[:160]
        feedback = (signal or "").strip()[:1_000]
        if outcome != "failure":
            raise ValueError("only explicit failure outcomes are stored separately")
        if not episode_id or not session or not title or not feedback:
            return None
        with self._lock:
            episode = self._conn.execute(
                "SELECT session_id FROM episodes WHERE episode_id=?",
                (episode_id,),
            ).fetchone()
            if episode is None or str(episode["session_id"] or "") != session:
                return None
            now = time.time()
            self._conn.execute(
                "INSERT OR IGNORE INTO procedure_outcomes "
                "(outcome_id, evidence_episode_id, session_id, task_title, "
                "outcome, signal, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, episode_id, session, title, outcome, feedback, now),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM procedure_outcomes WHERE evidence_episode_id=?",
                (episode_id,),
            ).fetchone()
        return self._procedure_outcome(row) if row is not None else None

    def similar_procedure_outcomes(
        self,
        task_title: str,
        *,
        session_id: str | None = None,
        limit: int = 5,
        minimum_similarity: float = 0.6,
    ) -> list[RelatedProcedureOutcome]:
        """Find explicit results for similar tasks.

        When *session_id* is supplied, outcomes from that conversation are
        excluded. Omitting it searches all retained outcomes, which is useful
        when building context before a conversation session is known.
        """
        title = re.sub(r"\s+", " ", (task_title or "").strip())[:160]
        session = (session_id or "").strip()[:128]
        if not title:
            return []
        limit = max(1, min(int(limit), 20))
        threshold = float(minimum_similarity)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_similarity must be between 0 and 1")
        with self._lock:
            if session:
                rows = self._conn.execute(
                    "SELECT * FROM procedure_outcomes WHERE session_id<>? "
                    "ORDER BY created_at DESC LIMIT 5000",
                    (session,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM procedure_outcomes "
                    "ORDER BY created_at DESC LIMIT 5000"
                ).fetchall()
        matches: list[RelatedProcedureOutcome] = []
        for row in rows:
            outcome = self._procedure_outcome(row)
            score = procedure_title_similarity(title, outcome.task_title)
            if score >= threshold:
                matches.append(RelatedProcedureOutcome(outcome, score))
        matches.sort(
            key=lambda item: (item.similarity, item.outcome.created_at),
            reverse=True,
        )
        return matches[:limit]

    def list_procedure_outcomes(self, *, limit: int = 50) -> list[ProcedureOutcome]:
        """List bounded explicit failure reports for user review."""
        limit = max(1, min(int(limit), 500))
        with self._lock:
            self._prune_locked(time.time())
            self._conn.commit()
            rows = self._conn.execute(
                "SELECT * FROM procedure_outcomes ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._procedure_outcome(row) for row in rows]

    def list_procedure_proposals(
        self, *, status: str = "proposed", limit: int = 100
    ) -> list[ProcedureProposal]:
        """List review proposals by status; ``all`` returns every state."""
        if status not in {"proposed", "approved", "rejected", "all"}:
            raise ValueError("status must be proposed, approved, rejected, or all")
        limit = max(1, min(int(limit), 500))
        with self._lock:
            if status == "all":
                rows = self._conn.execute(
                    "SELECT * FROM procedure_proposals "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM procedure_proposals WHERE status=? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
        return [self._procedure_proposal(row) for row in rows]

    def set_procedure_proposal_status(
        self,
        proposal_id: str,
        *,
        status: str,
        skill_path: str = "",
    ) -> bool:
        """Transition a pending proposal to an explicit user-reviewed state."""
        if status not in {"approved", "rejected"}:
            raise ValueError("proposal status must be approved or rejected")
        proposal = self.get_procedure_proposal(proposal_id)
        if proposal is None or proposal.status != "proposed":
            return False
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE procedure_proposals SET status=?, skill_path=?, updated_at=? "
                "WHERE proposal_id=? AND status='proposed'",
                (status, (skill_path or "")[:2_000], time.time(), proposal.proposal_id),
            )
            self._conn.commit()
        return cursor.rowcount == 1

    def _get_embedder(self) -> OllamaEmbedder | None:
        if not self._semantic_enabled:
            return None
        if self._embedder is None:
            self._embedder = _local_embedder(self._embedding_model)
        return self._embedder

    @staticmethod
    def _embed_exchange(
        embedder: OllamaEmbedder, user_text: str, assistant_text: str
    ) -> bytes | None:
        text = f"User: {user_text}\nAssistant: {assistant_text}"[:_MAX_EMBED_CHARS]
        try:
            return embedder.embed(text)
        except Exception:
            logger.debug("Local episode embedding failed", exc_info=True)
            return None

    def reindex_missing_embeddings(self, *, limit: int = 100) -> tuple[int, int]:
        """Embed old episodes on explicit request; returns (indexed, considered)."""
        embedder = self._get_embedder()
        if embedder is None:
            raise RuntimeError(
                "Local episode embeddings are unavailable. Start Ollama on this "
                "computer and install the configured embedding model."
            )
        limit = max(1, min(int(limit), 1_000))
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.episode_id, e.user_text, e.assistant_text "
                "FROM episodes e LEFT JOIN episode_embeddings v "
                "ON v.episode_id=e.episode_id "
                "WHERE v.episode_id IS NULL OR v.model_version != ? "
                "ORDER BY e.created_at ASC LIMIT ?",
                (embedder.model_version, limit),
            ).fetchall()
        indexed = 0
        for row in rows:
            vector = self._embed_exchange(
                embedder,
                str(row["user_text"]),
                str(row["assistant_text"]),
            )
            if not vector:
                continue
            with self._lock:
                self._conn.execute(
                    "INSERT INTO episode_embeddings "
                    "(episode_id, model_version, vector, updated_at) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(episode_id) DO UPDATE SET "
                    "model_version=excluded.model_version, vector=excluded.vector, "
                    "updated_at=excluded.updated_at",
                    (
                        row["episode_id"],
                        embedder.model_version,
                        vector,
                        time.time(),
                    ),
                )
                self._conn.commit()
            indexed += 1
        return indexed, len(rows)

    def _prune_locked(self, now: float) -> None:
        if self._retention_days:
            cutoff = now - self._retention_days * 86_400
            self._conn.execute("DELETE FROM episodes WHERE created_at < ?", (cutoff,))
        self._conn.execute(
            "DELETE FROM episodes WHERE row_id IN "
            "(SELECT row_id FROM episodes ORDER BY created_at DESC "
            "LIMIT -1 OFFSET ?)",
            (self._max_episodes,),
        )

    def list(self, *, limit: int = 50) -> list[Episode]:
        limit = max(1, min(int(limit), 500))
        with self._lock:
            self._prune_locked(time.time())
            self._conn.commit()
            rows = self._conn.execute(
                "SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._episode(row) for row in rows]

    def get_by_id(self, episode_id: str) -> Episode | None:
        """Return one retained episode by its full stable ID."""
        episode_id = (episode_id or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{32}", episode_id):
            return None
        with self._lock:
            self._prune_locked(time.time())
            row = self._conn.execute(
                "SELECT * FROM episodes WHERE episode_id=?", (episode_id,)
            ).fetchone()
            self._conn.commit()
        return self._episode(row) if row is not None else None

    @staticmethod
    def read_retained(
        episode_id: str,
        *,
        path: str | Path | None = None,
        max_episodes: int = 5_000,
        retention_days: int = 365,
    ) -> Episode | None:
        """Read one visible episode without migrating or pruning the archive."""
        episode_id = episode_id.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{32}", episode_id):
            return None
        db_path = Path(path).expanduser() if path else get_config_dir() / "episodes.db"
        if not db_path.is_file():
            return None

        limit = max(1, min(int(max_episodes), 100_000))
        days = max(0, min(int(retention_days), 3_650))
        cutoff = time.time() - days * 86_400 if days else None
        uri = db_path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            row = conn.execute(
                "SELECT episode_id, user_text, assistant_text, source, created_at "
                "FROM episodes WHERE episode_id = ? "
                "AND (? IS NULL OR created_at >= ?) "
                "AND row_id IN (SELECT row_id FROM episodes "
                "ORDER BY created_at DESC LIMIT ?)",
                (episode_id, cutoff, cutoff, limit),
            ).fetchone()
        return Episode(*row) if row is not None else None

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        source: str | None = None,
    ) -> list[RetrievalResult]:
        """Fuse local FTS5 and vector results, optionally limiting by source."""
        terms = list(dict.fromkeys(re.findall(r"[\w]+", query or "", flags=re.UNICODE)))
        if not terms:
            return []

        fts_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms[:32])
        top_k = max(1, min(int(top_k), 20))
        candidate_k = min(max(top_k * 8, 40), 200)
        embedder = self._get_embedder()
        query_vector = None
        if embedder is not None:
            try:
                query_blob = embedder.embed(query)
                if query_blob:
                    from openjarvis.connectors.embeddings import decode_embedding

                    query_vector = decode_embedding(query_blob)
            except Exception:
                logger.debug("Local episode query embedding failed", exc_info=True)
        with self._lock:
            self._prune_locked(time.time())
            self._conn.commit()
            lexical_where = "WHERE episodes_fts MATCH ?"
            lexical_params: list[Any] = [fts_query]
            if source is not None:
                lexical_where += " AND e.source = ?"
                lexical_params.append(source)
            lexical_params.append(candidate_k)
            lexical_rows = self._conn.execute(
                "SELECT e.*, bm25(episodes_fts) AS rank "
                "FROM episodes_fts JOIN episodes e ON e.row_id=episodes_fts.rowid "
                + lexical_where
                + " ORDER BY rank, e.created_at DESC LIMIT ?",
                lexical_params,
            ).fetchall()
            vector_where = "WHERE v.model_version = ?"
            vector_params: list[Any] = (
                [embedder.model_version] if embedder is not None else []
            )
            if source is not None:
                vector_where += " AND e.source = ?"
                vector_params.append(source)
            vector_rows = (
                self._conn.execute(
                    "SELECT e.*, v.vector AS vector_blob "
                    "FROM episode_embeddings v JOIN episodes e "
                    "ON e.episode_id=v.episode_id " + vector_where,
                    vector_params,
                ).fetchall()
                if query_vector is not None and embedder is not None
                else []
            )
        row_by_id = {str(row["episode_id"]): row for row in lexical_rows}
        vector_ids = self._rank_vectors(query_vector, vector_rows, row_by_id)
        fused: dict[str, float] = {}
        for rank, row in enumerate(lexical_rows):
            episode_id = str(row["episode_id"])
            fused[episode_id] = fused.get(episode_id, 0.0) + 1.0 / (60 + rank + 1)
        for rank, episode_id in enumerate(vector_ids):
            fused[episode_id] = fused.get(episode_id, 0.0) + 1.0 / (60 + rank + 1)
        ordered_ids = sorted(
            fused,
            key=lambda episode_id: (
                -fused[episode_id],
                -float(row_by_id[episode_id]["created_at"]),
            ),
        )[:top_k]
        results: list[RetrievalResult] = []
        for rank, episode_id in enumerate(ordered_ids):
            episode = self._episode(row_by_id[episode_id])
            results.append(self._retrieval_result(episode, 1.0 / (rank + 1)))
        return results

    def retrieve_by_ids(
        self,
        episode_ids: Sequence[str],
        *,
        limit: int = 3,
    ) -> list[RetrievalResult]:
        """Retrieve a small ordered set of explicitly linked episodes."""
        limit = max(0, min(int(limit), 20))
        if limit == 0:
            return []
        selected_ids: list[str] = []
        seen: set[str] = set()
        for value in episode_ids:
            episode_id = str(value or "").strip().lower()
            if not re.fullmatch(r"[0-9a-f]{32}", episode_id):
                continue
            if episode_id in seen:
                continue
            seen.add(episode_id)
            selected_ids.append(episode_id)
            if len(selected_ids) >= limit:
                break

        if not selected_ids:
            return []
        marks = ",".join("?" for _ in selected_ids)
        with self._lock:
            self._prune_locked(time.time())
            rows = self._conn.execute(
                f"SELECT * FROM episodes WHERE episode_id IN ({marks})",
                selected_ids,
            ).fetchall()
            self._conn.commit()
        row_by_id = {str(row["episode_id"]): row for row in rows}
        return [
            self._retrieval_result(
                self._episode(row_by_id[episode_id]), 1.0, linked=True
            )
            for episode_id in selected_ids
            if episode_id in row_by_id
        ]

    @staticmethod
    def _retrieval_result(
        episode: Episode,
        score: float,
        *,
        linked: bool = False,
    ) -> RetrievalResult:
        from openjarvis.tools.storage._stubs import RetrievalResult

        date_text = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(episode.created_at))
        content = (
            f"Date: {date_text}\nUser: {episode.user_text}\n"
            f"A.C.E.: {episode.assistant_text}"
        )
        metadata: dict[str, Any] = {
            "trust": "auto",
            "memory_kind": "episode",
            "episode_id": episode.episode_id,
            "created_at": episode.created_at,
        }
        if linked:
            metadata["graph_linked"] = True
        return RetrievalResult(
            content=content[:3_000],
            score=score,
            source=episode.source,
            metadata=metadata,
        )

    @staticmethod
    def _rank_vectors(
        query_vector: Any,
        vector_rows: list[sqlite3.Row],
        row_by_id: dict[str, sqlite3.Row],
    ) -> list[str]:
        if query_vector is None or not vector_rows:
            return []
        try:
            import numpy as np

            from openjarvis.connectors.embeddings import decode_embedding

            query = np.asarray(query_vector, dtype=np.float32)
            query_norm = float(np.linalg.norm(query))
            if query_norm == 0:
                return []
            query /= query_norm
            vectors: list[Any] = []
            episode_ids: list[str] = []
            for row in vector_rows:
                vector = decode_embedding(row["vector_blob"])
                if vector is None or vector.size != query.size:
                    continue
                vectors.append(vector)
                episode_id = str(row["episode_id"])
                episode_ids.append(episode_id)
                row_by_id[episode_id] = row
            if not vectors:
                return []
            matrix = np.vstack(vectors).astype(np.float32, copy=False)
            norms = np.linalg.norm(matrix, axis=1)
            norms[norms == 0] = 1.0
            scores = (matrix / norms[:, None]) @ query
            order = np.argsort(-scores)[: min(200, len(scores))]
            return [episode_ids[int(index)] for index in order]
        except Exception:
            logger.debug("Could not rank local episode vectors", exc_info=True)
            return []

    @staticmethod
    def _deletion_scope_id(episode_id: str) -> str:
        normalized = episode_id.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{32}", normalized):
            raise ValueError("Episode ID must be a full 32-character ID.")
        return normalized

    def record_deletion_intent(self, episode_id: str) -> None:
        """Persist a cleanup intent before changing any related memory store."""
        scope_id = self._deletion_scope_id(episode_id)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO episode_deletion_journal "
                "(scope_id, created_at) VALUES (?, ?)",
                (scope_id, time.time()),
            )

    def pending_deletion_intents(self) -> list[str]:
        """Return the IDs of the individual pending cleanups."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT scope_id FROM episode_deletion_journal "
                "WHERE scope_id <> '*' ORDER BY scope_id"
            ).fetchall()
        return [self._deletion_scope_id(str(row["scope_id"])) for row in rows]

    def finish_deletion_intent(self, episode_id: str) -> None:
        """Remove an individual journal entry after its cleanup succeeds."""
        scope_id = self._deletion_scope_id(episode_id)
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM episode_deletion_journal WHERE scope_id = ?",
                (scope_id,),
            )

    def archive_episode_ids(self) -> tuple[str, ...]:
        """Read the current episode IDs without changing the archive."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT episode_id FROM episodes ORDER BY episode_id"
            ).fetchall()
        return tuple(self._deletion_scope_id(row["episode_id"]) for row in rows)

    def record_archive_clear(self, intent: ArchiveClearIntent) -> None:
        """Save the reviewed snapshot before removing any data."""
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO episode_clear_journal (singleton, snapshot) "
                "VALUES (1, ?)",
                (json.dumps(asdict(intent), ensure_ascii=False),),
            )
            if cursor.rowcount == 0:
                raise RuntimeError(
                    "An archive clear is already pending. "
                    "Run jarvis memory episodes recover."
                )
            self._conn.execute(
                "DELETE FROM episode_deletion_journal WHERE scope_id = '*'"
            )

    def pending_archive_clear(self) -> ArchiveClearIntent | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT snapshot FROM episode_clear_journal WHERE singleton = 1"
            ).fetchone()
        return self._decode_clear_intent(row["snapshot"]) if row is not None else None

    @classmethod
    def _decode_clear_intent(cls, snapshot: str) -> ArchiveClearIntent:
        data = json.loads(snapshot)
        if not isinstance(data, dict) or not isinstance(data.get("token"), str):
            raise ValueError("Invalid episode archive clear snapshot.")
        for key in ("episode_ids", "graph_episode_ids", "conversation_sources"):
            values = data.get(key)
            if not isinstance(values, list) or not all(
                isinstance(value, str) for value in values
            ):
                raise ValueError("Invalid episode archive clear snapshot.")
        return ArchiveClearIntent(
            token=cls._deletion_scope_id(data["token"]),
            episode_ids=tuple(cls._deletion_scope_id(v) for v in data["episode_ids"]),
            graph_episode_ids=tuple(
                cls._deletion_scope_id(v) for v in data["graph_episode_ids"]
            ),
            conversation_sources=tuple(data["conversation_sources"]),
        )

    def has_legacy_archive_clear(self) -> bool:
        """An older unscoped clear requires a new confirmation before replay."""
        with self._lock:
            return (
                self._conn.execute(
                    "SELECT 1 FROM episode_deletion_journal WHERE scope_id = '*'"
                ).fetchone()
                is not None
            )

    def finish_archive_clear(self, intent: ArchiveClearIntent) -> int:
        """Delete only the snapshot's episodes and atomically retire its intent."""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT snapshot FROM episode_clear_journal WHERE singleton = 1"
            ).fetchone()
            if row is None or self._decode_clear_intent(row["snapshot"]) != intent:
                return 0
            # Acquire the SQLite writer lock before deleting any episode, and
            # require the same snapshot in case another process finished first.
            cursor = self._conn.execute(
                "DELETE FROM episode_clear_journal "
                "WHERE singleton = 1 AND snapshot = ?",
                (row["snapshot"],),
            )
            if cursor.rowcount == 0:
                return 0
            return self._conn.executemany(
                "DELETE FROM episodes WHERE episode_id = ?",
                ((episode_id,) for episode_id in intent.episode_ids),
            ).rowcount

    def delete(self, episode_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM episodes WHERE episode_id = ?", (episode_id.strip(),)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> int:
        with self._lock:
            count = int(
                self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            )
            self._conn.execute("DELETE FROM episodes")
            self._conn.commit()
        return count

    def count(self) -> int:
        with self._lock:
            self._prune_locked(time.time())
            self._conn.commit()
            row = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
            return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


__all__ = ["Episode", "EpisodeStore"]
