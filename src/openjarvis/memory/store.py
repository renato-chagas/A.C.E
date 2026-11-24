"""Persistent stores for automatically extracted long-term memory facts.

A *fact* is a short, durable statement worth remembering about the user
(e.g. ``"Prefers concise answers"``).  Facts are produced by the memory
service's background extractor and persisted here so they survive across
sessions.  The store is intentionally small and self-contained: it dedupes,
caps the total number of facts, and is safe to call from multiple threads.
"""

from __future__ import annotations

import errno
import json
import sys
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, List

from openjarvis.core.paths import get_config_dir
from openjarvis.core.registry import FactStoreRegistry
from openjarvis.security.file_utils import (
    secure_create,
    secure_mkdir,
    secure_write_text,
)

if sys.platform != "win32":
    import fcntl


_WINDOWS_LOCK_RETRY_SECONDS = 0.01
_WINDOWS_LOCK_RETRY_ERRNOS = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
_WINDOWS_LOCK_RETRY_WINERRORS = {33}  # ERROR_LOCK_VIOLATION


def _lock_windows_blocking(
    file_obj: Any,
    *,
    locking_api: Any | None = None,
    retry_interval: float = _WINDOWS_LOCK_RETRY_SECONDS,
) -> None:
    """Acquire the first-byte Windows lock, retrying until it is available.

    ``msvcrt.LK_LOCK`` only retries a fixed number of times before raising.
    Use the non-blocking operation in our own retry loop so contention never
    turns an otherwise valid store operation into a spurious failure.
    """
    if locking_api is None:
        import msvcrt as locking_api

    while True:
        file_obj.seek(0)
        try:
            locking_api.locking(file_obj.fileno(), locking_api.LK_NBLCK, 1)
            return
        except OSError as exc:
            if (
                exc.errno not in _WINDOWS_LOCK_RETRY_ERRNOS
                and getattr(exc, "winerror", None) not in _WINDOWS_LOCK_RETRY_WINERRORS
            ):
                raise
            time.sleep(retry_interval)


def _unlock_windows(file_obj: Any, *, locking_api: Any | None = None) -> None:
    if locking_api is None:
        import msvcrt as locking_api

    file_obj.seek(0)
    locking_api.locking(file_obj.fileno(), locking_api.LK_UNLCK, 1)


def _default_fact_path() -> Path:
    """Return the env-aware default JSONL path for automatic memory facts."""
    return get_config_dir() / "memory_facts.jsonl"


# Provenance tiers, recorded on every fact.
TRUST_AUTO = "auto"  # auto-extracted and scanner-clean → recallable
TRUST_TRUSTED = "trusted"  # explicitly vouched for by the user
TRUST_UNTRUSTED = "untrusted"  # scanner flagged the text → quarantined

# "" is the legacy on-disk value, written before provenance existed. It stays
# recallable so upgrading does not silently erase a user's existing memory.
# Exported because the same tier vocabulary gates document recall in
# tools.storage.context — a second, drifting copy of this set would be a
# security bug waiting to happen.
RECALLABLE_TRUST_TIERS = frozenset({"", TRUST_AUTO, TRUST_TRUSTED})
_RECALLABLE_TIERS = RECALLABLE_TRUST_TIERS  # internal alias


@contextmanager
def _cross_process_lock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive, blocking OS-level lock on *lock_path*.

    ``threading.Lock`` only serializes within one process, so it cannot
    protect the load -> modify -> flush cycle when two separate
    ``LocalFactStore`` instances (e.g. in different processes) touch the
    same file (#755). This blocks until the lock is free rather than
    failing fast, since callers want to wait their turn, not error out.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+b")
    try:
        if sys.platform == "win32":
            # Windows can lock a byte range beyond EOF. Avoid writing an
            # initialization byte here: simultaneous processes may race while
            # extending the shared lock file before either owns the lock.
            _lock_windows_blocking(f)
        else:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if sys.platform == "win32":
                _unlock_windows(f)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


@dataclass(slots=True)
class Fact:
    """A single durable memory entry."""

    text: str
    source: str = ""
    created_at: float = 0.0
    # Provenance tier: one of the TRUST_* constants above ("" for legacy rows).
    trust: str = ""
    # Episode IDs that contributed this fact. Legacy or unlinked origins remain
    # explicitly incomplete so deletion never removes a potentially shared fact.
    source_ids: List[str] = field(default_factory=list)
    source_tracking_complete: bool = False
    superseded: bool = False

    @property
    def trusted_for_recall(self) -> bool:
        """Whether this fact may be placed in model-facing context.

        Superseded entries and unsafe or unknown trust tiers are withheld.
        """
        return (
            not self.superseded
            and (self.trust or "").strip().lower() in _RECALLABLE_TIERS
        )


class FactStore(ABC):
    """Abstract persistent store for extracted memory facts."""

    def remove_reviewed(self, index: int, *, expected_text: str | None = None) -> bool:
        """Remove a zero-based fact when supported by this backend.

        ``expected_text`` lets a store refuse to remove a different fact if its
        list changed after the user reviewed it. The default keeps existing
        third-party FactStore implementations source-compatible.
        """
        del index, expected_text
        return False

    @abstractmethod
    def add(self, text: str, source: str = "") -> bool:
        """Store *text* as a fact. Returns True if a new fact was stored."""

    def set_trust(self, index: int, trust: str) -> bool:
        """Set the provenance tier of the *index*-th fact (0-based) as returned
        by :meth:`list`. Returns True if a fact was updated."""
        raise NotImplementedError

    def add_many(self, texts: Iterable[str], source: str = "") -> int:
        """Store several facts, returning the count of newly stored ones."""
        added = 0
        for text in texts:
            if self.add(text, source=source):
                added += 1
        return added

    def add_with_trust(self, text: str, source: str = "", trust: str = "") -> bool:
        """Store a provenance-aware fact without breaking legacy backends.

        Third-party stores implementing the original ``add(text, source)``
        contract inherit this adapter. Recallable facts are stored normally;
        quarantined or unknown tiers are dropped because a backend that cannot
        persist provenance cannot safely retain them for model-facing recall.
        Provenance-aware stores should override this method.
        """
        if (trust or "").strip().lower() not in _RECALLABLE_TIERS:
            return False
        return self.add(text, source=source)

    def add_many_with_trust(
        self,
        texts: Iterable[str],
        source: str = "",
        trust: str = "",
    ) -> int:
        """Store several provenance-aware facts."""
        return sum(
            bool(self.add_with_trust(text, source=source, trust=trust))
            for text in texts
        )

    def add_many_with_provenance(
        self,
        texts: Iterable[str],
        source: str = "",
        trust: str = "",
        source_id: str | None = None,
    ) -> int:
        """Store facts with an optional source episode.

        Backends predating episode provenance remain compatible, but cannot
        guarantee source-aware deletion.
        """
        del source_id
        return self.add_many_with_trust(texts, source=source, trust=trust)

    def purge_episode_sources(
        self,
        episode_ids: Iterable[str] | None,
    ) -> dict[str, int]:
        """Remove facts whose only known origins are the selected episodes."""
        del episode_ids
        return {
            "removed_facts": 0,
            "removed_links": 0,
            "retained_facts": 0,
            "untracked_remaining": 0,
            "unsupported": 1,
        }

    def promote_reviewed(self, index: int, expected_text: str) -> bool:
        """Promote the reviewed fact at *index* when it still matches.

        The default preserves compatibility with third-party implementations;
        stores with concurrent writers should override this with an atomic
        identity check.
        """
        del expected_text
        return self.set_trust(index, TRUST_TRUSTED)

    def revise(self, expected_text: str, replacement: str) -> None:
        """Replace a specific fact while retaining the previous wording."""
        raise NotImplementedError("This fact backend does not support revisions.")

    @abstractmethod
    def list(self) -> List[Fact]:
        """Return all stored facts, oldest first."""

    @abstractmethod
    def clear(self) -> int:
        """Remove all stored facts, returning the number removed."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored facts."""


@FactStoreRegistry.register("local")
class LocalFactStore(FactStore):
    """Append-only JSONL fact store on the local filesystem.

    Facts are kept human-readable (one JSON object per line) so they can be
    inspected or edited by hand.  Writes are atomic (temp file + rename) and
    guarded by a lock, so concurrent ``add`` calls from the extraction worker
    and ``list``/``clear`` from the CLI never corrupt the file.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_facts: int = 1000,
    ) -> None:
        max_facts = int(max_facts)
        if max_facts <= 0:
            raise ValueError(f"max_facts must be a positive integer, got {max_facts}")
        self._path = (
            Path(path).expanduser() if path is not None else _default_fact_path()
        )
        self._max_facts = max_facts
        self._lock = threading.Lock()
        # Readers must take the same OS lock as writers. On Windows,
        # os.replace() can fail while another process has the JSONL file open.
        with self._lock, _cross_process_lock(self._lock_path()):
            secure_mkdir(self._path.parent)
            secure_create(self._lock_path())
            secure_create(self._path)
            self._facts: List[Fact] = self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> List[Fact]:
        if not self._path.exists():
            return []
        facts: List[Fact] = []
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError:
            return []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue  # skip malformed lines rather than crashing
            fact_text = str(obj.get("text", "")).strip()
            if not fact_text:
                continue
            raw_source_ids = obj.get("source_ids", [])
            source_ids: list[str] = []
            raw_ids_valid = isinstance(raw_source_ids, list)
            if raw_ids_valid:
                for raw_id in raw_source_ids:
                    source_id = str(raw_id or "").strip().lower()
                    if (
                        len(source_id) == 32
                        and all(char in "0123456789abcdef" for char in source_id)
                        and source_id not in source_ids
                    ):
                        source_ids.append(source_id)
                    else:
                        raw_ids_valid = False
            tracking_complete = (
                obj.get("source_tracking_complete") is True
                and raw_ids_valid
                and bool(source_ids)
                and len(raw_source_ids) <= 64
            )
            facts.append(
                Fact(
                    text=fact_text,
                    source=str(obj.get("source", "")),
                    created_at=float(obj.get("created_at", 0.0) or 0.0),
                    trust=str(obj.get("trust", "")),
                    source_ids=source_ids[:64],
                    source_tracking_complete=tracking_complete,
                    superseded=obj.get("superseded") is True,
                )
            )
        return facts

    def _flush(self) -> None:
        """Atomically rewrite the JSONL file from the in-memory list."""
        payload = "".join(
            json.dumps(asdict(f), ensure_ascii=False) + "\n" for f in self._facts
        )
        secure_write_text(self._path, payload)

    def _lock_path(self) -> Path:
        return self._path.with_suffix(self._path.suffix + ".lock")

    def _sync_from_disk_locked(self) -> None:
        """Refresh in-memory facts from disk while holding ``self._lock``."""
        self._facts = self._load()

    # -- FactStore API ------------------------------------------------------

    def add(self, text: str, source: str = "", trust: str = "") -> bool:
        return self._add_with_provenance(text, source=source, trust=trust)

    def add_with_trust(self, text: str, source: str = "", trust: str = "") -> bool:
        return self._add_with_provenance(text, source=source, trust=trust)

    def add_many_with_provenance(
        self,
        texts: Iterable[str],
        source: str = "",
        trust: str = "",
        source_id: str | None = None,
    ) -> int:
        return sum(
            self._add_with_provenance(
                text,
                source=source,
                trust=trust,
                source_id=source_id,
            )
            for text in texts
        )

    @staticmethod
    def _normalize_episode_id(source_id: str | None) -> str:
        value = str(source_id or "").strip().lower()
        if len(value) != 32 or any(char not in "0123456789abcdef" for char in value):
            return ""
        return value

    def _add_with_provenance(
        self,
        text: str,
        *,
        source: str = "",
        trust: str = "",
        source_id: str | None = None,
    ) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        trust = (trust or "").strip().lower()
        episode_id = self._normalize_episode_id(source_id)
        with self._lock, _cross_process_lock(self._lock_path()):
            self._sync_from_disk_locked()
            lowered = text.lower()
            duplicate = next(
                (fact for fact in self._facts if fact.text.lower() == lowered),
                None,
            )
            if duplicate is not None:
                changed = False
                if episode_id:
                    if episode_id not in duplicate.source_ids:
                        if len(duplicate.source_ids) < 64:
                            duplicate.source_ids.append(episode_id)
                        else:
                            duplicate.source_tracking_complete = False
                        changed = True
                elif duplicate.source_ids and duplicate.source_tracking_complete:
                    # A repeated fact without a retained episode is another
                    # origin that deletion cannot safely identify.
                    duplicate.source_tracking_complete = False
                    changed = True
                # A later hostile derivation may quarantine an older copy.
                if trust not in _RECALLABLE_TIERS and duplicate.trusted_for_recall:
                    duplicate.trust = trust or TRUST_UNTRUSTED
                    changed = True
                if changed:
                    self._flush()
                return False
            self._facts.append(
                Fact(
                    text=text,
                    source=source,
                    created_at=time.time(),
                    trust=trust,
                    source_ids=[episode_id] if episode_id else [],
                    source_tracking_complete=bool(episode_id),
                )
            )
            if len(self._facts) > self._max_facts:
                self._facts = self._facts[-self._max_facts :]
            self._flush()
        return True

    def purge_episode_sources(
        self,
        episode_ids: Iterable[str] | None,
    ) -> dict[str, int]:
        """Remove episode-only facts while preserving shared or legacy origins."""
        selected = (
            None
            if episode_ids is None
            else {
                normalized
                for value in episode_ids
                if (normalized := self._normalize_episode_id(value))
            }
        )
        removed_facts = 0
        removed_links = 0
        retained_facts = 0
        with self._lock, _cross_process_lock(self._lock_path()):
            self._sync_from_disk_locked()
            kept: list[Fact] = []
            for fact in self._facts:
                removed = [
                    source_id
                    for source_id in fact.source_ids
                    if selected is None or source_id in selected
                ]
                if not removed:
                    kept.append(fact)
                    continue
                removed_links += len(removed)
                fact.source_ids = [
                    source_id
                    for source_id in fact.source_ids
                    if source_id not in removed
                ]
                if fact.source_tracking_complete and not fact.source_ids:
                    removed_facts += 1
                    continue
                retained_facts += 1
                kept.append(fact)
            changed = removed_links > 0 or removed_facts > 0
            self._facts = kept
            if changed:
                self._flush()
            untracked_remaining = sum(
                not fact.source_tracking_complete for fact in self._facts
            )
        return {
            "removed_facts": removed_facts,
            "removed_links": removed_links,
            "retained_facts": retained_facts,
            "untracked_remaining": untracked_remaining,
            "unsupported": 0,
        }

    def set_trust(self, index: int, trust: str) -> bool:
        trust = (trust or "").strip().lower()
        if trust not in {TRUST_AUTO, TRUST_TRUSTED, TRUST_UNTRUSTED}:
            raise ValueError(f"Unknown fact trust tier: {trust!r}")
        with self._lock, _cross_process_lock(self._lock_path()):
            self._sync_from_disk_locked()
            if not 0 <= index < len(self._facts):
                return False
            self._facts[index].trust = trust
            self._flush()
        return True

    def remove_reviewed(self, index: int, *, expected_text: str | None = None) -> bool:
        """Atomically remove one fact after checking its reviewed identity."""
        with self._lock, _cross_process_lock(self._lock_path()):
            self._sync_from_disk_locked()
            if not 0 <= index < len(self._facts):
                return False
            if expected_text is not None and self._facts[index].text != expected_text:
                return False
            self._facts.pop(index)
            self._flush()
        return True

    def promote_reviewed(self, index: int, expected_text: str) -> bool:
        """Atomically promote exactly the fact a user reviewed."""
        with self._lock, _cross_process_lock(self._lock_path()):
            self._sync_from_disk_locked()
            if not 0 <= index < len(self._facts):
                return False
            fact = self._facts[index]
            if fact.text != expected_text or fact.trusted_for_recall or fact.superseded:
                return False
            fact.trust = TRUST_TRUSTED
            self._flush()
        return True

    def revise(self, expected_text: str, replacement: str) -> None:
        """Atomically supersede one exact statement and save its correction."""
        expected_text = expected_text.strip()
        replacement = replacement.strip()
        if not expected_text or not replacement:
            raise ValueError("Provide both the existing fact and its correction.")
        if expected_text.lower() == replacement.lower():
            raise ValueError("The correction must change the existing fact.")

        with self._lock, _cross_process_lock(self._lock_path()):
            self._sync_from_disk_locked()
            matches = [fact for fact in self._facts if fact.text == expected_text]
            if len(matches) != 1:
                raise ValueError(
                    "The existing fact is missing or ambiguous. "
                    "Use the exact wording from jarvis memory list."
                )
            current = matches[0]
            if current.superseded:
                raise ValueError("That fact is already superseded.")
            if not current.trusted_for_recall:
                raise ValueError(
                    "Only a current, recallable fact can be revised. "
                    "Review quarantined facts separately."
                )
            alternatives = [
                fact for fact in self._facts if fact.text.lower() == replacement.lower()
            ]
            if len(alternatives) > 1:
                raise ValueError("The correction matches multiple older records.")
            if alternatives:
                corrected = alternatives[0]
                if not corrected.superseded and not corrected.trusted_for_recall:
                    raise ValueError("The correction matches a quarantined fact.")
                corrected.superseded = False
                corrected.trust = TRUST_TRUSTED
                # The user's correction supplies a new origin independent
                # of any episode associated with the existing wording.
                corrected.source_tracking_complete = False
            else:
                if len(self._facts) >= self._max_facts:
                    raise ValueError(
                        "There is no room to retain this revision. "
                        "Forget a fact or increase the fact limit first."
                    )
                self._facts.append(
                    Fact(
                        text=replacement,
                        source="manual",
                        created_at=time.time(),
                        trust=TRUST_TRUSTED,
                    )
                )
            current.superseded = True
            self._flush()

    def list(self) -> List[Fact]:
        with self._lock, _cross_process_lock(self._lock_path()):
            self._sync_from_disk_locked()
            return list(self._facts)

    def clear(self) -> int:
        with self._lock, _cross_process_lock(self._lock_path()):
            self._sync_from_disk_locked()
            removed = len(self._facts)
            self._facts = []
            if self._path.exists():
                try:
                    self._path.unlink()
                except OSError:
                    self._flush()
        return removed

    def count(self) -> int:
        with self._lock, _cross_process_lock(self._lock_path()):
            self._sync_from_disk_locked()
            return len(self._facts)

    @property
    def path(self) -> Path:
        """Filesystem location of the JSONL store."""
        return self._path


def _ensure_fact_store_backends_registered() -> None:
    """Restore built-in fact-store registrations if a test cleared registries."""
    if not FactStoreRegistry.contains("local"):
        FactStoreRegistry.register_value("local", LocalFactStore)


def create_fact_store(
    backend: str = "local",
    *,
    path: str | Path | None = None,
    max_facts: int = 1000,
) -> FactStore:
    """Construct a fact store for the configured *backend*.

    Only the ``"local"`` (on-disk JSONL) backend is supported today; the
    registry-backed constructor exists so additional backends can be added
    without changing the service or CLI wiring.
    """
    _ensure_fact_store_backends_registered()
    key = (backend or "local").strip().lower()
    if not FactStoreRegistry.contains(key):
        supported = ", ".join(FactStoreRegistry.keys())
        raise ValueError(
            f"Unknown memory backend '{backend}'. Supported backends: {supported}"
        )
    return FactStoreRegistry.create(key, path, max_facts=max_facts)


def load_configured_facts(config: Any) -> List[Fact]:
    """Load automatic-memory facts from *config* when the service is enabled.

    Context injection is also used by short-lived commands such as
    ``jarvis ask``, where no :class:`MemoryService` instance exists.  This
    helper gives those callers the same configured fact-store view without
    coupling them to the service lifecycle.
    """
    memory = getattr(config, "memory", None)
    if memory is None or not getattr(memory, "enabled", False):
        return []

    store = create_fact_store(
        getattr(memory, "backend", "local"),
        path=getattr(memory, "facts_path", None),
        max_facts=getattr(memory, "max_facts", 1000),
    )
    # This helper feeds short-lived model paths (ask, SDK, system
    # orchestrator). Keep the full list available through FactStore.list() for
    # auditing/CLI use, but never return quarantined facts for model recall.
    return [fact for fact in store.list() if fact.trusted_for_recall]


__all__ = [
    "RECALLABLE_TRUST_TIERS",
    "TRUST_AUTO",
    "TRUST_TRUSTED",
    "TRUST_UNTRUSTED",
    "Fact",
    "FactStore",
    "LocalFactStore",
    "create_fact_store",
    "load_configured_facts",
]
