"""Persistent memory service: async facts and optional graph capture.

``MemoryService`` runs fact extraction on a dedicated background thread so it
never blocks ``jarvis serve`` request handling or the ``jarvis chat`` REPL.
Callers hand off an exchange via :meth:`submit`, which enqueues the work and
returns immediately — local fact/graph extraction and persistence happen out
of band.
The worker swallows every per-job error (including ``BrokenPipeError`` when a
client disconnects mid-extraction), so a flaky extraction model can never take
down the host process.

The service is started and stopped as part of the OpenJarvis lifecycle (see
``cli/serve.py`` and ``cli/chat_cmd.py``) and is configured through the
``[memory]`` section of ``config.toml``.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, List, Literal, Optional

from openjarvis.core.events import Event, EventBus, EventType
from openjarvis.memory.episodes import EpisodeStore
from openjarvis.memory.extractor import FactExtractor
from openjarvis.memory.procedures import (
    build_procedure_candidate,
    has_structured_procedure,
    is_explicit_failure,
    is_explicit_success,
)
from openjarvis.memory.store import (
    TRUST_AUTO,
    TRUST_UNTRUSTED,
    Fact,
    FactStore,
    create_fact_store,
)

if TYPE_CHECKING:
    from openjarvis.memory.episodes import EpisodeDerivationClaim

logger = logging.getLogger(__name__)

_ASK_DERIVATION_POLL_SECONDS = 5.0

# Only these severities suppress a whole exchange; see _blocks_exchange().
_BLOCKING_THREAT_LEVELS = frozenset({"high", "critical"})

# Sentinel pushed onto the queue to wake the worker for shutdown.
_STOP = object()


class MemoryService:
    """Background long-term-memory extraction and persistence service."""

    def __init__(
        self,
        store: FactStore,
        extractor: FactExtractor,
        *,
        event_bus: EventBus | None = None,
        scanner: Any = None,
        episode_store: EpisodeStore | None = None,
        graph_extractor: Any = None,
        graph_db_path: str | None = None,
        secret_scanner: Any = None,
        require_secret_scanner: bool = False,
        procedure_proposals_enabled: bool = False,
        facts_semantic_enabled: bool = False,
        facts_path: str = "",
        facts_embedding_model: str = "nomic-embed-text",
        max_queue: int = 256,
        ask_derivation_kinds: tuple[str, ...] = (),
        ask_derivation_is_local: Callable[[], bool] | None = None,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._event_bus = event_bus
        self._scanner = scanner
        self._episode_store = episode_store
        self._graph_extractor = graph_extractor
        self._graph_db_path = graph_db_path
        self._secret_scanner = secret_scanner
        self._require_secret_scanner = require_secret_scanner
        self._procedure_proposals_enabled = bool(procedure_proposals_enabled)
        self._facts_semantic_enabled = bool(facts_semantic_enabled)
        self._facts_path = str(facts_path or "")
        self._facts_embedding_model = str(
            facts_embedding_model or "nomic-embed-text"
        ).strip()
        self._ask_derivation_kinds = ask_derivation_kinds
        self._ask_derivation_is_local = ask_derivation_is_local
        self._subscribed = False
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, max_queue))
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the background worker thread (idempotent)."""
        if self._running.is_set():
            return
        self._running.set()
        self._subscribe_events()
        self._thread = threading.Thread(
            target=self._loop,
            name="memory-service",
            daemon=True,
        )
        self._thread.start()
        logger.debug("Memory service started")

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the worker to drain and stop, then join it (idempotent)."""
        if not self._running.is_set():
            return
        self._running.clear()
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass  # worker will notice the cleared flag on its next loop
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None
        self._unsubscribe_events()
        logger.debug("Memory service stopped")

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    # -- submission ---------------------------------------------------------

    def submit(
        self,
        user_text: str,
        assistant_text: str = "",
        *,
        source: str = "conversation",
        session_id: str = "",
    ) -> bool:
        """Queue an exchange for extraction. Non-blocking; never raises.

        Returns True if the job was enqueued, False if the service is not
        running or the queue is full (in which case the exchange is dropped
        rather than blocking the caller — extraction is best-effort).
        """
        if not self._running.is_set():
            return False
        if not user_text or not user_text.strip():
            return False
        try:
            self._queue.put_nowait(
                (
                    user_text,
                    assistant_text,
                    source,
                    uuid.uuid4().hex,
                    str(session_id or "")[:128],
                )
            )
            return True
        except queue.Full:
            logger.debug("Memory service queue full; dropping exchange")
            return False

    def _subscribe_events(self) -> None:
        """Subscribe to lifecycle events that feed automatic memory."""
        if self._event_bus is None or self._subscribed:
            return
        self._event_bus.subscribe(
            EventType.CHAT_EXCHANGE_COMPLETED,
            self._on_completed_exchange,
        )
        self._subscribed = True

    def _unsubscribe_events(self) -> None:
        """Unsubscribe from lifecycle events (idempotent)."""
        if self._event_bus is None or not self._subscribed:
            return
        self._event_bus.unsubscribe(
            EventType.CHAT_EXCHANGE_COMPLETED,
            self._on_completed_exchange,
        )
        self._subscribed = False

    def _on_completed_exchange(self, event: Event) -> None:
        """Queue a completed chat exchange published on the event bus."""
        data = event.data or {}
        self.submit(
            str(data.get("user_text", "") or ""),
            str(data.get("assistant_text", "") or ""),
            source=str(data.get("source", "conversation") or "conversation"),
            session_id=str(data.get("session_id", "") or ""),
        )

    # -- worker -------------------------------------------------------------

    def _loop(self) -> None:
        next_derivation_check = 0.0
        while True:
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                if not self._running.is_set():
                    break
                if (
                    self._ask_derivation_kinds
                    and time.monotonic() >= next_derivation_check
                ):
                    try:
                        self.process_pending_ask_derivation()
                    except Exception:
                        logger.warning("Could not advance the ask episode queue")
                    next_derivation_check = (
                        time.monotonic() + _ASK_DERIVATION_POLL_SECONDS
                    )
                continue
            if job is _STOP:
                self._queue.task_done()
                break
            try:
                self._process(job)
            except Exception:  # noqa: BLE001 — a bad job must not kill the worker
                logger.debug("Memory extraction job failed", exc_info=True)
            finally:
                self._queue.task_done()
            if not self._running.is_set() and self._queue.empty():
                break

    def _scan(self, text: str) -> Optional[Any]:
        """Run the injection scanner, or return ``None`` when it is absent or
        errors. Scanning fails *open*: an outage must not silently switch off
        memory capture."""
        if self._scanner is None:
            return None
        try:
            return self._scanner.scan(text)
        except BaseException as exc:  # noqa: BLE001 — scanning is best-effort
            # PyO3 deliberately exposes a Rust panic as ``PanicException``, a
            # direct BaseException subclass. Keep the memory worker alive if a
            # native scanner ever panics, while preserving Python's control-
            # flow exceptions instead of swallowing shutdown/interrupts.
            rust_panic = (
                exc.__class__.__module__ == "pyo3_runtime"
                and exc.__class__.__name__ == "PanicException"
            )
            if not isinstance(exc, Exception) and not rust_panic:
                raise
            logger.debug("Injection scan failed; proceeding (fail-open)", exc_info=True)
            return None

    @staticmethod
    def _flagged(result: Any) -> bool:
        """True if *result* carries any finding at all."""
        return result is not None and not getattr(result, "is_clean", True)

    @classmethod
    def _blocks_exchange(cls, result: Any) -> bool:
        """True only for a *severe* finding.

        Dropping an exchange is destructive and unrecoverable, so it is
        reserved for HIGH/CRITICAL hits. Lower-severity patterns (a pasted
        shell one-liner, a role-delimiter fence quoted in a code discussion)
        are ordinary developer traffic; those exchanges are still extracted,
        and anything suspicious is caught per-fact by the quarantine tier.
        """
        if not cls._flagged(result):
            return False
        level = getattr(result, "threat_level", "")
        name = str(getattr(level, "value", level) or "").strip().lower()
        return name in _BLOCKING_THREAT_LEVELS

    def _process(self, job: Any) -> None:
        if len(job) >= 5:
            user_text, assistant_text, source, exchange_id, session_id = job[:5]
        elif len(job) >= 4:
            user_text, assistant_text, source, exchange_id = job[:4]
            session_id = ""
        else:
            # Keep direct callers using the original two-string form
            # source-compatible.
            user_text, assistant_text = job
            source = "conversation"
            exchange_id = uuid.uuid4().hex
            session_id = ""
        user_text = str(user_text or "")
        assistant_text = str(assistant_text or "")
        exchange_id = str(exchange_id or uuid.uuid4().hex)
        stored_episode_id = ""
        # Scan BEFORE extraction so an overt injection attempt never reaches the
        # extraction model or the store at all.
        if self._blocks_exchange(self._scan(f"{user_text}\n{assistant_text}")):
            # info, not debug: a silently-dropped exchange must be distinguishable
            # from "nothing to extract" in the logs.
            logger.info("Memory extraction skipped: injection detected in exchange")
            return
        if self._episode_store is not None:
            exchange_text = f"{user_text}\n{assistant_text}"
            # Episodes retain original conversation text, so secret scanning
            # and injection scanning both fail closed for this raw transcript.
            if self._episode_input_safe(exchange_text):
                try:
                    success_signal = (
                        self._procedure_proposals_enabled
                        and bool(session_id)
                        and is_explicit_success(user_text)
                    )
                    failure_signal = (
                        self._procedure_proposals_enabled
                        and bool(session_id)
                        and is_explicit_failure(user_text)
                    )
                    if success_signal:
                        prior_episodes = self._episode_store.list_for_session(
                            session_id,
                            limit=1,
                        )
                    elif failure_signal:
                        prior_episodes = [
                            episode
                            for episode in self._episode_store.list_for_session(
                                session_id,
                                limit=10,
                            )
                            if has_structured_procedure(episode.assistant_text)
                        ][:1]
                    else:
                        prior_episodes = []
                    if prior_episodes and (
                        time.time() - prior_episodes[0].created_at > 30 * 60
                    ):
                        prior_episodes = []
                    stored_episode = self._episode_store.record(
                        user_text,
                        assistant_text,
                        source=str(source),
                        session_id=str(session_id or ""),
                        idempotency_key=exchange_id,
                    )
                    if stored_episode:
                        stored_episode_id = self._episode_store.episode_id_for(
                            source=str(source),
                            session_id=str(session_id or ""),
                            idempotency_key=exchange_id,
                        )
                    outcome_episodes = (
                        self._episode_store.list_for_session(session_id, limit=1)
                        if stored_episode and prior_episodes
                        else []
                    )
                    if stored_episode and prior_episodes and outcome_episodes:
                        prior = prior_episodes[0]
                        outcome_episode = outcome_episodes[0]
                        if success_signal:
                            candidate = build_procedure_candidate(
                                episode_id=prior.episode_id,
                                user_text=prior.user_text,
                                assistant_text=prior.assistant_text,
                                success_signal=user_text,
                            )
                            if candidate is not None:
                                episode_store = self._episode_store
                                proposal = episode_store.create_procedure_proposal(
                                    candidate,
                                    evidence_episode_id=prior.episode_id,
                                    session_id=str(session_id),
                                )
                                if proposal is not None:
                                    logger.info(
                                        "Memory: proposed procedure %s from episode %s",
                                        proposal.proposal_id,
                                        prior.episode_id,
                                    )
                        elif failure_signal:
                            outcome = self._episode_store.record_procedure_outcome(
                                evidence_episode_id=outcome_episode.episode_id,
                                session_id=str(session_id),
                                task_title=prior.user_text,
                                outcome="failure",
                                signal=user_text,
                            )
                            if outcome is not None:
                                logger.info(
                                    "Memory: recorded a reported procedure failure "
                                    "for episode %s",
                                    prior.episode_id,
                                )
                except Exception:
                    logger.warning(
                        "Conversation episode persistence failed", exc_info=True
                    )
        try:
            facts = self._extractor.extract(user_text, assistant_text)
        except Exception:  # noqa: BLE001 — graph capture must still be attempted
            logger.debug("Fact extraction failed", exc_info=True)
            facts = []
        if facts:
            # Provenance is per fact, not per exchange: a fact whose own text
            # trips the scanner is quarantined, while clean facts stay
            # recallable.
            clean: List[str] = []
            quarantined: List[str] = []
            for fact in facts:
                target = quarantined if self._flagged(self._scan(fact)) else clean
                target.append(fact)
            self._persist_facts(clean, quarantined, stored_episode_id)
            if self._facts_semantic_enabled and self._facts_path:
                self._index_fact_embeddings(clean)

        self._capture_graph(
            user_text,
            source=str(source),
            exchange_id=exchange_id,
            episode_id=stored_episode_id,
        )

    def _persist_facts(
        self, clean: list[str], quarantined: list[str], episode_id: str
    ) -> None:
        stored = self._store.add_many_with_provenance(
            clean,
            source="auto",
            trust=TRUST_AUTO,
            source_id=episode_id or None,
        )
        if quarantined:
            stored += self._store.add_many_with_provenance(
                quarantined,
                source="auto",
                trust=TRUST_UNTRUSTED,
                source_id=episode_id or None,
            )
            logger.info(
                "Memory: quarantined %d extracted fact(s) as untrusted",
                len(quarantined),
            )
        if stored:
            logger.debug("Memory service stored %d new fact(s)", stored)

    def process_pending_ask_derivation(self) -> Literal["completed", "deferred"] | None:
        """Run one due task with verified local inference and a live source."""
        episodes = self._episode_store
        route_is_local = self._ask_derivation_is_local
        if (
            episodes is None
            or not self._ask_derivation_kinds
            or route_is_local is None
            or not route_is_local()
        ):
            return None
        claim = episodes.claim_derivation(self._ask_derivation_kinds)
        if claim is None:
            return None
        try:
            return (
                "completed" if self._derive_ask_episode(episodes, claim) else "deferred"
            )
        except Exception:
            # Transport and provider errors can include private request text.
            logger.info("Deferred an ask episode %s extraction", claim.kind)
            episodes.fail_derivation(claim)
            return "deferred"

    @staticmethod
    def _derivation_flagged(scanner: Any, text: str) -> bool:
        if scanner is None:
            raise RuntimeError("Required local memory scanner unavailable")
        try:
            result = scanner.scan(text)
        except BaseException as exc:
            if (
                exc.__class__.__module__ == "pyo3_runtime"
                and exc.__class__.__name__ == "PanicException"
            ):
                raise RuntimeError("Required local memory scanner failed") from None
            raise
        if not isinstance(getattr(result, "is_clean", None), bool):
            raise RuntimeError("Required local memory scan did not complete")
        return not result.is_clean

    def _safe_derivation_text(self, text: str) -> bool:
        return not (
            self._derivation_flagged(self._secret_scanner, text)
            or self._derivation_flagged(self._scanner, text)
        )

    def _derive_ask_episode(
        self, episodes: EpisodeStore, claim: EpisodeDerivationClaim
    ) -> bool:
        episode = claim.episode
        if not self._safe_derivation_text(
            f"{episode.user_text}\n{episode.assistant_text}"
        ):
            # A previously admitted episode can be rejected by newer scanners.
            return episodes.complete_derivation(claim, lambda: None)
        if self._ask_derivation_is_local is None or not self._ask_derivation_is_local():
            raise RuntimeError("The extraction model route is no longer local")

        if claim.kind == "facts":
            clean: list[str] = []
            quarantined: list[str] = []
            for fact in self._extractor.extract(
                episode.user_text, episode.assistant_text, strict=True
            ):
                if self._derivation_flagged(self._secret_scanner, fact):
                    continue
                target = (
                    quarantined
                    if self._derivation_flagged(self._scanner, fact)
                    else clean
                )
                target.append(fact)
            completed = episodes.complete_derivation(
                claim,
                lambda: self._persist_facts(clean, quarantined, episode.episode_id),
            )
            if (
                completed
                and clean
                and self._facts_semantic_enabled
                and self._facts_path
            ):
                self._index_fact_embeddings(clean)
            return completed

        graph = self._graph_extractor
        if claim.kind != "graph" or graph is None or not self._graph_db_path:
            raise RuntimeError("The extraction task is unavailable")
        extraction = graph.extract(episode.user_text, strict=True)

        def persist_graph() -> None:
            graph.persist(
                extraction,
                self._graph_db_path,
                f"conversation:{episode.source}:{episode.episode_id}",
                episode_id=episode.episode_id,
                quote_is_safe=self._safe_derivation_text,
            )

        return episodes.complete_derivation(claim, persist_graph)

    def _index_fact_embeddings(self, texts: List[str]) -> None:
        """Index newly extracted facts without delaying the conversation."""
        try:
            from openjarvis.memory.fact_embeddings import (
                FactEmbeddingIndex,
                fact_embedding_index_path,
            )

            requested = {text.strip().casefold() for text in texts}
            stored_facts = self._store.list()
            facts = [
                fact
                for fact in stored_facts
                if str(getattr(fact, "text", "")).strip().casefold() in requested
                and bool(getattr(fact, "trusted_for_recall", True))
            ]
            with FactEmbeddingIndex(
                fact_embedding_index_path(self._facts_path),
                embedding_model=self._facts_embedding_model,
            ) as index:
                index.prune_to(stored_facts)
                if facts:
                    index.index_missing_embeddings(facts, limit=len(facts))
        except RuntimeError:
            logger.debug(
                "Local durable-fact embeddings unavailable; using lexical recall"
            )
        except Exception:
            logger.debug("Durable-fact embedding persistence failed", exc_info=True)

    def _secret_flagged(self, text: str) -> bool:
        """Fail closed when configured secret scanning cannot verify text."""
        if self._secret_scanner is None:
            return self._require_secret_scanner
        try:
            return self._flagged(self._secret_scanner.scan(text))
        except Exception:
            logger.warning("Secret scan failed; skipping conversation graph capture")
            return True

    def _episode_input_safe(self, text: str) -> bool:
        """Require both local scanners to pass before retaining raw dialogue."""
        if self._secret_scanner is None or self._scanner is None:
            logger.info("Episode capture skipped: a required scanner is unavailable")
            return False
        try:
            if self._flagged(self._secret_scanner.scan(text)):
                logger.info("Episode capture skipped: possible secret detected")
                return False
            if self._flagged(self._scanner.scan(text)):
                logger.info("Episode capture skipped: untrusted input detected")
                return False
        except Exception:
            logger.warning("Episode capture skipped: a scanner failed", exc_info=True)
            return False
        return True

    def _capture_graph(
        self,
        user_text: str,
        *,
        source: str,
        exchange_id: str,
        episode_id: str = "",
    ) -> None:
        if self._graph_extractor is None or not self._graph_db_path:
            return
        if self._secret_flagged(user_text):
            logger.info("Conversation graph capture skipped: possible secret detected")
            return
        try:
            extraction = self._graph_extractor.extract(user_text)
            source_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", source)[:40]
            source_id = f"conversation:{source_name}:{exchange_id}"
            stored_entities, stored_relations = self._graph_extractor.persist(
                extraction,
                self._graph_db_path,
                source_id,
                episode_id=episode_id,
                quote_is_safe=lambda quote: (
                    not self._secret_flagged(quote)
                    and not self._flagged(self._scan(quote))
                ),
            )
            if stored_entities or stored_relations:
                logger.info(
                    "Conversation graph stored %d entities and %d relations",
                    stored_entities,
                    stored_relations,
                )
        except Exception:  # noqa: BLE001 — graph capture is best-effort
            logger.warning("Conversation graph capture failed", exc_info=True)

    # -- store passthroughs -------------------------------------------------

    def list_facts(self) -> List[Fact]:
        return self._store.list()

    def clear_facts(self) -> int:
        return self._store.clear()

    def fact_count(self) -> int:
        return self._store.count()


def build_memory_service(
    config: Any,
    engine: Any,
    default_model: str = "",
    *,
    event_bus: EventBus | None = None,
) -> Optional[MemoryService]:
    """Build a :class:`MemoryService` from config, or ``None`` if disabled.

    Reads ``[memory]`` for fact-memory settings and optional conversation-graph
    capture. Returns ``None`` when memory is disabled or no engine /
    extraction model is available, so callers can simply do::

        svc = build_memory_service(config, engine, model)
        if svc is not None:
            svc.start()
    """
    mem = getattr(config, "memory", None)
    if mem is None or not getattr(mem, "enabled", False):
        return None
    if engine is None:
        return None

    model = getattr(mem, "extraction_model", "") or default_model
    if not model:
        logger.debug("Memory service disabled: no extraction model available")
        return None

    from openjarvis.engine._locality import is_local_engine_for_model

    store = create_fact_store(
        getattr(mem, "backend", "local"),
        path=getattr(mem, "facts_path", None),
        max_facts=getattr(mem, "max_facts", 1000),
    )
    extractor = FactExtractor(engine, model)
    graph_extractor = None
    graph_db_path = None
    secret_scanner = None
    episode_store = None
    episodes_enabled = bool(getattr(mem, "episodes_enabled", False))
    if episodes_enabled:
        try:
            episode_store = EpisodeStore(
                getattr(mem, "episodes_path", None),
                max_episodes=getattr(mem, "episodes_max_count", 5_000),
                retention_days=getattr(mem, "episodes_retention_days", 365),
                semantic_enabled=getattr(mem, "episodes_semantic_enabled", False),
                embedding_model=getattr(
                    mem, "episodes_embedding_model", "nomic-embed-text"
                ),
            )
            # Raw exchanges are more sensitive than extracted facts. Never
            # persist them when no secret scanner is available.
        except Exception:
            logger.warning("Conversation episode store unavailable", exc_info=True)
    security = getattr(config, "security", None)
    graph_capture_enabled = bool(getattr(mem, "graph_capture_enabled", False))
    require_secret_scanner = bool(
        episode_store is not None
        or (
            graph_capture_enabled
            and security is not None
            and getattr(security, "secret_scanner", False)
        )
    )
    if require_secret_scanner:
        secret_scanner = _build_secret_scanner()
    if graph_capture_enabled:
        if not is_local_engine_for_model(engine, model):
            logger.warning(
                "Conversation graph capture disabled for an unverified non-local "
                "inference engine"
            )
        else:
            if require_secret_scanner and secret_scanner is None:
                logger.warning(
                    "Conversation graph capture disabled: configured secret "
                    "scanner is unavailable"
                )
            else:
                from openjarvis.memory.graph_extractor import (
                    ConversationGraphExtractor,
                )

                graph_extractor = ConversationGraphExtractor(engine, model)
                graph_db_path = getattr(mem, "graph_db_path", None)
                if not graph_db_path:
                    from openjarvis.core.config import DEFAULT_CONFIG_DIR

                    graph_db_path = str(DEFAULT_CONFIG_DIR / "knowledge_graph.db")
    injection_scanner = _build_scanner()
    if episode_store is not None and (
        secret_scanner is None or injection_scanner is None
    ):
        logger.warning(
            "Conversation episode capture disabled: a required local scanner "
            "is unavailable"
        )
        episode_store.close()
        episode_store = None
    available_derivations = tuple(
        kind
        for kind in _ask_derivation_kinds(mem)
        if kind != "graph" or (graph_extractor is not None and graph_db_path)
    )
    return MemoryService(
        store,
        extractor,
        event_bus=event_bus,
        scanner=injection_scanner,
        episode_store=episode_store,
        graph_extractor=graph_extractor,
        graph_db_path=graph_db_path,
        secret_scanner=secret_scanner,
        require_secret_scanner=require_secret_scanner,
        procedure_proposals_enabled=bool(
            episode_store is not None
            and getattr(mem, "procedure_proposals_enabled", False)
        ),
        facts_semantic_enabled=bool(getattr(mem, "facts_semantic_enabled", False)),
        facts_path=str(getattr(mem, "facts_path", "") or ""),
        facts_embedding_model=str(
            getattr(mem, "facts_embedding_model", "nomic-embed-text")
        ),
        ask_derivation_kinds=available_derivations if episode_store is not None else (),
        ask_derivation_is_local=lambda: is_local_engine_for_model(engine, model),
    )


def _ask_derivation_kinds(mem: Any) -> tuple[str, ...]:
    """Queue only supported, source-aware fact and opted-in graph derivations."""
    facts = ("facts",) if getattr(mem, "backend", "local") == "local" else ()
    graph = ("graph",) if getattr(mem, "graph_capture_enabled", False) else ()
    return facts + graph


def drain_ask_derivations(
    config: Any,
    engine: Any,
    default_model: str,
    *,
    limit: int,
) -> tuple[int, int]:
    """Process up to limit due tasks without starting a background worker."""
    service = build_memory_service(config, engine, default_model)
    if service is None or service._episode_store is None:
        return (0, 0)

    completed = deferred = 0
    try:
        for _ in range(max(1, min(int(limit), 100))):
            result = service.process_pending_ask_derivation()
            if result is None:
                break
            completed += result == "completed"
            deferred += result == "deferred"
    finally:
        service._episode_store.close()
    return completed, deferred


def _build_scanner() -> Any:
    """Construct an injection scanner, or ``None`` if unavailable. Never raises —
    memory must work even if the security module can't load."""
    try:
        from openjarvis.security.injection_scanner import InjectionScanner

        return InjectionScanner()
    except Exception:  # noqa: BLE001 — scanner is an optional defence layer
        logger.debug(
            "Injection scanner unavailable; memory capture unguarded", exc_info=True
        )
        return None


def _build_secret_scanner() -> Any:
    """Construct the local secret scanner, with a Python pattern fallback."""
    try:
        from openjarvis.security.scanner import SecretScanner

        return SecretScanner()
    except Exception:  # noqa: BLE001 — retain local capture without the Rust extra
        logger.debug("Native secret scanner unavailable; using Python patterns")

    try:
        from openjarvis.security.scanner import SecretScanner

        patterns = [
            re.compile(pattern)
            for pattern, _level, _description in SecretScanner.PATTERNS.values()
        ]
    except Exception:
        logger.warning("Secret scanner patterns unavailable", exc_info=True)
        return None

    class _ScanResult:
        def __init__(self, is_clean: bool) -> None:
            self.is_clean = is_clean

    class _PythonSecretScanner:
        def scan(self, text: str) -> Any:
            clean = not any(pattern.search(text or "") for pattern in patterns)
            return _ScanResult(clean)

    return _PythonSecretScanner()


def is_complete_text_reply(result: Mapping[str, Any]) -> bool:
    """Accept final text replies without tool activity or interrupted generation."""
    return (
        not result.get("tool_calls")
        and not result.get("tool_results")
        and result.get("finish_reason") in (None, "", "stop", "end_turn", "eos")
    )


def record_ask_episode(
    config: Any,
    user_text: str,
    assistant_text: str,
    *,
    source: str = "cli.ask",
) -> bool:
    """Best-effort persistence for a completed, verified local text request."""
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        return False
    mem = getattr(config, "memory", None)
    if not getattr(mem, "enabled", False) or not getattr(
        mem, "episodes_enabled", False
    ):
        return False
    try:
        return _record_scanned_episode(
            mem,
            user_text,
            assistant_text,
            source=source,
            derivations=_ask_derivation_kinds(mem),
        )
    except Exception:
        logger.warning("Could not save an ask episode")
        return False


def record_external_episode(
    config: Any,
    user_text: str,
    assistant_text: str,
    *,
    source: str,
    session_id: str,
    idempotency_key: str,
) -> bool:
    """Capture an explicitly enabled, scanner-approved external event.

    Supported sources have separate opt-ins. This path stores no embeddings
    synchronously and fails closed if either scanner is unavailable or flags
    the synthetic event record.
    """
    mem = getattr(config, "memory", None)
    proactive = getattr(config, "proactive", None)
    capture_enabled = {
        "local_monitor": getattr(
            proactive, "local_monitor_memory_capture_enabled", False
        ),
        "proactive_action": getattr(
            proactive, "proactive_action_memory_capture_enabled", False
        ),
    }.get(source, False)
    if (
        not capture_enabled
        or not getattr(mem, "episodes_enabled", False)
        or not session_id
        or not idempotency_key
    ):
        return False
    return _record_scanned_episode(
        mem,
        user_text,
        assistant_text,
        source=source,
        session_id=session_id,
        idempotency_key=idempotency_key,
    )


def _record_scanned_episode(
    mem: Any,
    user_text: str,
    assistant_text: str,
    *,
    source: str,
    session_id: str = "",
    idempotency_key: str = "",
    derivations: tuple[str, ...] = (),
) -> bool:
    if not EpisodeStore.is_worthy(user_text, assistant_text):
        return False

    scanner = _build_scanner()
    secret_scanner = _build_secret_scanner()
    if scanner is None or secret_scanner is None:
        logger.info(
            "Episode capture skipped for %s: a required scanner is unavailable",
            source,
        )
        return False
    exchange_text = f"{user_text}\n{assistant_text}"
    try:
        if MemoryService._flagged(secret_scanner.scan(exchange_text)):
            logger.info(
                "Episode capture skipped for %s: possible secret detected", source
            )
            return False
        if MemoryService._flagged(scanner.scan(exchange_text)):
            logger.info(
                "Episode capture skipped for %s: untrusted input detected", source
            )
            return False
    except Exception:
        logger.warning("Episode capture skipped for %s: a scanner failed", source)
        return False

    store = EpisodeStore(
        getattr(mem, "episodes_path", None),
        max_episodes=getattr(mem, "episodes_max_count", 5_000),
        retention_days=getattr(mem, "episodes_retention_days", 365),
        semantic_enabled=False,
        embedding_model=getattr(mem, "episodes_embedding_model", "nomic-embed-text"),
    )
    try:
        return store.record(
            user_text,
            assistant_text,
            source=source,
            session_id=session_id,
            idempotency_key=idempotency_key,
            derivations=derivations,
        )
    finally:
        store.close()


def record_proactive_action_outcome(
    config: Any,
    *,
    action_type: str,
    succeeded: bool,
    action_id: str,
) -> bool:
    """Store a minimal, idempotent outcome for an executed connector action.

    Only the built-in action category and success state enter the episode. The
    connector payload, target, description, and response are intentionally
    excluded because they can contain private user or third-party content.
    """
    supported_actions = {
        "email_delete",
        "email_archive",
        "sms_send",
        "sms_draft_reply",
        "calendar_decline",
        "calendar_accept",
    }
    normalized_type = str(action_type or "").strip()
    stable_action_id = str(action_id or "").strip()
    if normalized_type not in supported_actions or not stable_action_id:
        return False
    outcome = "succeeded" if succeeded else "failed"
    user_text = f"An approved proactive {normalized_type} action outcome was recorded."
    assistant_text = (
        "The connector action completed successfully."
        if succeeded
        else "The connector action failed and was returned for review."
    )
    return record_external_episode(
        config,
        user_text,
        assistant_text,
        source="proactive_action",
        session_id="proactive_actions",
        idempotency_key=f"{stable_action_id}:{outcome}",
    )


def publish_completed_exchange(
    bus: EventBus | None,
    user_text: str,
    assistant_text: str = "",
    *,
    source: str = "",
    session_id: str = "",
) -> bool:
    """Publish a completed chat exchange for lifecycle subscribers."""
    if bus is None or not user_text or not user_text.strip():
        return False
    bus.publish(
        EventType.CHAT_EXCHANGE_COMPLETED,
        {
            "user_text": user_text,
            "assistant_text": assistant_text or "",
            "source": source,
            "session_id": str(session_id or "")[:128],
        },
    )
    return True


__all__ = [
    "MemoryService",
    "build_memory_service",
    "publish_completed_exchange",
    "is_complete_text_reply",
    "drain_ask_derivations",
    "record_ask_episode",
    "record_external_episode",
    "record_proactive_action_outcome",
]
