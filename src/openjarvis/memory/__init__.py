"""Native persistent long-term memory for OpenJarvis.

This package provides background fact extraction, substantial conversation
episodes, and optional evidence-backed graph capture, persisted across
sessions. It is started and stopped as part of the ``jarvis serve`` /
``jarvis chat`` lifecycle and configured via ``[memory]`` in ``config.toml``.
"""

from __future__ import annotations

from openjarvis.memory.episodes import Episode, EpisodeStore
from openjarvis.memory.extractor import FactExtractor
from openjarvis.memory.graph_extractor import ConversationGraphExtractor
from openjarvis.memory.procedures import (
    ProcedureCandidate,
    ProcedureProposal,
    build_procedure_candidate,
    is_explicit_success,
)
from openjarvis.memory.service import (
    MemoryService,
    build_memory_service,
    publish_completed_exchange,
)
from openjarvis.memory.store import (
    Fact,
    FactStore,
    LocalFactStore,
    create_fact_store,
    load_configured_facts,
)

__all__ = [
    "Fact",
    "Episode",
    "EpisodeStore",
    "FactStore",
    "FactExtractor",
    "ConversationGraphExtractor",
    "ProcedureCandidate",
    "ProcedureProposal",
    "build_procedure_candidate",
    "is_explicit_success",
    "LocalFactStore",
    "MemoryService",
    "build_memory_service",
    "create_fact_store",
    "load_configured_facts",
    "publish_completed_exchange",
]
