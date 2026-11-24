"""Extended API routes for agents, workflows, memory, traces, etc."""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlsplit

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
_MEMORY_BACKEND_LOCK_SETUP = threading.Lock()
GraphStatus = Literal[
    "reported", "hypothesis", "verified", "contradicted", "superseded"
]

# ---- Request/Response models ----


class AgentCreateRequest(BaseModel):
    agent_type: str
    tools: Optional[List[str]] = None
    agent_id: Optional[str] = None


class AgentMessageRequest(BaseModel):
    message: str


class SpeechSynthesizeRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    speed: Optional[float] = None


class MemoryStoreRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = 5


class MemoryIndexRequest(BaseModel):
    path: str
    ocr: bool = False
    transcribe_media: bool = False


class GraphEntityRequest(BaseModel):
    entity_id: str
    entity_type: str = "concept"
    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    source: str = "manual"
    source_url: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: GraphStatus = "reported"


class GraphRelationRequest(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    status: GraphStatus = "reported"
    polarity: Literal["affirmed", "negated"] = "affirmed"
    source: str = "manual"
    source_url: str = ""
    evidence_quote: str = ""
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None


class BudgetLimitsRequest(BaseModel):
    max_tokens_per_day: Optional[int] = None
    max_requests_per_hour: Optional[int] = None


class FeedbackScoreRequest(BaseModel):
    trace_id: str
    score: float
    source: str = "api"


class OptimizeRunRequest(BaseModel):
    benchmark: str
    max_trials: int = 20
    optimizer_model: str = "claude-sonnet-4-6"
    max_samples: int = 50


# ---- Agent routes ----

agents_router = APIRouter(prefix="/v1/agents", tags=["agents"])


def _execute_agent_admin_tool(request: Request, tool: Any, params: Dict[str, Any]):
    """Execute an agent lifecycle operation through server security gates."""
    from openjarvis.security.runtime import execute_secured_tool

    state = request.app.state
    return execute_secured_tool(
        tool,
        params,
        bus=getattr(state, "bus", None),
        capability_policy=getattr(state, "capability_policy", None),
        rate_limiter=getattr(state, "rate_limiter", None),
        agent_id="server:api",
    )


def _raise_agent_tool_failure(result: Any, *, not_found: bool = False) -> None:
    if result.success:
        return
    if "Capability '" in result.content and " denied " in result.content:
        raise HTTPException(status_code=403, detail=result.content)
    if result.content.startswith("Rate limit exceeded"):
        raise HTTPException(status_code=429, detail=result.content)
    raise HTTPException(status_code=404 if not_found else 400, detail=result.content)


@agents_router.get("")
async def list_agents(request: Request):
    """List available agent types and running agents."""
    try:
        from openjarvis.tools.agent_tools import AgentListTool

        # Registry names and live agent metadata are administrative state,
        # just like spawn/send/kill.  Authorize before reading either source
        # so default-deny and rate-limit policies cannot be bypassed by GET.
        result = _execute_agent_admin_tool(request, AgentListTool(), {})
        _raise_agent_tool_failure(result)
    except ImportError:
        raise HTTPException(status_code=501, detail="Agent tools not available")

    registered = []
    try:
        import openjarvis.agents  # noqa: F401 — side-effect registration
        from openjarvis.core.registry import AgentRegistry

        for key in sorted(AgentRegistry.keys()):
            cls = AgentRegistry.get(key)
            registered.append(
                {
                    "key": key,
                    "class": cls.__name__,
                    "accepts_tools": getattr(cls, "accepts_tools", False),
                }
            )
    except Exception as exc:
        logger.warning("Failed to list registered agents: %s", exc)

    running = []
    try:
        from openjarvis.tools.agent_tools import _SPAWNED_AGENTS

        running = [{"id": k, **v} for k, v in _SPAWNED_AGENTS.items()]
    except ImportError:
        pass

    return {"registered": registered, "running": running}


@agents_router.post("")
async def create_agent(req: AgentCreateRequest, request: Request):
    """Spawn a new agent."""
    try:
        from openjarvis.tools.agent_tools import AgentSpawnTool

        tool = AgentSpawnTool()
        params = {"agent_type": req.agent_type}
        if req.tools:
            params["tools"] = ",".join(req.tools)
        if req.agent_id:
            params["agent_id"] = req.agent_id
        result = _execute_agent_admin_tool(request, tool, params)
        _raise_agent_tool_failure(result)
        return {
            "status": "created",
            "content": result.content,
            "metadata": result.metadata,
        }
    except ImportError:
        raise HTTPException(status_code=501, detail="Agent tools not available")


@agents_router.delete("/{agent_id}")
async def kill_agent(agent_id: str, request: Request):
    """Kill a running agent."""
    try:
        from openjarvis.tools.agent_tools import AgentKillTool

        tool = AgentKillTool()
        result = _execute_agent_admin_tool(request, tool, {"agent_id": agent_id})
        _raise_agent_tool_failure(result, not_found=True)
        return {"status": "stopped", "agent_id": agent_id}
    except ImportError:
        raise HTTPException(status_code=501, detail="Agent tools not available")


@agents_router.post("/{agent_id}/message")
async def message_agent(agent_id: str, req: AgentMessageRequest, request: Request):
    """Send a message to a running agent."""
    try:
        from openjarvis.tools.agent_tools import AgentSendTool

        tool = AgentSendTool()
        result = _execute_agent_admin_tool(
            request,
            tool,
            {"agent_id": agent_id, "message": req.message},
        )
        _raise_agent_tool_failure(result, not_found=True)
        return {"status": "sent", "content": result.content}
    except ImportError:
        raise HTTPException(status_code=501, detail="Agent tools not available")


# ---- Memory routes ----

memory_router = APIRouter(prefix="/v1/memory", tags=["memory"])


def _is_loopback_address(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) else None
    return address.is_loopback or bool(mapped and mapped.is_loopback)


def _allows_private_memory_access(request: Request) -> bool:
    """Limit private memory to direct loopback clients and local browser origins."""
    if request.client is None or not _is_loopback_address(request.client.host):
        return False
    if any(
        name in request.headers
        for name in ("forwarded", "x-forwarded-for", "x-forwarded-host")
    ):
        return False
    origin = request.headers.get("origin")
    try:
        host = urlsplit("//" + request.headers.get("host", ""))
        if (
            host.username is not None
            or host.path
            or host.query
            or host.fragment
            or host.port == 0
            or not (
                host.hostname == "localhost"
                or _is_loopback_address(host.hostname or "")
            )
        ):
            return False
        if origin is None or origin in {
            "tauri://localhost",
            "http://tauri.localhost",
            "https://tauri.localhost",
        }:
            return True
        browser_origin = urlsplit(origin)
        return (
            browser_origin.scheme in {"http", "https"}
            and browser_origin.username is None
            and not browser_origin.path
            and not browser_origin.query
            and not browser_origin.fragment
            and browser_origin.port != 0
            and (
                browser_origin.hostname == "localhost"
                or _is_loopback_address(browser_origin.hostname or "")
            )
        )
    except ValueError:
        return False


@memory_router.get("/episodes/{episode_id}")
def memory_episode(episode_id: str, request: Request, response: Response) -> dict:
    """Read a locally retained exchange by its full ID."""
    if not _allows_private_memory_access(request):
        raise HTTPException(status_code=403, detail="Episode viewing is local only.")

    from openjarvis.core.config import load_config
    from openjarvis.memory.episodes import EpisodeStore

    config = getattr(request.app.state, "config", None) or load_config()
    memory = config.memory
    try:
        episode = (
            EpisodeStore.read_retained(
                episode_id,
                path=memory.episodes_path,
                max_episodes=memory.episodes_max_count,
                retention_days=memory.episodes_retention_days,
            )
            if memory.episodes_enabled
            else None
        )
    except (OSError, sqlite3.Error):
        logger.warning("Could not read the retained episode archive", exc_info=True)
        raise HTTPException(
            status_code=503, detail="The episode archive is unavailable."
        ) from None
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode unavailable.")
    response.headers["Cache-Control"] = "no-store"
    return {
        "episode_id": episode.episode_id,
        "user_text": episode.user_text,
        "assistant_text": episode.assistant_text,
        "source": episode.source,
        "created_at": episode.created_at,
    }


def _get_local_fact_store(request: Request):
    from openjarvis.core.config import get_config_dir, load_config
    from openjarvis.memory.store import LocalFactStore

    config = getattr(request.app.state, "config", None) or load_config()
    memory = config.memory
    if (memory.backend or "local").strip().lower() != "local":
        raise HTTPException(status_code=501, detail="Local fact review is unavailable.")

    path = Path(
        memory.facts_path or get_config_dir() / "memory_facts.jsonl"
    ).expanduser()
    if not path.exists():
        return None
    try:
        return LocalFactStore(path, max_facts=memory.max_facts)
    except (OSError, ValueError):
        raise HTTPException(
            status_code=503, detail="The fact store is unavailable."
        ) from None


@memory_router.get("/facts")
def memory_facts(request: Request, response: Response) -> dict:
    """List durable fact text and recall status for local review."""
    if not _allows_private_memory_access(request):
        raise HTTPException(status_code=403, detail="Fact review is local only.")
    store = _get_local_fact_store(request)
    try:
        facts = store.list() if store is not None else []
    except (OSError, ValueError):
        raise HTTPException(
            status_code=503, detail="The fact store is unavailable."
        ) from None

    response.headers["Cache-Control"] = "no-store"
    return {
        "facts": [
            {
                "text": fact.text,
                "source": fact.source,
                "trust": fact.trust,
                "superseded": fact.superseded,
                "trusted_for_recall": fact.trusted_for_recall,
            }
            for fact in reversed(facts)
        ]
    }


def _revise_local_fact(request: Request, existing_fact: str, correction: str) -> None:
    from openjarvis.memory.service import _build_scanner, _build_secret_scanner

    store = _get_local_fact_store(request)
    if store is None:
        raise HTTPException(status_code=409, detail="The fact store is empty.")

    for scanner_factory in (_build_secret_scanner, _build_scanner):
        try:
            scanner = scanner_factory()
            clean = scanner is not None and scanner.scan(correction).is_clean is True
        except Exception:
            clean = False
        if not clean:
            raise HTTPException(
                status_code=422,
                detail="The local scanners could not clear this correction.",
            )

    try:
        store.revise(existing_fact, correction)
    except ValueError:
        raise HTTPException(
            status_code=409, detail="The saved fact could not be revised."
        ) from None
    except OSError:
        raise HTTPException(
            status_code=503, detail="The fact store is unavailable."
        ) from None


@memory_router.post("/facts/revise", status_code=204)
async def memory_fact_revise(request: Request) -> Response:
    """Replace the exact wording of a recallable fact after local scanning."""
    if not _allows_private_memory_access(request):
        raise HTTPException(status_code=403, detail="Fact review is local only.")

    content_type = request.headers.get("content-type", "").split(";", 1)[0]
    if content_type.strip().lower() != "application/json":
        raise HTTPException(status_code=415, detail="A JSON correction is required.")

    # Bound input before parsing. Validation errors never echo private fact text.
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 65_536:
            raise HTTPException(status_code=413, detail="The correction is too large.")
        body.extend(chunk)
    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):
        payload = None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid fact correction.")

    existing_fact = payload.get("existing_fact")
    correction = payload.get("correction")
    if (
        not isinstance(existing_fact, str)
        or not existing_fact.strip()
        or not isinstance(correction, str)
        or not correction.strip().isprintable()
        or len(correction.strip()) > 1_000
    ):
        raise HTTPException(
            status_code=400,
            detail="Specify a saved fact and a one-line correction of up to 1,000 characters.",
        )

    await run_in_threadpool(_revise_local_fact, request, existing_fact, correction.strip())
    return Response(status_code=204, headers={"Cache-Control": "no-store"})


def _get_graph_backend(request: Request):
    from openjarvis.core.config import load_config
    from openjarvis.tools.storage.knowledge_graph import KnowledgeGraphMemory

    config = getattr(request.app.state, "config", None) or load_config()
    return KnowledgeGraphMemory(db_path=config.memory.graph_db_path)


def _graph_entity_json(entity) -> dict:
    return {
        "id": entity.entity_id,
        "type": entity.entity_type,
        "name": entity.name,
        "properties": entity.properties,
        "source": entity.source,
        "source_url": entity.source_url,
        "confidence": entity.confidence,
        "status": entity.status,
        "created_at": entity.created_at,
        "updated_at": entity.updated_at,
    }


def _graph_relation_json(relation) -> dict:
    return {
        "id": relation.relation_id,
        "source": relation.source_id,
        "target": relation.target_id,
        "type": relation.relation_type,
        "weight": relation.weight,
        "properties": relation.properties,
        "confidence": relation.confidence,
        "status": relation.status,
        "polarity": relation.polarity,
        "evidence": relation.evidence,
        "created_at": relation.created_at,
        "updated_at": relation.updated_at,
    }


def _graph_evidence(req: GraphRelationRequest) -> list[dict[str, Any]]:
    if not (req.evidence_quote or req.source_url or req.valid_from or req.valid_until):
        return []
    return [
        {
            "source": req.source,
            "source_url": req.source_url,
            "quote": req.evidence_quote,
            "confidence": req.confidence,
            "status": req.status,
            "valid_from": req.valid_from or "",
            "valid_until": req.valid_until or "",
        }
    ]


def _graph_event_snapshot(event_type: str, value: dict[str, Any]) -> dict[str, Any]:
    """Keep event history useful while bounding stored text returned to the UI."""

    def short_text(item: Any, *, limit: int = 500) -> str:
        return str(item or "")[:limit]

    def count(item: Any) -> int:
        try:
            return max(0, int(item))
        except (TypeError, ValueError, OverflowError):
            return 0

    if not value:
        return {}
    if event_type == "entity_deleted" and value.get("details_purged") is True:
        return {
            "details_purged": True,
            "removed_relation_count": count(value.get("removed_relation_count", 0)),
        }
    if event_type == "relation_deleted" and value.get("details_purged") is True:
        return {
            "details_purged": True,
            "evidence_count": count(value.get("evidence_count", 0)),
        }
    if event_type.startswith("entity_"):
        entity = value.get("entity") if isinstance(value.get("entity"), dict) else value
        result = {
            key: entity.get(key)
            for key in ("entity_id", "entity_type", "name", "status", "confidence")
            if key in entity
        }
        if "source_url" in entity:
            result["source_url"] = short_text(entity.get("source_url"))
        raw_properties = entity.get("properties")
        if isinstance(raw_properties, str):
            try:
                raw_properties = json.loads(raw_properties)
            except json.JSONDecodeError:
                raw_properties = {}
        if isinstance(raw_properties, dict):
            result["property_keys"] = sorted(
                short_text(key, limit=80) for key in raw_properties
            )[:30]
        if event_type == "entity_deleted" and isinstance(value.get("relations"), list):
            result["removed_relation_count"] = len(value["relations"])
        return result
    if event_type.startswith("relation_"):
        relation = (
            value.get("relation") if isinstance(value.get("relation"), dict) else value
        )
        result = {
            key: relation.get(key)
            for key in (
                "relation_key",
                "source_id",
                "target_id",
                "relation_type",
                "status",
                "polarity",
                "confidence",
            )
            if key in relation
        }
        evidence = value.get("evidence")
        if isinstance(evidence, list):
            result["evidence"] = [
                {
                    key: short_text(item.get(key), limit=500)
                    for key in ("source", "source_url", "quote", "status")
                    if item.get(key)
                }
                for item in evidence[:5]
                if isinstance(item, dict)
            ]
            result["evidence_count"] = int(value.get("evidence_count", len(evidence)))
        return result
    return {
        key: value[key]
        for key in (
            "entities",
            "relations",
            "entities_removed",
            "entities_updated",
            "relations_removed",
            "relations_updated",
            "evidence_removed",
        )
        if key in value and isinstance(value[key], (int, float))
    }


@memory_router.get("/graph")
def memory_graph(
    request: Request,
    query: str = "",
    entity_type: Optional[str] = None,
    limit: int = 200,
    focus_id: str | None = None,
    before_relation_id: int | None = None,
):
    """Return a bounded graph view or a page around one exact entity."""
    if focus_id is not None:
        if not focus_id.strip() or query.strip() or entity_type is not None:
            raise HTTPException(
                status_code=400,
                detail="Use an exact focus ID separately from global graph filters.",
            )
        if before_relation_id is not None and not 0 < before_relation_id <= 2**63 - 1:
            raise HTTPException(status_code=400, detail="Invalid graph page cursor.")
    elif before_relation_id is not None:
        raise HTTPException(
            status_code=400, detail="A graph page cursor needs a focus ID."
        )
    from pathlib import Path

    config = getattr(request.app.state, "config", None)
    if config is None:
        from openjarvis.core.config import load_config

        config = load_config()
    graph_path = Path(config.memory.graph_db_path).expanduser()
    if not graph_path.exists():
        if focus_id is not None:
            raise HTTPException(status_code=404, detail="Graph entity unavailable.")
        return {
            "entities": [],
            "relations": [],
            "possible_conflicts": [],
            "evidence_review_candidates": [],
            "counts": {"entities": 0, "relations": 0, "evidence": 0},
        }
    graph = _get_graph_backend(request)
    try:
        limit = max(1, min(int(limit), 200))
        next_relation_id = None
        if focus_id is not None:
            try:
                result, next_relation_id = graph.neighborhood_page(
                    focus_id, limit=limit, before_id=before_relation_id
                )
            except KeyError:
                raise HTTPException(
                    status_code=404, detail="Graph entity unavailable."
                ) from None
        elif query.strip():
            result = graph.search_graph(
                query, top_k=min(limit, 50), depth=1, max_relations=100
            )
        else:
            result = graph.query_pattern(entity_type=entity_type, limit=limit)
        entities = result.entities
        if entity_type:
            entities = [
                entity
                for entity in entities
                if entity.entity_type.casefold() == entity_type.casefold()
            ]
        entity_ids = {entity.entity_id for entity in entities}
        relations = [
            relation
            for relation in result.relations
            if relation.source_id in entity_ids and relation.target_id in entity_ids
        ]
        review_entities = [focus_id] if focus_id is not None else list(entity_ids)
        possible_conflicts = graph.possible_conflicts(
            entity_ids=review_entities, limit=min(limit, 50)
        )
        evidence_review_candidates = graph.evidence_review_candidates(
            entity_ids=review_entities, limit=min(limit, 50)
        )
        return {
            "next_relation_id": next_relation_id,
            "entities": [_graph_entity_json(entity) for entity in entities],
            "relations": [_graph_relation_json(relation) for relation in relations],
            "possible_conflicts": [
                {
                    "subject": _graph_entity_json(conflict["subject"]),
                    "predicate": conflict["predicate"],
                    "kind": conflict["kind"],
                    "claims": [
                        {
                            "relation": _graph_relation_json(claim["relation"]),
                            "target": _graph_entity_json(claim["target"]),
                        }
                        for claim in conflict["claims"]
                    ],
                }
                for conflict in possible_conflicts
            ],
            "evidence_review_candidates": [
                {
                    "subject": _graph_entity_json(candidate["subject"]),
                    "relation": _graph_relation_json(candidate["relation"]),
                    "target": _graph_entity_json(candidate["target"]),
                    "last_evidence": candidate["last_evidence"],
                    "last_evidence_at": candidate["last_evidence_at"],
                    "age_days": candidate["age_days"],
                    "review_after_days": candidate["review_after_days"],
                }
                for candidate in evidence_review_candidates
            ],
            "counts": {
                "entities": graph.entity_count(),
                "relations": graph.relation_count(),
                "evidence": graph.evidence_count(),
            },
        }
    finally:
        graph.close()


@memory_router.get("/graph/events")
def memory_graph_events(
    request: Request,
    limit: int = 20,
    before_id: Optional[int] = None,
):
    """Return a bounded, newest-first page of local graph change history."""
    config = getattr(request.app.state, "config", None)
    if config is None:
        from openjarvis.core.config import load_config

        config = load_config()
    graph_path = Path(config.memory.graph_db_path).expanduser()
    if not graph_path.exists():
        return {"events": [], "next_before_id": None}

    limit = max(1, min(int(limit), 100))
    graph = _get_graph_backend(request)
    try:
        rows = graph.recent_events(limit=limit + 1, before_id=before_id)
        has_more = len(rows) > limit
        page = rows[:limit]
        return {
            "events": [
                {
                    "id": event["id"],
                    "event_type": event["event_type"],
                    "entity_id": event["entity_id"],
                    "relation_id": event["relation_id"],
                    "source": str(event["source"] or "")[:120],
                    "created_at": event["created_at"],
                    "before": _graph_event_snapshot(
                        event["event_type"], event["before"]
                    ),
                    "after": _graph_event_snapshot(event["event_type"], event["after"]),
                }
                for event in page
            ],
            "next_before_id": page[-1]["id"] if has_more and page else None,
        }
    finally:
        graph.close()


@memory_router.post("/graph/entities")
def memory_graph_add_entity(req: GraphEntityRequest, request: Request):
    from openjarvis.tools.storage.knowledge_graph import Entity

    graph = _get_graph_backend(request)
    try:
        graph.add_entity(
            Entity(
                entity_id=req.entity_id,
                entity_type=req.entity_type,
                name=req.name,
                properties=req.properties,
                source=req.source,
                source_url=req.source_url,
                confidence=req.confidence,
                status=req.status,
            )
        )
        return {"entity": _graph_entity_json(graph.get_entity(req.entity_id))}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        graph.close()


@memory_router.post("/graph/relations")
def memory_graph_add_relation(req: GraphRelationRequest, request: Request):
    from openjarvis.tools.storage.knowledge_graph import Relation

    graph = _get_graph_backend(request)
    try:
        relation_key = graph.add_relation(
            Relation(
                source_id=req.source_id,
                target_id=req.target_id,
                relation_type=req.relation_type,
                weight=req.weight,
                confidence=req.confidence,
                status=req.status,
                polarity=req.polarity,
                source=req.source,
                source_url=req.source_url,
                evidence=_graph_evidence(req),
            )
        )
        return {"relation": _graph_relation_json(graph.relation_by_key(relation_key))}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        graph.close()


@memory_router.put("/graph/relations/{relation_id}")
def memory_graph_update_relation(
    relation_id: str, req: GraphRelationRequest, request: Request
):
    from openjarvis.tools.storage.knowledge_graph import Relation

    graph = _get_graph_backend(request)
    try:
        updated_key = graph.update_relation(
            relation_id,
            Relation(
                source_id=req.source_id,
                target_id=req.target_id,
                relation_type=req.relation_type,
                weight=req.weight,
                confidence=req.confidence,
                status=req.status,
                polarity=req.polarity,
                source=req.source,
                source_url=req.source_url,
                evidence=_graph_evidence(req),
            ),
        )
        if updated_key is None:
            raise HTTPException(status_code=404, detail="Relation not found")
        return {"relation": _graph_relation_json(graph.relation_by_key(updated_key))}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        graph.close()


@memory_router.delete("/graph/entities/{entity_id}")
def memory_graph_delete_entity(entity_id: str, request: Request):
    graph = _get_graph_backend(request)
    try:
        if not graph.delete(entity_id):
            raise HTTPException(status_code=404, detail="Entity not found")
        return {"deleted": entity_id}
    finally:
        graph.close()


@memory_router.delete("/graph/relations/{relation_id}")
def memory_graph_delete_relation(relation_id: str, request: Request):
    graph = _get_graph_backend(request)
    try:
        if not graph.delete_relation(relation_id):
            raise HTTPException(status_code=404, detail="Relation not found")
        return {"deleted": relation_id}
    finally:
        graph.close()


def _get_memory_backend(request: Request):
    """Return the app-level memory backend, creating it once when needed.

    The default SQLite backend uses Python's built-in FTS5 implementation when
    the optional ``openjarvis_rust`` extension is not installed. A 503 is
    reserved for an unavailable backend (for example, a SQLite build without
    FTS5), distinct from a benign "memory not configured" case.
    """
    state = request.app.state
    backend = getattr(state, "memory_backend", None)
    if backend is not None:
        return backend

    lock = getattr(state, "_memory_backend_lock", None)
    if lock is None:
        with _MEMORY_BACKEND_LOCK_SETUP:
            lock = getattr(state, "_memory_backend_lock", None)
            if lock is None:
                lock = threading.Lock()
                state._memory_backend_lock = lock
    with lock:
        backend = getattr(state, "memory_backend", None)
        if backend is not None:
            return backend
        from openjarvis.tools.storage._stubs import MemoryBackendUnavailable

        try:
            from openjarvis.tools.storage.sqlite import SQLiteMemory

            memory_config = getattr(getattr(state, "config", None), "memory", None)
            db_path = getattr(memory_config, "db_path", "")
            backend = SQLiteMemory(db_path=db_path)
        except MemoryBackendUnavailable as exc:
            logger.error("%s", exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception:
            logger.exception("Could not initialize the SQLite memory backend")
            return None
        state.memory_backend = backend
        state._owns_memory_backend = True
        return backend


def _local_graph_extraction_runtime(request: Request, config: Any):
    """Return a strictly local engine/model pair for document graph capture."""
    from openjarvis.engine._locality import is_local_engine

    engine = getattr(request.app.state, "engine", None)
    engine_name = str(getattr(request.app.state, "engine_name", "") or "")
    preferred_model = str(
        getattr(getattr(config, "memory", None), "extraction_model", "") or ""
    )

    def unwrap(candidate: Any) -> Any:
        seen: set[int] = set()
        while candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            inner = getattr(candidate, "_inner", None)
            if inner is None or inner is candidate:
                break
            candidate = inner
        return candidate

    def model_for(candidate: Any, *, allow_unlisted: bool = False) -> str:
        models: list[str] = []
        try:
            models = [str(model) for model in candidate.list_models() if model]
        except Exception:
            logger.debug("Could not list local extraction models", exc_info=True)
        active_model = str(getattr(request.app.state, "model", "") or "")
        for model in (preferred_model, active_model):
            if model and model in models:
                return model
        if models and (preferred_model or active_model):
            return ""
        if allow_unlisted:
            return preferred_model or active_model
        return ""

    if engine is None:
        return None

    # A MultiEngine may also contain cloud providers. Select one advertised
    # local engine directly instead of routing extraction through the mixed pool.
    if engine_name == "multi":
        for key, candidate in getattr(engine, "_engines", ()):
            if not is_local_engine(unwrap(candidate), str(key)):
                continue
            model = model_for(candidate)
            if model:
                return candidate, model
        return None

    if not is_local_engine(unwrap(engine), engine_name):
        return None
    model = model_for(engine, allow_unlisted=True)
    return (engine, model) if model else None


def _document_quote_safety(request: Request, config: Any):
    """Build local scanners and return a fail-closed graph quote predicate."""
    service = getattr(request.app.state, "memory_service", None)
    secret_scanner = getattr(service, "_secret_scanner", None)
    injection_scanner = getattr(service, "_scanner", None)
    try:
        from openjarvis.memory.service import _build_scanner, _build_secret_scanner

        if secret_scanner is None:
            secret_scanner = _build_secret_scanner()
        if injection_scanner is None:
            injection_scanner = _build_scanner()
    except Exception:
        logger.warning("Document graph scanners are unavailable", exc_info=True)

    security = getattr(config, "security", None)
    require_secret_scanner = bool(
        getattr(security, "secret_scanner", False) if security is not None else False
    )
    if injection_scanner is None or (require_secret_scanner and secret_scanner is None):
        return None

    def is_clean(scanner: Any, quote: str) -> bool:
        if scanner is None:
            return True
        result = scanner.scan(quote)
        return bool(getattr(result, "is_clean", True))

    def quote_is_safe(quote: str) -> bool:
        try:
            return is_clean(secret_scanner, quote) and is_clean(
                injection_scanner, quote
            )
        except Exception:
            logger.warning("Document graph quote scan failed; rejecting evidence")
            return False

    return quote_is_safe


def _capture_indexed_documents(
    request: Request, config: Any, chunks: list[Any]
) -> dict:
    """Capture bounded, source-linked graph claims from explicitly indexed text."""
    runtime = _local_graph_extraction_runtime(request, config)
    if runtime is None:
        return {"status": "skipped", "reason": "local_model_unavailable"}

    quote_is_safe = _document_quote_safety(request, config)
    if quote_is_safe is None:
        return {"status": "skipped", "reason": "local_scanners_unavailable"}

    grouped: dict[str, list[str]] = {}
    for chunk in chunks:
        source = str(getattr(chunk, "source", "") or "indexed document")
        grouped.setdefault(source, []).append(str(getattr(chunk, "content", "") or ""))

    max_documents = 12
    selected = list(grouped.items())[:max_documents]
    documents_processed = entities_stored = relations_stored = failures = 0
    try:
        from openjarvis.memory.graph_extractor import DocumentGraphExtractor

        engine, model = runtime
        extractor = DocumentGraphExtractor(engine, model)
        graph_db_path = getattr(config.memory, "graph_db_path", None)
        if not graph_db_path:
            from openjarvis.core.config import DEFAULT_CONFIG_DIR

            graph_db_path = str(DEFAULT_CONFIG_DIR / "knowledge_graph.db")

        for source, parts in selected:
            remaining = 10_000
            excerpt_parts: list[str] = []
            for part in parts:
                if remaining <= 0:
                    break
                excerpt = part[:remaining]
                if excerpt:
                    excerpt_parts.append(excerpt)
                    remaining -= len(excerpt)
            document_text = "\n".join(excerpt_parts).strip()
            if not document_text:
                continue
            try:
                extraction = extractor.extract(document_text)
                source_id = f"document:{source}"
                entity_count, relation_count = extractor.persist(
                    extraction,
                    graph_db_path,
                    source_id,
                    quote_is_safe=quote_is_safe,
                    source_kind="document",
                )
                documents_processed += 1
                entities_stored += entity_count
                relations_stored += relation_count
            except Exception:
                failures += 1
                logger.warning(
                    "Graph extraction failed for indexed document %s",
                    source,
                    exc_info=True,
                )
    except Exception:
        logger.warning("Document graph capture is unavailable", exc_info=True)
        return {"status": "skipped", "reason": "graph_storage_unavailable"}

    skipped = max(0, len(grouped) - len(selected))
    return {
        "status": "partial" if skipped or failures else "complete",
        "documents_processed": documents_processed,
        "documents_skipped": skipped,
        "entities_stored": entities_stored,
        "relations_stored": relations_stored,
        "failures": failures,
    }


@memory_router.post("/store")
def memory_store(req: MemoryStoreRequest, request: Request):
    """Store content in memory."""
    backend = _get_memory_backend(request)
    if backend is None:
        # Memory is intentionally disabled; report it honestly instead of a
        # 200 that silently discards the write (#502).
        raise HTTPException(status_code=503, detail="Memory is not configured")
    try:
        backend.store(req.content, metadata=req.metadata or {})
        return {"status": "stored"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@memory_router.post("/search")
def memory_search(req: MemorySearchRequest, request: Request):
    """Search memory for relevant content."""
    backend = _get_memory_backend(request)
    if backend is None:
        return {"results": []}
    try:
        results = backend.retrieve(req.query, top_k=req.top_k)
        items = [
            {
                "content": r.content,
                "score": getattr(r, "score", 0.0),
                "metadata": getattr(r, "metadata", {}),
            }
            for r in results
        ]
        return {"results": items}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@memory_router.get("/stats")
def memory_stats(request: Request):
    """Get memory backend statistics."""
    backend = _get_memory_backend(request)
    if backend is None:
        return {"entries": 0, "backend": "none", "status": "not_configured"}
    try:
        return {
            "entries": backend.count(),
            "backend": getattr(backend, "backend_id", "unknown"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@memory_router.get("/config")
async def memory_config(request: Request):
    """Return current memory configuration.

    Reports the built-in Python SQLite/FTS5 backend as available when the
    optional native extension is missing. If neither implementation can be
    constructed, the UI receives the actual backend error.
    """
    try:
        config = getattr(request.app.state, "config", None)
        if config is None:
            from openjarvis.core.config import load_config

            config = load_config()
        backend = getattr(request.app.state, "memory_backend", None)
        available = True
        detail: Optional[str] = None
        if backend is None:
            try:
                backend = _get_memory_backend(request)
            except HTTPException as exc:
                if exc.status_code != 503:
                    raise
                available = False
                detail = str(exc.detail)
        return {
            "backend_type": (
                backend.backend_id
                if backend is not None
                else config.memory.default_backend
            ),
            "available": available,
            "detail": detail,
            "context_top_k": config.memory.context_top_k,
            "context_min_score": config.memory.context_min_score,
            "context_max_tokens": config.memory.context_max_tokens,
            "context_from_memory": config.agent.context_from_memory,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@memory_router.post("/index")
def memory_index(req: MemoryIndexRequest, request: Request):
    """Index files from a path into memory."""
    try:
        import os
        from pathlib import Path

        from openjarvis.security.file_policy import is_sensitive_file
        from openjarvis.tools.storage.ingest import ingest_path
        from openjarvis.tools.storage.media import (
            MediaProcessingError,
            MediaTranscriptionUnavailableError,
        )
        from openjarvis.tools.storage.ocr import (
            OcrProcessingError,
            OcrUnavailableError,
        )

        target = Path(req.path).expanduser().resolve()
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")

        # Sandbox: when workspace roots are configured via OPENJARVIS_WORKSPACE
        # (os.pathsep-separated), only allow indexing inside them. This endpoint
        # must not become an arbitrary-filesystem read primitive over the API.
        workspace = os.environ.get("OPENJARVIS_WORKSPACE", "").strip()
        if workspace:
            roots = [
                Path(d).expanduser().resolve()
                for d in workspace.split(os.pathsep)
                if d.strip()
            ]
            if not any(target == root or root in target.parents for root in roots):
                raise HTTPException(
                    status_code=403,
                    detail="Path is outside the allowed workspace directories.",
                )
        # Never ingest sensitive files (.env, private keys, credentials, ...).
        if target.is_file() and is_sensitive_file(target):
            raise HTTPException(
                status_code=403, detail="Refusing to index a sensitive file."
            )

        backend = _get_memory_backend(request)
        if backend is None:
            raise HTTPException(status_code=503, detail="Memory is not configured")

        try:
            ingest_options = {"ocr": req.ocr}
            if req.transcribe_media:
                config = getattr(request.app.state, "config", None)
                if config is None:
                    from openjarvis.core.config import load_config

                    config = load_config()
                ingest_options.update(
                    transcribe_media=True,
                    speech_config=getattr(config, "speech", None),
                )
            chunks = ingest_path(target, **ingest_options)
        except OcrUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except MediaTranscriptionUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (OcrProcessingError, MediaProcessingError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        stored = 0
        for chunk in chunks:
            metadata = {"source": getattr(chunk, "source", str(target))}
            if hasattr(chunk, "metadata") and chunk.metadata:
                metadata.update(chunk.metadata)
            backend.store(chunk.content, metadata=metadata)
            stored += 1

        result = {
            "status": "indexed",
            "chunks_indexed": stored,
            "ocr_requested": req.ocr,
            "transcription_requested": req.transcribe_media,
        }
        if stored:
            config = getattr(request.app.state, "config", None)
            if config is None:
                from openjarvis.core.config import load_config

                config = load_config()
            result["graph_capture"] = _capture_indexed_documents(
                request, config, chunks
            )
        if stored == 0:
            # "indexed" must never silently mean "stored nothing". Surface why
            # so a folder of short notes doesn't look like a successful no-op
            # (#502 follow-up).
            result["note"] = (
                "no content was indexed — the path contained no readable "
                "documents with indexable text"
            )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- Traces routes ----

traces_router = APIRouter(prefix="/v1/traces", tags=["traces"])


def _serialise_trace(trace) -> dict:
    """Convert a Trace dataclass to a frontend-friendly dict."""
    import datetime
    from dataclasses import asdict

    d = asdict(trace)
    d["id"] = d.pop("trace_id", "")
    started = d.pop("started_at", 0.0)
    d["created_at"] = (
        datetime.datetime.fromtimestamp(started, tz=datetime.timezone.utc).isoformat()
        if started
        else None
    )
    dur = d.pop("total_latency_seconds", 0.0)
    d["duration_ms"] = round(dur * 1000)
    for step in d.get("steps", []):
        st = step.get("step_type")
        if hasattr(st, "value"):
            step["step_type"] = st.value
    return d


@traces_router.get("")
async def list_traces(request: Request, limit: int = 20):
    """List recent traces."""
    try:
        store = getattr(request.app.state, "trace_store", None)
        if store is None:
            return {"traces": []}
        traces = store.list_traces(limit=limit)
        items = [_serialise_trace(t) for t in traces]
        return {"traces": items}
    except Exception as exc:
        return {"traces": [], "error": str(exc)}


@traces_router.get("/{trace_id}")
async def get_trace(trace_id: str, request: Request):
    """Get a specific trace by ID."""
    try:
        store = getattr(request.app.state, "trace_store", None)
        if store is None:
            raise HTTPException(status_code=404, detail="Trace not found")
        trace = store.get(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="Trace not found")
        return _serialise_trace(trace)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- Telemetry routes ----

telemetry_router = APIRouter(prefix="/v1/telemetry", tags=["telemetry"])


def _telemetry_db_path(request: Request) -> Path:
    """Resolve telemetry storage from the active app configuration.

    ``DEFAULT_CONFIG_DIR`` is fixed when :mod:`openjarvis.core.config` is
    imported, so it cannot honor a later ``OPENJARVIS_HOME`` override.  The
    running app's config is authoritative; lightweight apps that include these
    routes directly fall back to the env-aware path resolver.
    """
    config = getattr(request.app.state, "config", None)
    telemetry = getattr(config, "telemetry", None)
    configured_path = getattr(telemetry, "db_path", None)
    if configured_path:
        return Path(configured_path).expanduser()

    from openjarvis.core.paths import get_config_dir

    return get_config_dir() / "telemetry.db"


@telemetry_router.get("/stats")
async def telemetry_stats(request: Request):
    """Get aggregated telemetry statistics."""
    try:
        from dataclasses import asdict

        from openjarvis.telemetry.aggregator import TelemetryAggregator

        db_path = _telemetry_db_path(request)
        if not db_path.exists():
            return {"total_requests": 0, "total_tokens": 0}

        session_start = getattr(request.app.state, "session_start", None)
        agg = TelemetryAggregator(db_path)
        try:
            stats = agg.summary(since=session_start)
            d = asdict(stats)
            d.pop("per_model", None)
            d.pop("per_engine", None)
            d["total_requests"] = d.pop("total_calls", 0)
            return d
        finally:
            agg.close()
    except Exception as exc:
        return {"error": str(exc)}


@telemetry_router.get("/energy")
async def telemetry_energy(request: Request):
    """Get energy monitoring data."""
    try:
        from openjarvis.telemetry.aggregator import TelemetryAggregator

        db_path = _telemetry_db_path(request)
        if not db_path.exists():
            return {
                "total_energy_j": 0,
                "energy_per_token_j": 0,
                "avg_power_w": 0,
                "cpu_temp_c": None,
                "gpu_temp_c": None,
            }

        session_start = getattr(request.app.state, "session_start", None)
        agg = TelemetryAggregator(db_path)
        try:
            stats = agg.summary(since=session_start)
            total_energy = stats.total_energy_joules
            total_tokens = stats.total_tokens
            total_latency = stats.total_latency
            return {
                "total_energy_j": total_energy,
                "energy_per_token_j": (
                    total_energy / total_tokens if total_tokens > 0 else 0
                ),
                "avg_power_w": (
                    total_energy / total_latency if total_latency > 0 else 0
                ),
                "cpu_temp_c": None,
                "gpu_temp_c": None,
            }
        finally:
            agg.close()
    except Exception as exc:
        return {"error": str(exc)}


# ---- Skills routes ----

skills_router = APIRouter(prefix="/v1/skills", tags=["skills"])


@skills_router.get("")
async def list_skills(request: Request):
    """List installed skills."""
    try:
        from openjarvis.core.registry import SkillRegistry

        skills = []
        for key in sorted(SkillRegistry.keys()):
            skills.append({"name": key})
        return {"skills": skills}
    except Exception as exc:
        logger.warning("Failed to list skills: %s", exc)
        return {"skills": []}


@skills_router.post("")
async def install_skill(request: Request):
    """Install a skill (placeholder)."""
    return {
        "status": "not_implemented",
        "message": "Use TOML files in ~/.openjarvis/skills/",
    }


@skills_router.delete("/{skill_name}")
async def remove_skill(skill_name: str, request: Request):
    """Remove a skill (placeholder)."""
    return {
        "status": "not_implemented",
        "message": "Skill removal not yet supported via API",
    }


# ---- Sessions routes ----

sessions_router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@sessions_router.get("")
async def list_sessions(request: Request, limit: int = 20):
    """List active sessions."""
    try:
        from openjarvis.sessions.store import SessionStore

        store = SessionStore()
        sessions = store.recent(limit=limit)
        items = [s.to_dict() if hasattr(s, "to_dict") else str(s) for s in sessions]
        return {"sessions": items}
    except Exception as exc:
        return {"sessions": [], "error": str(exc)}


@sessions_router.get("/{session_id}")
async def get_session(session_id: str, request: Request):
    """Get a specific session."""
    try:
        from openjarvis.sessions.store import SessionStore

        store = SessionStore()
        session = store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.to_dict() if hasattr(session, "to_dict") else {"id": session_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- Budget routes ----

budget_router = APIRouter(prefix="/v1/budget", tags=["budget"])

_budget_limits: Dict[str, Any] = {
    "max_tokens_per_day": None,
    "max_requests_per_hour": None,
}
_budget_usage: Dict[str, int] = {
    "tokens_today": 0,
    "requests_this_hour": 0,
}


@budget_router.get("")
async def get_budget(request: Request):
    """Get current budget usage and limits."""
    return {"limits": _budget_limits, "usage": _budget_usage}


@budget_router.put("/limits")
async def set_budget_limits(req: BudgetLimitsRequest, request: Request):
    """Update budget limits."""
    if req.max_tokens_per_day is not None:
        _budget_limits["max_tokens_per_day"] = req.max_tokens_per_day
    if req.max_requests_per_hour is not None:
        _budget_limits["max_requests_per_hour"] = req.max_requests_per_hour
    return {"status": "updated", "limits": _budget_limits}


# ---- Prometheus metrics ----

metrics_router = APIRouter(tags=["metrics"])


@metrics_router.get("/metrics")
async def prometheus_metrics(request: Request):
    """Prometheus-compatible metrics endpoint."""
    try:
        from openjarvis.telemetry.aggregator import TelemetryAggregator

        db_path = _telemetry_db_path(request)
        if not db_path.exists():
            from starlette.responses import PlainTextResponse

            return PlainTextResponse("# no telemetry data\n", media_type="text/plain")

        agg = TelemetryAggregator(db_path)
        try:
            stats = agg.summary()
        finally:
            agg.close()

        avg_latency_ms = (
            (stats.total_latency / stats.total_calls) * 1000
            if stats.total_calls
            else 0.0
        )

        lines = [
            "# HELP openjarvis_requests_total Total requests processed",
            "# TYPE openjarvis_requests_total counter",
            f"openjarvis_requests_total {stats.total_calls}",
            "# HELP openjarvis_tokens_total Total tokens generated",
            "# TYPE openjarvis_tokens_total counter",
            f"openjarvis_tokens_total {stats.total_tokens}",
            "# HELP openjarvis_latency_avg_ms Average latency in milliseconds",
            "# TYPE openjarvis_latency_avg_ms gauge",
            f"openjarvis_latency_avg_ms {avg_latency_ms}",
        ]
        from starlette.responses import PlainTextResponse

        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")
    except Exception as exc:
        logger.warning("Failed to collect Prometheus metrics: %s", exc)
        from starlette.responses import PlainTextResponse

        return PlainTextResponse("# No metrics available\n", media_type="text/plain")


# ---- WebSocket streaming routes ----

websocket_router = APIRouter(tags=["websocket"])
_SYNC_STREAM_END = object()


async def _next_sync_stream(iterator: Any) -> Any:
    """Fetch one sync-stream item without blocking or racing cancellation."""
    next_task = asyncio.create_task(asyncio.to_thread(next, iterator, _SYNC_STREAM_END))
    try:
        return await asyncio.shield(next_task)
    except asyncio.CancelledError:
        # ``to_thread`` cannot stop a running ``next()``. Wait for it before
        # allowing the iterator to be closed so cancellation cannot race a
        # generator that is still executing in the worker thread.
        try:
            await next_task
        except Exception:
            pass
        raise


async def _iterate_sync_stream(iterator: Any):
    """Adapt a blocking iterator to an async generator, closing it safely."""
    try:
        while True:
            item = await _next_sync_stream(iterator)
            if item is _SYNC_STREAM_END:
                return
            yield item
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            try:
                await asyncio.to_thread(close)
            except Exception as exc:
                logger.warning("Failed to close synchronous engine stream: %s", exc)


def _record_ws_trace(
    trace_store,
    *,
    query: str,
    result: str,
    model: str,
    started_at: float,
    ended_at: float,
) -> None:
    """Record a trace for a completed WebSocket chat (best-effort)."""
    if trace_store is None or not result:
        return
    from openjarvis.traces.collector import record_response_trace

    record_response_trace(
        trace_store,
        query=query,
        result=result,
        model=model,
        started_at=started_at,
        ended_at=ended_at,
    )


@websocket_router.websocket("/v1/chat/stream")
async def websocket_chat_stream(websocket: WebSocket):
    """Stream chat responses over a WebSocket connection.

    Accepts JSON messages of the form::

        {"message": "...", "model": "...", "agent": "..."}

    Sends back JSON chunks::

        {"type": "chunk", "content": "..."}   -- per-token streaming
        {"type": "done",  "content": "..."}   -- final assembled response
        {"type": "error", "detail": "..."}    -- on failure
    """
    from openjarvis.server.auth_middleware import authenticate_websocket

    expected_key = getattr(websocket.app.state, "api_key", "")
    authorized, subprotocol = authenticate_websocket(websocket, expected_key)
    if not authorized:
        # Closing before accept rejects the HTTP upgrade request.
        await websocket.close(code=1008)
        return
    await websocket.accept(subprotocol=subprotocol)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                await websocket.send_json(
                    {"type": "error", "detail": "Invalid JSON"},
                )
                continue

            message = data.get("message")
            if not message:
                await websocket.send_json(
                    {"type": "error", "detail": "Missing 'message' field"},
                )
                continue

            model = data.get("model") or getattr(
                websocket.app.state,
                "model",
                "default",
            )
            engine = getattr(websocket.app.state, "engine", None)
            if engine is None:
                await websocket.send_json(
                    {"type": "error", "detail": "No engine configured"},
                )
                continue

            messages = [{"role": "user", "content": message}]

            # This WS path streams straight from the engine (no agent /
            # TraceCollector), so record the interaction directly once it
            # finishes — otherwise WebSocket chats never reach traces.db.
            import time as _time

            trace_store = getattr(websocket.app.state, "trace_store", None)
            _ws_started_at = _time.time()

            try:
                # Prefer streaming if the engine supports it
                stream_fn = getattr(engine, "stream", None)
                if stream_fn is not None and (
                    inspect.isasyncgenfunction(stream_fn) or callable(stream_fn)
                ):
                    full_content = ""
                    try:
                        gen = stream_fn(messages, model=model)
                        # Handle both async and sync generators
                        if inspect.isasyncgen(gen):
                            async for token in gen:
                                full_content += token
                                await websocket.send_json(
                                    {"type": "chunk", "content": token},
                                )
                        else:
                            # Each ``next()`` can perform a blocking upstream
                            # read, so offload iteration one item at a time.
                            async for token in _iterate_sync_stream(iter(gen)):
                                full_content += token
                                await websocket.send_json(
                                    {"type": "chunk", "content": token},
                                )
                    except TypeError:
                        # stream() didn't return an iterable; fall back to
                        # generate(). It makes a blocking upstream call, so run
                        # it in a worker thread to keep the event loop free.
                        result = await asyncio.to_thread(
                            engine.generate, messages, model=model
                        )
                        content = (
                            result.get("content", "")
                            if isinstance(
                                result,
                                dict,
                            )
                            else str(result)
                        )
                        full_content = content
                        await websocket.send_json(
                            {"type": "chunk", "content": content},
                        )
                    await websocket.send_json(
                        {"type": "done", "content": full_content},
                    )
                    _record_ws_trace(
                        trace_store,
                        query=message,
                        result=full_content,
                        model=model,
                        started_at=_ws_started_at,
                        ended_at=_time.time(),
                    )
                else:
                    # No stream method — single-shot generate. Blocking upstream
                    # call, so run in a worker thread to keep the event loop free.
                    result = await asyncio.to_thread(
                        engine.generate, messages, model=model
                    )
                    content = (
                        result.get("content", "")
                        if isinstance(
                            result,
                            dict,
                        )
                        else str(result)
                    )
                    await websocket.send_json(
                        {"type": "chunk", "content": content},
                    )
                    await websocket.send_json(
                        {"type": "done", "content": content},
                    )
                    _record_ws_trace(
                        trace_store,
                        query=message,
                        result=content,
                        model=model,
                        started_at=_ws_started_at,
                        ended_at=_time.time(),
                    )
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                await websocket.send_json(
                    {"type": "error", "detail": str(exc)},
                )
    except WebSocketDisconnect:
        pass  # Client disconnected — nothing to clean up


# ---- Learning routes ----

learning_router = APIRouter(prefix="/v1/learning", tags=["learning"])


@learning_router.get("/stats")
async def learning_stats(request: Request):
    """Return learning system statistics across all sub-policies."""
    result: Dict[str, Any] = {}

    # Skill discovery
    try:
        from openjarvis.learning.agents.skill_discovery import SkillDiscovery

        discovery = SkillDiscovery()
        result["skill_discovery"] = {
            "available": True,
            "discovered_count": len(discovery.discovered_skills),
        }
    except Exception as exc:
        logger.warning("Failed to load skill discovery stats: %s", exc)
        result["skill_discovery"] = {"available": False}

    return result


@learning_router.get("/policy")
async def learning_policy(request: Request):
    """Return current routing policy configuration."""
    result: Dict[str, Any] = {}

    # Load config and extract learning section
    try:
        from openjarvis.core.config import load_config

        config = load_config()
        lc = config.learning
        result["enabled"] = lc.enabled
        result["update_interval"] = lc.update_interval
        result["auto_update"] = lc.auto_update
        result["routing"] = {
            "policy": lc.routing.policy,
            "min_samples": lc.routing.min_samples,
        }
        result["intelligence"] = {
            "policy": lc.intelligence.policy,
        }
        result["agent"] = {
            "policy": lc.agent.policy,
        }
        result["metrics"] = {
            "accuracy_weight": lc.metrics.accuracy_weight,
            "latency_weight": lc.metrics.latency_weight,
            "cost_weight": lc.metrics.cost_weight,
            "efficiency_weight": lc.metrics.efficiency_weight,
        }
    except Exception as exc:
        logger.warning("Failed to load learning config: %s", exc)
        result["enabled"] = False
        result["routing"] = {"policy": "heuristic", "min_samples": 5}
        result["intelligence"] = {"policy": "none"}
        result["agent"] = {"policy": "none"}
        result["metrics"] = {}

    return result


# ---- Speech routes ----

speech_router = APIRouter(prefix="/v1/speech", tags=["speech"])


@speech_router.post("/transcribe")
async def transcribe_speech(request: Request):
    """Transcribe uploaded audio to text."""
    backend = getattr(request.app.state, "speech_backend", None)
    if backend is None:
        raise HTTPException(status_code=501, detail="Speech backend not configured")

    form = await request.form()
    audio_file = form.get("file")
    if audio_file is None:
        raise HTTPException(status_code=400, detail="Missing 'file' field")

    audio_bytes = await audio_file.read()
    language = form.get("language")

    # Detect format from filename
    filename = getattr(audio_file, "filename", "audio.wav")
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "wav"

    try:
        result = await asyncio.to_thread(
            backend.transcribe,
            audio_bytes,
            format=ext,
            language=language or None,
        )
    except Exception as exc:
        logger.exception("Speech transcription failed")
        raise HTTPException(
            status_code=500,
            detail=f"Speech transcription failed: {exc}",
        ) from exc

    return {
        "text": result.text,
        "language": result.language,
        "confidence": result.confidence,
        "duration_seconds": result.duration_seconds,
    }


@speech_router.get("/health")
async def speech_health(request: Request):
    """Check if a speech backend is available."""
    backend = getattr(request.app.state, "speech_backend", None)
    if backend is None:
        return {"available": False, "reason": "No speech backend configured"}
    try:
        available = backend.health()
        reason = None
    except Exception as exc:
        logger.exception("Speech health check failed")
        available = False
        reason = str(exc)

    if not available and reason is None:
        last_error = getattr(backend, "last_error", None)
        if callable(last_error):
            reason = last_error()

    return {
        "available": available,
        "backend": backend.backend_id,
        **({"reason": reason} if reason else {}),
    }


# Synthesis is CPU-bound and runs in a worker thread; cap the input so one
# request cannot occupy that thread for minutes.
_MAX_TTS_CHARS = 5000


async def _resolve_tts_backend(request: Request):
    """Return a healthy TTS backend, resolved once and cached on app state.

    Backends load models on construction, so this stays lazy: a server whose
    users never ask for audio never pays for it.
    """
    app = request.app
    if getattr(app.state, "tts_resolved", False):
        return getattr(app.state, "tts_backend", None)

    # The first health probe can load Kokoro's model. A shared task keeps that
    # work off the event loop and coalesces concurrent probes, including when
    # the backend is unavailable. A later request may retry that failure.
    task = getattr(app.state, "tts_resolution_task", None)
    if task is None:

        def discover():
            from openjarvis.speech._tts_discovery import (
                get_tts_backend,
                voice_preferences,
            )

            config = getattr(app.state, "config", None)
            if config is None:
                from openjarvis.core.config import load_config

                config = load_config()

            preferred, _, _ = voice_preferences(config)
            speech_config = getattr(config, "speech", None)
            return get_tts_backend(
                preferred,
                offline_only=bool(getattr(speech_config, "offline_only", False)),
            )

        task = asyncio.create_task(asyncio.to_thread(discover))
        app.state.tts_resolution_task = task

    try:
        backend = await asyncio.shield(task)
    finally:
        if task.done() and getattr(app.state, "tts_resolution_task", None) is task:
            app.state.tts_resolution_task = None

    if backend is not None:
        app.state.tts_backend = backend
        app.state.tts_resolved = True
    return backend


def _tts_voice_and_speed(request: Request, backend) -> tuple[str, float]:
    """Resolve the voice ID and speed to use with *backend*.

    Voice IDs are not portable between backends, so a configured ``voice_id``
    applies only when the resolved backend is the configured one; otherwise the
    fallback backend's own default is used.
    """
    from openjarvis.speech._tts_discovery import default_voice_for, voice_preferences

    config = getattr(request.app.state, "config", None)
    if config is None:
        return "", 1.0

    preferred, voice_id, speed = voice_preferences(config)
    active = getattr(backend, "backend_id", "")
    if active and active != preferred:
        voice_id = default_voice_for(active)
    return voice_id, speed


@speech_router.post("/synthesize")
async def synthesize_speech(request: Request, body: SpeechSynthesizeRequest):
    """Synthesize text to speech and return the audio as WAV."""
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text'")
    if len(text) > _MAX_TTS_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Text exceeds {_MAX_TTS_CHARS} characters",
        )

    backend = await _resolve_tts_backend(request)
    if backend is None:
        raise HTTPException(
            status_code=501, detail="No text-to-speech backend available"
        )

    voice_id, speed = _tts_voice_and_speed(request, backend)
    if body.voice_id:
        voice_id = body.voice_id
    if body.speed is not None:
        speed = body.speed

    try:
        result = await asyncio.to_thread(
            backend.synthesize,
            text,
            voice_id=voice_id,
            speed=speed,
            output_format="wav",
        )
    except Exception as exc:
        logger.exception("Speech synthesis failed")
        raise HTTPException(status_code=500, detail="Speech synthesis failed") from exc

    return Response(
        content=result.audio,
        media_type="audio/wav",
        headers={
            "X-Voice-Id": result.voice_id or voice_id,
            "X-Sample-Rate": str(result.sample_rate),
        },
    )


@speech_router.get("/tts/health")
async def tts_health(request: Request):
    """Report whether voice output is available, and with which voice."""
    backend = await _resolve_tts_backend(request)
    if backend is None:
        return {"available": False, "reason": "No text-to-speech backend available"}

    voice_id, speed = _tts_voice_and_speed(request, backend)
    return {
        "available": True,
        "backend": getattr(backend, "backend_id", ""),
        "voice_id": voice_id,
        "speed": speed,
    }


# ---- Feedback routes ----

feedback_router = APIRouter(prefix="/v1/feedback", tags=["feedback"])


@feedback_router.post("")
async def submit_feedback(req: FeedbackScoreRequest, request: Request):
    """Submit feedback for a trace."""
    try:
        from openjarvis.core.config import DEFAULT_CONFIG_DIR
        from openjarvis.traces.store import TraceStore

        db_path = DEFAULT_CONFIG_DIR / "traces.db"
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="No trace database")

        store = TraceStore(db_path)
        updated = store.update_feedback(req.trace_id, req.score)
        store.close()

        if not updated:
            raise HTTPException(
                status_code=404, detail=f"Trace '{req.trace_id}' not found"
            )
        return {"status": "recorded", "trace_id": req.trace_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@feedback_router.get("/stats")
async def feedback_stats(request: Request):
    """Get feedback statistics."""
    return {"total": 0, "mean_score": 0.0}


# ---- Optimize routes ----

optimize_router = APIRouter(prefix="/v1/optimize", tags=["optimize"])


@optimize_router.get("/runs")
async def list_optimize_runs(request: Request):
    """List optimization runs."""
    try:
        from openjarvis.core.config import DEFAULT_CONFIG_DIR
        from openjarvis.learning.optimize.store import OptimizationStore

        db_path = DEFAULT_CONFIG_DIR / "optimize.db"
        if not db_path.exists():
            return {"runs": []}

        store = OptimizationStore(db_path)
        runs = store.list_runs()
        store.close()
        return {"runs": runs}
    except Exception as exc:
        logger.warning("Failed to list optimization runs: %s", exc)
        return {"runs": []}


@optimize_router.get("/runs/{run_id}")
async def get_optimize_run(run_id: str, request: Request):
    """Get optimization run details."""
    try:
        from openjarvis.core.config import DEFAULT_CONFIG_DIR
        from openjarvis.learning.optimize.store import OptimizationStore

        db_path = DEFAULT_CONFIG_DIR / "optimize.db"
        if not db_path.exists():
            return {"run_id": run_id, "status": "not_found"}

        store = OptimizationStore(db_path)
        run = store.get_run(run_id)
        store.close()

        if run is None:
            return {"run_id": run_id, "status": "not_found"}

        return {
            "run_id": run.run_id,
            "status": run.status,
            "benchmark": run.benchmark,
            "trials": len(run.trials),
            "best_trial_id": (run.best_trial.trial_id if run.best_trial else None),
        }
    except Exception as exc:
        logger.warning("Failed to get optimization run %s: %s", run_id, exc)
        return {"run_id": run_id, "status": "not_found"}


@optimize_router.post("/runs")
async def start_optimize_run(req: OptimizeRunRequest, request: Request):
    """Start a new optimization run."""
    return {"status": "started", "run_id": "placeholder"}


def include_all_routes(app) -> None:
    """Include all extended API routers in a FastAPI app."""
    from openjarvis.server.approval_routes import (
        router as approval_router,  # noqa: PLC0415
    )

    app.include_router(approval_router)
    app.include_router(agents_router)
    app.include_router(memory_router)
    app.include_router(traces_router)
    app.include_router(telemetry_router)
    app.include_router(skills_router)
    app.include_router(sessions_router)
    app.include_router(budget_router)
    app.include_router(metrics_router)
    app.include_router(websocket_router)
    app.include_router(learning_router)
    app.include_router(speech_router)
    app.include_router(feedback_router)
    app.include_router(optimize_router)

    # Agent Manager routes (if available)
    try:
        if hasattr(app.state, "agent_manager") and app.state.agent_manager:
            from openjarvis.server.agent_manager_routes import (  # noqa: PLC0415
                create_agent_manager_router,
            )

            (
                agents_r,
                templates_r,
                global_r,
                tools_r,
                sendblue_r,
            ) = create_agent_manager_router(app.state.agent_manager)
            app.include_router(agents_r)
            app.include_router(templates_r)
            app.include_router(global_r)
            app.include_router(tools_r)
            app.include_router(sendblue_r)
    except ImportError:
        pass

    # WebSocket bridge for real-time agent events. Must subscribe on the
    # same EventBus instance channels/agents actually publish to
    # (app.state.bus, set in server/app.py) — the get_event_bus() global
    # singleton is a *different* bus that nothing in `jarvis serve` ever
    # publishes to, so events silently never reached this endpoint.
    try:
        from openjarvis.core.events import get_event_bus
        from openjarvis.server.ws_bridge import create_ws_router

        ws_router = create_ws_router(getattr(app.state, "bus", None) or get_event_bus())
        app.include_router(ws_router)
    except Exception:
        logger.debug("WebSocket bridge not available", exc_info=True)


__all__ = [
    "include_all_routes",
    "agents_router",
    "memory_router",
    "traces_router",
    "telemetry_router",
    "skills_router",
    "sessions_router",
    "budget_router",
    "metrics_router",
    "websocket_router",
    "learning_router",
    "speech_router",
    "feedback_router",
    "optimize_router",
]
