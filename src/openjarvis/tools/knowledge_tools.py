"""Tools for querying and maintaining the provenance-aware knowledge graph."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


class _GraphTool(BaseTool):
    def __init__(
        self, backend: Optional[Any] = None, db_path: str | None = None
    ) -> None:
        self._backend = backend
        self._db_path = db_path

    @contextmanager
    def _graph(self) -> Iterator[Any]:
        if self._backend is not None:
            yield self._backend
            return
        if not self._db_path:
            raise RuntimeError("No knowledge graph backend available")
        from openjarvis.tools.storage.knowledge_graph import KnowledgeGraphMemory

        backend = KnowledgeGraphMemory(db_path=self._db_path)
        try:
            yield backend
        finally:
            backend.close()


@ToolRegistry.register("kg_add_entity")
class KGAddEntityTool(_GraphTool):
    """Add or update a graph entity with provenance."""

    tool_id = "kg_add_entity"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description=(
                "Add or update an entity in the local knowledge graph, including "
                "its provenance."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "entity_type": {"type": "string"},
                    "name": {"type": "string"},
                    "properties": {"type": "object"},
                    "source": {"type": "string"},
                    "source_url": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "status": {
                        "type": "string",
                        "enum": [
                            "reported",
                            "hypothesis",
                            "verified",
                            "contradicted",
                            "superseded",
                        ],
                    },
                },
                "required": ["entity_id", "entity_type", "name"],
            },
            category="knowledge_graph",
            required_capabilities=["memory:write"],
        )

    def execute(self, **params: Any) -> ToolResult:
        from openjarvis.tools.storage.knowledge_graph import Entity

        try:
            with self._graph() as graph:
                graph.add_entity(
                    Entity(
                        entity_id=params["entity_id"],
                        entity_type=params["entity_type"],
                        name=params["name"],
                        properties=params.get("properties", {}),
                        source=params.get("source", ""),
                        source_url=params.get("source_url", ""),
                        confidence=params.get("confidence", 1.0),
                        status=params.get("status", "reported"),
                    )
                )
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Entity '{params['name']}' saved.",
            )
        except Exception as exc:
            return ToolResult(tool_name=self.tool_id, content=str(exc), success=False)


@ToolRegistry.register("kg_add_relation")
class KGAddRelationTool(_GraphTool):
    """Add an evidence-backed relation between two existing entities."""

    tool_id = "kg_add_relation"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description=(
                "Add a relation between existing graph entities and attach its "
                "source and supporting quote."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "weight": {"type": "number"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "status": {
                        "type": "string",
                        "enum": [
                            "reported",
                            "hypothesis",
                            "verified",
                            "contradicted",
                            "superseded",
                        ],
                    },
                    "source": {"type": "string"},
                    "source_url": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                    "valid_from": {
                        "type": "string",
                        "description": (
                            "Inclusive ISO date (YYYY-MM-DD) when the source "
                            "claim starts to apply."
                        ),
                    },
                    "valid_until": {
                        "type": "string",
                        "description": (
                            "Inclusive ISO date (YYYY-MM-DD) when the source "
                            "claim stops applying."
                        ),
                    },
                    "properties": {"type": "object"},
                },
                "required": ["source_id", "target_id", "relation_type"],
            },
            category="knowledge_graph",
            required_capabilities=["memory:write"],
        )

    def execute(self, **params: Any) -> ToolResult:
        from openjarvis.tools.storage.knowledge_graph import Relation

        confidence = params.get("confidence", 0.5)
        source = params.get("source", "")
        source_url = params.get("source_url", "")
        quote = params.get("evidence_quote", "")
        try:
            with self._graph() as graph:
                graph.add_relation(
                    Relation(
                        source_id=params["source_id"],
                        target_id=params["target_id"],
                        relation_type=params["relation_type"],
                        weight=params.get("weight", 1.0),
                        properties=params.get("properties", {}),
                        confidence=confidence,
                        status=params.get("status", "reported"),
                        source=source,
                        source_url=source_url,
                        evidence=(
                            [
                                {
                                    "source": source,
                                    "source_url": source_url,
                                    "quote": quote,
                                    "confidence": confidence,
                                    "status": params.get("status", "reported"),
                                    "valid_from": params.get("valid_from", ""),
                                    "valid_until": params.get("valid_until", ""),
                                }
                            ]
                            if (
                                quote
                                or source_url
                                or params.get("valid_from")
                                or params.get("valid_until")
                            )
                            else []
                        ),
                    )
                )
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Relation '{params['relation_type']}' saved with provenance.",
            )
        except Exception as exc:
            return ToolResult(tool_name=self.tool_id, content=str(exc), success=False)


@ToolRegistry.register("kg_query")
class KGQueryTool(_GraphTool):
    """Search graph entities and relations by text or type."""

    tool_id = "kg_query"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description=(
                "Search entities and relations in the local graph. Results "
                "include evidence and source URLs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to match against entity names/properties.",
                    },
                    "entity_type": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
            },
            category="knowledge_graph",
            required_capabilities=["memory:read"],
        )

    def execute(self, **params: Any) -> ToolResult:
        try:
            with self._graph() as graph:
                query = str(params.get("query", "")).strip()
                if query:
                    result = graph.search_graph(
                        query, top_k=params.get("limit", 20), depth=1
                    )
                    if params.get("entity_type"):
                        result.entities = [
                            entity
                            for entity in result.entities
                            if entity.entity_type.casefold()
                            == str(params["entity_type"]).casefold()
                        ]
                    if params.get("relation_type"):
                        result.relations = [
                            relation
                            for relation in result.relations
                            if relation.relation_type.casefold()
                            == str(params["relation_type"]).casefold()
                        ]
                else:
                    result = graph.query_pattern(
                        entity_type=params.get("entity_type"),
                        relation_type=params.get("relation_type"),
                        limit=params.get("limit", 50),
                    )
                output = {
                    "entities": [
                        {
                            "id": entity.entity_id,
                            "type": entity.entity_type,
                            "name": entity.name,
                            "source": entity.source,
                            "source_url": entity.source_url,
                            "confidence": entity.confidence,
                            "status": entity.status,
                        }
                        for entity in result.entities
                    ],
                    "relations": [
                        {
                            "id": relation.relation_id,
                            "source": relation.source_id,
                            "target": relation.target_id,
                            "type": relation.relation_type,
                            "weight": relation.weight,
                            "confidence": relation.confidence,
                            "status": relation.status,
                            "evidence": relation.evidence,
                        }
                        for relation in result.relations
                    ],
                }
            return ToolResult(
                tool_name=self.tool_id,
                content=json.dumps(output, indent=2, ensure_ascii=False),
            )
        except Exception as exc:
            return ToolResult(tool_name=self.tool_id, content=str(exc), success=False)


@ToolRegistry.register("kg_neighbors")
class KGNeighborsTool(_GraphTool):
    """Find adjacent entities with supporting relation evidence."""

    tool_id = "kg_neighbors"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description=(
                "Find entities connected to a graph entity, including edge "
                "direction, status, and evidence."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "direction": {"type": "string", "enum": ["in", "out", "both"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["entity_id"],
            },
            category="knowledge_graph",
            required_capabilities=["memory:read"],
        )

    def execute(self, **params: Any) -> ToolResult:
        try:
            with self._graph() as graph:
                center_id = params["entity_id"]
                relations = graph.relations_for_entity(
                    center_id,
                    relation_type=params.get("relation_type"),
                    direction=params.get("direction", "both"),
                    limit=params.get("limit", 50),
                )
                neighbor_ids = {
                    r.target_id if r.source_id == center_id else r.source_id
                    for r in relations
                }
                output = {
                    "neighbors": [
                        {
                            "id": entity.entity_id,
                            "type": entity.entity_type,
                            "name": entity.name,
                            "source_url": entity.source_url,
                        }
                        for key in neighbor_ids
                        if (entity := graph.get_entity(key)) is not None
                    ],
                    "relations": [
                        {
                            "source": relation.source_id,
                            "target": relation.target_id,
                            "type": relation.relation_type,
                            "confidence": relation.confidence,
                            "status": relation.status,
                            "evidence": relation.evidence,
                        }
                        for relation in relations
                    ],
                }
            return ToolResult(
                tool_name=self.tool_id,
                content=json.dumps(output, indent=2, ensure_ascii=False),
            )
        except Exception as exc:
            return ToolResult(tool_name=self.tool_id, content=str(exc), success=False)


__all__ = [
    "KGAddEntityTool",
    "KGAddRelationTool",
    "KGNeighborsTool",
    "KGQueryTool",
]
