"""Tests for evidence-backed graph capture from ACE conversations."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from openjarvis.core.config import StorageConfig
from openjarvis.core.events import EventBus
from openjarvis.core.types import Message, Role
from openjarvis.engine.multi import MultiEngine
from openjarvis.memory.episodes import EpisodeStore
from openjarvis.memory.graph_extractor import (
    ConversationGraphExtractor,
    DocumentGraphExtractor,
)
from openjarvis.memory.service import MemoryService, build_memory_service
from openjarvis.memory.store import LocalFactStore
from openjarvis.telemetry.instrumented_engine import InstrumentedEngine
from openjarvis.tools.storage.context import ContextConfig, inject_context
from openjarvis.tools.storage.knowledge_graph import (
    Entity,
    KnowledgeGraphMemory,
    Relation,
    stable_entity_id,
)


class _FakeEngine:
    is_cloud = False
    is_local = True

    def __init__(self, payload, models=None):
        self.payload = payload
        self.calls = []
        self.models = list(models or [])

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"content": json.dumps(self.payload)}

    def list_models(self):
        return self.models


@pytest.mark.parametrize(
    "payload",
    [
        "",
        {},
        [{"entities": [], "relations": []}],
        {"entities": {}, "relations": []},
        {"entities": [{"name": "ACE"}], "relations": []},
    ],
)
def test_strict_graph_extraction_rejects_invalid_responses(payload):
    extractor = ConversationGraphExtractor(_FakeEngine(payload), "local-model")
    with pytest.raises(ValueError, match="invalid response"):
        extractor.extract("Private user text", strict=True)


@pytest.mark.parametrize(
    "extractor_class", [ConversationGraphExtractor, DocumentGraphExtractor]
)
def test_strict_graph_extraction_accepts_empty_arrays(extractor_class):
    extractor = extractor_class(
        _FakeEngine({"entities": [], "relations": []}), "local-model"
    )
    result = extractor.extract("No durable claims here.", strict=True)
    assert result.entities == result.relations == []


def test_strict_document_extraction_rejects_quotes_that_would_lose_evidence():
    long_quote = f"ACE uses {'documented ' * 60}OpenJarvis on 2026-05-04."
    extractor = DocumentGraphExtractor(
        _FakeEngine(
            {
                "entities": [
                    {"name": "ACE", "quote": "ACE"},
                    {"name": "OpenJarvis", "quote": "OpenJarvis"},
                    {"name": "2026-05-04", "quote": long_quote},
                ],
                "relations": [
                    {
                        "subject": "ACE",
                        "predicate": "uses",
                        "object": "OpenJarvis",
                        "quote": long_quote,
                        "valid_from": "2026-05-04",
                    }
                ],
            }
        ),
        "local-model",
    )
    extraction = extractor.extract(long_quote, strict=True)
    assert [entity.name for entity in extraction.entities] == ["ACE", "OpenJarvis"]
    assert extraction.relations == []


def _web_extraction():
    quote = "ACE uses OpenJarvis since 2026-05-04."
    extractor = DocumentGraphExtractor(
        _FakeEngine(
            {
                "entities": [
                    {"name": "ACE", "type": "project", "quote": "ACE"},
                    {"name": "OpenJarvis", "type": "software", "quote": "OpenJarvis"},
                ],
                "relations": [
                    {
                        "subject": "ACE",
                        "predicate": "uses",
                        "object": "OpenJarvis",
                        "quote": quote,
                        "valid_from": "2026-05-04",
                    }
                ],
            }
        ),
        "local-model",
    )
    return extractor, extractor.extract(quote, strict=True)


def test_web_graph_persists_url_and_distinct_capture_evidence(tmp_path):
    extractor, extraction = _web_extraction()
    path = tmp_path / "web-graph.db"
    url = "https://example.test/ace"
    ace_id = stable_entity_id("project", "ACE")
    jarvis_id = stable_entity_id("software", "OpenJarvis")
    with KnowledgeGraphMemory(path) as graph:
        graph.add_entity(Entity(ace_id, "project", "ACE", status="verified"))
        graph.add_entity(Entity(jarvis_id, "software", "OpenJarvis"))
        graph.add_relation(Relation(ace_id, jarvis_id, "uses", status="verified"))

    for capture_id, observed_at in (
        ("web:original", 1_761_000_000.0),
        ("web:original", 1_761_000_000.0),
        ("web:recapture", 1_762_000_000.0),
    ):
        assert extractor.persist(
            extraction,
            path,
            capture_id,
            source_kind="web",
            source_url=url,
            observed_at=observed_at,
        ) == (2, 1)

    with KnowledgeGraphMemory(path) as graph:
        assert graph.entity_count() == 2
        assert graph.relation_count() == 1
        assert graph.evidence_count() == 2
        ace = graph.get_entity(ace_id)
        assert ace.status == "verified"
        assert ace.source == "web"
        assert ace.source_url == url
        assert [item["capture_id"] for item in ace.properties["evidence_sources"]] == [
            "web:original",
            "web:recapture",
        ]
        relation = graph.query_pattern(relation_type="uses").relations[0]
        assert relation.status == "verified"
        assert relation.source == "web"
        assert relation.source_url == url
        assert [item["capture_id"] for item in relation.evidence] == [
            "web:recapture",
            "web:original",
        ]
        assert [item["observed_at"] for item in relation.evidence] == [
            1_762_000_000.0,
            1_761_000_000.0,
        ]
        for evidence in relation.evidence:
            assert evidence["source"] == "web"
            assert evidence["source_url"] == url
            assert evidence["quote"] == extraction.relations[0].quote
            assert evidence["valid_from"] == "2026-05-04"
            assert evidence["status"] == "reported"


def test_web_graph_rolls_back_batch_after_relation_write_fails(tmp_path, monkeypatch):
    extractor, extraction = _web_extraction()
    path = tmp_path / "web-graph.db"
    add_relation = KnowledgeGraphMemory.add_relation

    def fail_after_insert(graph, relation):
        add_relation(graph, relation)
        raise RuntimeError("simulated storage failure")

    monkeypatch.setattr(KnowledgeGraphMemory, "add_relation", fail_after_insert)
    with pytest.raises(RuntimeError, match="simulated storage failure"):
        extractor.persist(
            extraction,
            path,
            "web:original",
            source_kind="web",
            source_url="https://example.test/ace",
            observed_at=1_761_000_000.0,
        )

    with KnowledgeGraphMemory(path) as graph:
        assert graph.entity_count() == 0
        assert graph.relation_count() == 0
        assert graph.evidence_count() == 0
        assert graph.recent_events() == []


def test_strict_graph_extraction_suppresses_private_engine_diagnostics(caplog):
    secret = "Private engine diagnostics"

    class UnavailableEngine(_FakeEngine):
        def generate(self, *_args, **_kwargs):
            raise TimeoutError(secret)

    extractor = ConversationGraphExtractor(UnavailableEngine({}), "local-model")
    with caplog.at_level("DEBUG"):
        assert extractor.extract("Private user text").entities == []
        with pytest.raises(
            RuntimeError, match="Conversation graph inference failed"
        ) as error:
            extractor.extract("Private user text", strict=True)
    assert secret not in str(error.value)
    assert error.value.__suppress_context__
    assert secret not in caplog.text


def test_conversation_graph_requires_verbatim_user_evidence(tmp_path):
    user_text = "I am building ACE on OpenJarvis."
    engine = _FakeEngine(
        {
            "entities": [
                {
                    "name": "ACE",
                    "type": "project",
                    "quote": "building ACE",
                    "confidence": 0.9,
                },
                {
                    "name": "OpenJarvis",
                    "type": "software",
                    "quote": "on OpenJarvis",
                    "confidence": 0.9,
                },
                {
                    "name": "SQLite",
                    "type": "software",
                    "quote": "SQLite stores the graph",
                    "confidence": 1.0,
                },
            ],
            "relations": [
                {
                    "subject": "ACE",
                    "predicate": "built_on",
                    "object": "OpenJarvis",
                    "quote": "building ACE on OpenJarvis",
                    "confidence": 0.85,
                },
                {
                    "subject": "ACE",
                    "predicate": "uses",
                    "object": "SQLite",
                    "quote": "ACE uses SQLite",
                    "confidence": 0.99,
                },
            ],
        }
    )
    extractor = ConversationGraphExtractor(engine, "local-model")

    result = extractor.extract(user_text)

    assert extractor.extract(user_text, strict=True) == result
    assert [entity.name for entity in result.entities] == ["ACE", "OpenJarvis"]
    assert len(result.relations) == 1
    assert result.relations[0].quote == "building ACE on OpenJarvis"
    submitted_prompt = engine.calls[0][0][1].text
    assert user_text in submitted_prompt
    assert "Assistant:" not in submitted_prompt

    stored_entities, stored_relations = extractor.persist(
        result,
        tmp_path / "graph.db",
        "conversation:test:turn-1",
    )
    assert (stored_entities, stored_relations) == (2, 1)

    graph = KnowledgeGraphMemory(tmp_path / "graph.db")
    try:
        relation = graph.query_pattern(relation_type="built_on").relations[0]
        assert relation.evidence[0]["source"] == "conversation:test:turn-1"
        assert relation.evidence[0]["quote"] == "building ACE on OpenJarvis"
        assert graph.relation_count() == 1
        assert "source: conversation:test:turn-1" in graph.context_for("ACE")
    finally:
        graph.close()


def test_conversation_graph_accepts_only_quoted_validity_dates(tmp_path):
    text = "ACE used Qwen 3B from 2024-01-01 through 2024-08-31."
    engine = _FakeEngine(
        {
            "entities": [
                {"name": "ACE", "type": "project", "quote": "ACE"},
                {"name": "Qwen 3B", "type": "model", "quote": "Qwen 3B"},
            ],
            "relations": [
                {
                    "subject": "ACE",
                    "predicate": "used",
                    "object": "Qwen 3B",
                    "quote": text,
                    "valid_from": "2024-01-01",
                    "valid_until": "2024-08-31",
                }
            ],
        }
    )
    extractor = ConversationGraphExtractor(engine, "local-model")
    extracted = extractor.extract(text)
    assert len(extracted.relations) == 1
    assert extracted.relations[0].valid_from == "2024-01-01"
    assert extracted.relations[0].valid_until == "2024-08-31"
    assert extractor.persist(
        extracted, tmp_path / "dated-graph.db", "conversation:test"
    ) == (2, 1)

    graph = KnowledgeGraphMemory(tmp_path / "dated-graph.db")
    try:
        relation = graph.query_pattern(relation_type="used").relations[0]
        assert relation.evidence[0]["valid_from"] == "2024-01-01"
        assert relation.evidence[0]["valid_until"] == "2024-08-31"
    finally:
        graph.close()

    unsupported = _FakeEngine(
        {
            "entities": [
                {"name": "ACE", "type": "project", "quote": "ACE"},
                {"name": "Qwen 3B", "type": "model", "quote": "Qwen 3B"},
            ],
            "relations": [
                {
                    "subject": "ACE",
                    "predicate": "uses",
                    "object": "Qwen 3B",
                    "quote": "ACE uses Qwen 3B.",
                    "valid_from": "2024-01-01",
                }
            ],
        }
    )
    unsupported_result = ConversationGraphExtractor(
        unsupported, "local-model"
    ).extract("ACE uses Qwen 3B.")
    assert unsupported_result.relations[0].valid_from == ""


def test_negated_claims_are_preserved_and_flagged_for_review(tmp_path):
    text = "ACE uses Ollama. ACE does not use Ollama."
    engine = _FakeEngine(
        {
            "entities": [
                {
                    "name": "ACE",
                    "type": "project",
                    "quote": "ACE",
                    "confidence": 0.9,
                },
                {
                    "name": "Ollama",
                    "type": "software",
                    "quote": "Ollama",
                    "confidence": 0.9,
                },
            ],
            "relations": [
                {
                    "subject": "ACE",
                    "predicate": "uses",
                    "object": "Ollama",
                    "quote": "ACE uses Ollama.",
                    "confidence": 0.9,
                    "polarity": "affirmed",
                },
                {
                    "subject": "ACE",
                    "predicate": "uses",
                    "object": "Ollama",
                    "quote": "ACE does not use Ollama.",
                    "confidence": 0.9,
                    "polarity": "negated",
                },
            ],
        }
    )
    extractor = ConversationGraphExtractor(engine, "local-model")
    result = extractor.extract(text)
    assert [relation.polarity for relation in result.relations] == [
        "affirmed",
        "negated",
    ]

    assert extractor.persist(
        result, tmp_path / "graph.db", "conversation:test:polarity"
    ) == (2, 2)
    graph = KnowledgeGraphMemory(tmp_path / "graph.db")
    try:
        relations = graph.query_pattern(relation_type="uses").relations
        assert len(relations) == 2
        assert {relation.polarity for relation in relations} == {"affirmed", "negated"}
        assert len(graph.possible_conflicts()) == 1
        assert graph.possible_conflicts()[0]["kind"] == "opposing_polarity"
        context = graph.context_for("ACE")
        assert "affirmed" in context and "negated" in context
        assert graph.relation_count() == 2
    finally:
        graph.close()


def test_conversation_graph_skips_messages_with_detected_secrets(tmp_path):
    engine = _FakeEngine({"entities": [], "relations": []})

    class _NoFacts:
        def extract(self, _user_text, _assistant_text):
            return []

    class _CleanInjectionScanner:
        def scan(self, _text):
            return SimpleNamespace(is_clean=True, threat_level="low")

    class _SecretScanner:
        def scan(self, text):
            return SimpleNamespace(is_clean="api_key=" not in text)

    service = MemoryService(
        LocalFactStore(tmp_path / "facts.jsonl"),
        _NoFacts(),
        scanner=_CleanInjectionScanner(),
        graph_extractor=ConversationGraphExtractor(engine, "local-model"),
        graph_db_path=str(tmp_path / "graph.db"),
        secret_scanner=_SecretScanner(),
        require_secret_scanner=True,
    )

    service._process(
        ("My api_key=sk-example is secret", "Noted", "test", "turn-secret")
    )

    assert engine.calls == []
    assert not (tmp_path / "graph.db").exists()


def test_memory_service_persists_and_recalls_episode_linked_graph_evidence(
    tmp_path, monkeypatch
):
    engine = _FakeEngine(
        {
            "entities": [
                {
                    "name": "ACE",
                    "type": "project",
                    "quote": "building ACE",
                    "confidence": 0.9,
                },
                {
                    "name": "OpenJarvis",
                    "type": "software",
                    "quote": "on OpenJarvis",
                    "confidence": 0.9,
                },
            ],
            "relations": [
                {
                    "subject": "ACE",
                    "predicate": "built_on",
                    "object": "OpenJarvis",
                    "quote": "building ACE on OpenJarvis",
                    "confidence": 0.85,
                }
            ],
        }
    )

    class _NoFacts:
        def extract(self, _user_text, _assistant_text):
            return []

    class _CleanScanner:
        def scan(self, _text):
            return SimpleNamespace(is_clean=True, threat_level="low")

    episodes_path = tmp_path / "episodes.db"
    episode_store = EpisodeStore(episodes_path, semantic_enabled=False)
    service = MemoryService(
        LocalFactStore(tmp_path / "facts.jsonl"),
        _NoFacts(),
        scanner=_CleanScanner(),
        episode_store=episode_store,
        graph_extractor=ConversationGraphExtractor(engine, "local-model"),
        graph_db_path=str(tmp_path / "graph.db"),
        secret_scanner=_CleanScanner(),
        require_secret_scanner=True,
    )
    service._process(
        (
            "This project is building ACE on OpenJarvis.",
            "The assistant also said ACE uses another tool.",
            "chat",
            "turn-123",
            "session-1",
        )
    )
    episode_id = EpisodeStore.episode_id_for(
        source="chat",
        session_id="session-1",
        idempotency_key="turn-123",
    )
    episode_store.close()

    graph = KnowledgeGraphMemory(tmp_path / "graph.db")
    try:
        relation = graph.query_pattern(relation_type="built_on").relations[0]
        assert relation.evidence[0]["source"] == "conversation:chat:turn-123"
        assert relation.evidence[0]["quote"] == "building ACE on OpenJarvis"
        assert relation.properties["episode_ids"] == [episode_id]
        assert "source: conversation:chat:turn-123" in graph.context_for("ACE")
    finally:
        graph.close()

    monkeypatch.setattr(EpisodeStore, "search", lambda self, query, *, top_k=3: [])
    recalled = inject_context(
        "ACE",
        [Message(role=Role.USER, content="ACE")],
        None,
        config=ContextConfig(
            graph_enabled=True,
            graph_db_path=str(tmp_path / "graph.db"),
            episodes_enabled=True,
            episodes_path=str(episodes_path),
            episodes_semantic_enabled=False,
            procedure_outcomes_top_k=0,
        ),
    )
    assert "Episode linked from graph" in recalled[0].content
    assert episode_id in recalled[0].content
    assert "The assistant also said ACE uses another tool." in recalled[0].content


def test_memory_service_links_graph_relations_to_saved_episodes(tmp_path):
    engine = _FakeEngine(
        {
            "entities": [
                {"name": "ACE", "type": "project", "quote": "building ACE"},
                {
                    "name": "OpenJarvis",
                    "type": "software",
                    "quote": "on OpenJarvis",
                },
            ],
            "relations": [
                {
                    "subject": "ACE",
                    "predicate": "built_on",
                    "object": "OpenJarvis",
                    "quote": "building ACE on OpenJarvis",
                }
            ],
        }
    )

    class _NoFacts:
        def extract(self, _user_text, _assistant_text):
            return []

    class _CleanScanner:
        def scan(self, _text):
            return SimpleNamespace(is_clean=True, threat_level="low")

    episodes = EpisodeStore(tmp_path / "episodes.db", semantic_enabled=False)
    service = MemoryService(
        LocalFactStore(tmp_path / "facts.jsonl"),
        _NoFacts(),
        scanner=_CleanScanner(),
        episode_store=episodes,
        graph_extractor=ConversationGraphExtractor(engine, "local-model"),
        graph_db_path=str(tmp_path / "graph.db"),
        secret_scanner=_CleanScanner(),
        require_secret_scanner=True,
    )
    user_text = "This project is building ACE on OpenJarvis."
    exchange_ids = [f"turn-{index}" for index in range(51)]
    for exchange_id in exchange_ids:
        service._process(
            (
                user_text,
                "I will keep that in mind.",
                "chat",
                exchange_id,
                "session-1",
            )
        )

    expected_episode_ids = [
        episodes.episode_id_for(
            source="chat",
            session_id="session-1",
            idempotency_key=exchange_id,
        )
        for exchange_id in exchange_ids
    ][-50:]
    assert episodes.count() == 51
    graph = KnowledgeGraphMemory(tmp_path / "graph.db")
    try:
        relation = graph.query_pattern(relation_type="built_on").relations[0]
        assert relation.properties["episode_ids"] == expected_episode_ids
        assert len(relation.evidence) == 51
        assert episodes.delete(expected_episode_ids[-1])
        assert episodes.count() == 50
        assert (
            expected_episode_ids[-1]
            in graph.query_pattern(relation_type="built_on")
            .relations[0]
            .properties["episode_ids"]
        )
    finally:
        graph.close()
        episodes.close()


def test_conversation_graph_uses_only_local_engines():
    engine = _FakeEngine({"entities": [], "relations": []})
    engine.is_cloud = True

    result = ConversationGraphExtractor(engine, "cloud-model").extract(
        "I am building ACE."
    )

    assert result.entities == []
    assert result.relations == []
    assert engine.calls == []


def test_conversation_graph_rejects_unverified_remote_endpoint():
    engine = _FakeEngine({"entities": [], "relations": []})
    engine.is_local = False
    engine._host = "https://inference.example"

    result = ConversationGraphExtractor(engine, "remote-model").extract(
        "I am building ACE."
    )

    assert result.entities == []
    assert result.relations == []
    assert engine.calls == []


def test_memory_service_builder_enables_capture_only_when_configured(tmp_path):
    memory = StorageConfig(
        enabled=True,
        extraction_model="local-model",
        facts_path=str(tmp_path / "facts.jsonl"),
        graph_db_path=str(tmp_path / "graph.db"),
        graph_capture_enabled=True,
    )
    config = SimpleNamespace(
        memory=memory,
        security=SimpleNamespace(secret_scanner=False),
    )

    local_service = build_memory_service(config, _FakeEngine({}), "local-model")
    assert local_service is not None
    assert isinstance(local_service._graph_extractor, ConversationGraphExtractor)
    assert local_service._graph_db_path == str(tmp_path / "graph.db")

    cloud_engine = _FakeEngine({})
    cloud_engine.is_cloud = True
    cloud_service = build_memory_service(config, cloud_engine, "cloud-model")
    assert cloud_service is not None
    assert cloud_service._graph_extractor is None

    instrumented_cloud = InstrumentedEngine(cloud_engine, EventBus())
    wrapped_cloud_service = build_memory_service(
        config, instrumented_cloud, "cloud-model"
    )
    assert wrapped_cloud_service is not None
    assert wrapped_cloud_service._graph_extractor is None


def test_memory_service_graph_capture_follows_selected_multi_engine_model(tmp_path):
    config = SimpleNamespace(
        memory=StorageConfig(
            enabled=True,
            extraction_model="",
            facts_path=str(tmp_path / "facts.jsonl"),
            graph_db_path=str(tmp_path / "graph.db"),
            graph_capture_enabled=True,
        ),
        security=SimpleNamespace(secret_scanner=False),
    )
    local_engine = _FakeEngine({}, models=["local-model"])
    local_engine.engine_id = "ollama"
    cloud_engine = _FakeEngine({}, models=["gpt-private"])
    cloud_engine.is_cloud = True
    cloud_engine.is_local = False
    cloud_engine.engine_id = "cloud"
    engine = MultiEngine([("ollama", local_engine), ("cloud", cloud_engine)])

    local_service = build_memory_service(config, engine, "local-model")
    cloud_service = build_memory_service(config, engine, "gpt-private")

    assert local_service is not None
    assert isinstance(local_service._graph_extractor, ConversationGraphExtractor)
    assert cloud_service is not None
    assert cloud_service._graph_extractor is None


def test_multi_engine_locality_check_does_not_refresh_unknown_routes():
    local_engine = _FakeEngine({}, models=["local-model"])
    local_engine.engine_id = "ollama"
    engine = MultiEngine([("ollama", local_engine)])

    # Model discovery can touch a remote provider. A privacy gate uses only
    # routes already known to the router and fails closed for unknown models.
    local_engine.models = ["new-local-model"]
    assert engine.is_local_for_model("new-local-model") is False
    assert engine.is_local_for_model("local-model") is True


def test_document_graph_capture_keeps_source_evidence_and_deduplicates(tmp_path):
    document_text = (
        "The ACE project uses OpenJarvis. The ACE project stores knowledge in SQLite."
    )
    payload = {
        "entities": [
            {"name": "ACE", "type": "project", "quote": "ACE project"},
            {"name": "OpenJarvis", "type": "software", "quote": "OpenJarvis"},
            {"name": "SQLite", "type": "software", "quote": "SQLite"},
        ],
        "relations": [
            {
                "subject": "ACE",
                "predicate": "uses",
                "object": "OpenJarvis",
                "quote": "ACE project uses OpenJarvis",
                "confidence": 0.9,
            },
            {
                "subject": "ACE",
                "predicate": "stores_knowledge_in",
                "object": "SQLite",
                "quote": "ACE project stores knowledge in SQLite",
                "confidence": 0.9,
            },
        ],
    }
    engine = _FakeEngine(payload)
    extractor = DocumentGraphExtractor(engine, "local-model")

    extraction = extractor.extract(document_text)

    assert len(extraction.entities) == 3
    assert len(extraction.relations) == 2
    prompt = engine.calls[0][0][1].text
    assert "SOURCE DOCUMENT (untrusted)" in prompt
    assert "explicitly stated in the source document" in prompt
    assert document_text in prompt
    assert extractor.persist(
        extraction,
        tmp_path / "graph.db",
        "document:/notes/ace.md",
        source_kind="document",
    ) == (3, 2)
    assert extractor.persist(
        extraction,
        tmp_path / "graph.db",
        "document:/archive/ace.md",
        source_kind="document",
    ) == (3, 2)

    graph = KnowledgeGraphMemory(tmp_path / "graph.db")
    try:
        relations = graph.query_pattern().relations
        assert len(relations) == 2
        assert all(len(relation.evidence) == 2 for relation in relations)
        assert {
            relation.source for relation in relations
        } <= {"document:/notes/ace.md", "document:/archive/ace.md"}
        assert {
            item["source"]
            for relation in relations
            for item in relation.evidence
        } == {"document:/notes/ace.md", "document:/archive/ace.md"}
        ace = next(
            entity for entity in graph.query_pattern().entities if entity.name == "ACE"
        )
        assert len(ace.properties["evidence_sources"]) == 2
        assert {item["source_kind"] for item in ace.properties["evidence_sources"]} == {
            "document"
        }
    finally:
        graph.close()


def test_document_graph_capture_rejects_unsafe_quotes(tmp_path):
    document_text = "ACE uses OpenJarvis."
    engine = _FakeEngine(
        {
            "entities": [
                {"name": "ACE", "type": "project", "quote": "ACE"},
                {"name": "OpenJarvis", "type": "software", "quote": "OpenJarvis"},
            ],
            "relations": [
                {
                    "subject": "ACE",
                    "predicate": "uses",
                    "object": "OpenJarvis",
                    "quote": document_text,
                }
            ],
        }
    )
    extractor = DocumentGraphExtractor(engine, "local-model")
    extraction = extractor.extract(document_text)

    assert extractor.persist(
        extraction,
        tmp_path / "graph.db",
        "document:/notes/unsafe.md",
        source_kind="document",
        quote_is_safe=lambda _quote: False,
    ) == (0, 0)
    graph = KnowledgeGraphMemory(tmp_path / "graph.db")
    try:
        assert graph.entity_count() == 0
        assert graph.relation_count() == 0
    finally:
        graph.close()
