"""Tests for extended API routes."""

import inspect
import json
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import openjarvis.server.api_routes as api_routes  # noqa: E402
from openjarvis.core.events import EventBus, EventType  # noqa: E402
from openjarvis.security.capabilities import CapabilityPolicy  # noqa: E402
from openjarvis.server.api_routes import (  # noqa: E402
    include_all_routes,
    memory_index,
    memory_router,
    memory_search,
    memory_stats,
    memory_store,
)


def _make_app():
    app = FastAPI()
    include_all_routes(app)
    return app


class TestLocalFactReviewRoutes:
    @pytest.fixture
    def local_facts(self, tmp_path):
        from openjarvis.core.config import JarvisConfig
        from openjarvis.memory.store import LocalFactStore

        path = tmp_path / "facts.jsonl"
        store = LocalFactStore(path)
        original = "ACE uses a green notebook."
        assert store.add(original, source="chat", trust="auto")
        assert store.add("A flagged entry.", trust="untrusted")
        config = JarvisConfig()
        config.memory.facts_path = str(path)
        app = FastAPI()
        app.include_router(memory_router)
        app.state.config = config
        return app, store, original

    def test_local_revision_preserves_history_and_updates_recall(
        self, local_facts
    ):
        app, store, original = local_facts
        replacement = "ACE uses a blue notebook."
        with TestClient(
            app, base_url="http://localhost", client=("127.0.0.1", 50000)
        ) as client:
            headers = {"origin": "http://tauri.localhost"}
            response = client.get("/v1/memory/facts", headers=headers)
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
            assert response.json()["facts"] == [
                {
                    "text": "A flagged entry.",
                    "source": "",
                    "trust": "untrusted",
                    "superseded": False,
                    "trusted_for_recall": False,
                },
                {
                    "text": original,
                    "source": "chat",
                    "trust": "auto",
                    "superseded": False,
                    "trusted_for_recall": True,
                },
            ]
            correction = {"existing_fact": original, "correction": replacement}
            response = client.post(
                "/v1/memory/facts/revise", json=correction, headers=headers
            )
            assert response.status_code == 204, response.text
            assert response.content == b""
            assert response.headers["cache-control"] == "no-store"
            assert client.post(
                "/v1/memory/facts/revise", json=correction, headers=headers
            ).status_code == 409

        rows = {fact.text: fact for fact in store.list()}
        assert rows[original].superseded
        assert not rows[original].trusted_for_recall
        assert rows[replacement].trust == "trusted"
        assert rows[replacement].trusted_for_recall

    @pytest.mark.parametrize(
        "peer,headers",
        [
            ("192.0.2.10", {}),
            ("127.0.0.1", {"host": "external.example"}),
            ("127.0.0.1", {"origin": "https://external.example"}),
            ("127.0.0.1", {"x-forwarded-for": "192.0.2.10"}),
        ],
    )
    def test_outside_loopback_cannot_read_or_change_facts(
        self, local_facts, peer, headers
    ):
        app, store, original = local_facts
        with TestClient(
            app, base_url="http://localhost", client=(peer, 50000)
        ) as client:
            response = client.get("/v1/memory/facts", headers=headers)
            assert response.status_code == 403
            assert original not in response.text
            response = client.post(
                "/v1/memory/facts/revise",
                json={"existing_fact": original, "correction": "Another fixture."},
                headers=headers,
            )
            assert response.status_code == 403
            assert original not in response.text
        assert len(store.list()) == 2

    def test_rejected_and_invalid_corrections_are_not_stored(
        self, local_facts, monkeypatch
    ):
        from openjarvis.memory import service

        app, store, original = local_facts
        monkeypatch.setattr(service, "_build_secret_scanner", lambda: None)
        with TestClient(
            app, base_url="http://localhost", client=("127.0.0.1", 50000)
        ) as client:
            url = "/v1/memory/facts/revise"
            response = client.post(
                url, json={"existing_fact": original, "correction": "Another fixture."}
            )
            assert response.status_code == 422
            assert "Another fixture." not in response.text
            response = client.post(
                url, json={"existing_fact": original, "correction": "One\nTwo"}
            )
            assert response.status_code == 400
            assert original not in response.text
            response = client.post(
                url, json={"existing_fact": original, "correction": "A" * 65_536}
            )
            assert response.status_code == 413
        assert len(store.list()) == 2


class TestEpisodeProvenanceRoutes:
    @staticmethod
    def _app(path):
        from openjarvis.core.config import JarvisConfig

        app = FastAPI()
        app.include_router(memory_router)
        config = JarvisConfig()
        config.memory.episodes_enabled = True
        config.memory.episodes_path = str(path)
        app.state.config = config
        return app

    @staticmethod
    def _record(path):
        from openjarvis.memory.episodes import EpisodeStore

        store = EpisodeStore(path, semantic_enabled=False)
        try:
            assert store.record(
                "The project decision was to keep ACE local.",
                "I recorded that decision.",
                source="chat",
                session_id="private session key",
                idempotency_key="test-turn",
            )
            return store.list()[0]
        finally:
            store.close()

    def test_local_view_returns_only_the_requested_episode_without_caching(
        self, tmp_path
    ):
        path = tmp_path / "episode archive #1.db"
        episode = self._record(path)
        app = self._app(path)
        with TestClient(
            app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000)
        ) as client:
            response = client.get(
                f"/v1/memory/episodes/{episode.episode_id.upper()}",
                headers={"origin": "http://tauri.localhost"},
            )
            assert response.status_code == 200, response.text
            assert response.headers["cache-control"] == "no-store"
            assert response.json() == {
                "episode_id": episode.episode_id,
                "user_text": episode.user_text,
                "assistant_text": episode.assistant_text,
                "source": "chat",
                "created_at": episode.created_at,
            }
            assert client.get("/v1/memory/episodes/not-an-id").status_code == 404
            app.state.config.memory.episodes_enabled = False
            assert (
                client.get(f"/v1/memory/episodes/{episode.episode_id}").status_code
                == 404
            )

    @pytest.mark.parametrize(
        "client_host,headers",
        [
            ("192.0.2.10", {}),
            ("127.0.0.1", {"host": "external.example"}),
            ("127.0.0.1", {"origin": "https://external.example"}),
            ("127.0.0.1", {"x-forwarded-for": "192.0.2.10"}),
        ],
    )
    def test_episode_text_is_unavailable_outside_direct_local_access(
        self, tmp_path, client_host, headers
    ):
        path = tmp_path / "episodes.db"
        episode = self._record(path)
        with TestClient(
            self._app(path),
            base_url="http://127.0.0.1",
            client=(client_host, 50000),
        ) as client:
            response = client.get(
                f"/v1/memory/episodes/{episode.episode_id}", headers=headers
            )
            assert response.status_code == 403
            assert episode.user_text not in response.text

    def test_missing_archive_returns_404_without_creating_a_database(self, tmp_path):
        path = tmp_path / "missing.db"
        with TestClient(
            self._app(path), base_url="http://localhost", client=("::1", 50000)
        ) as client:
            response = client.get(f"/v1/memory/episodes/{'a' * 32}")
            assert response.status_code == 404
        assert not path.exists()

    def test_local_episode_view_honors_configured_api_auth(self, tmp_path):
        from openjarvis.server.auth_middleware import AuthMiddleware

        path = tmp_path / "episodes.db"
        episode = self._record(path)
        app = self._app(path)
        app.add_middleware(AuthMiddleware, api_key="local-test-key")
        with TestClient(
            app, base_url="http://localhost", client=("127.0.0.1", 50000)
        ) as client:
            path = f"/v1/memory/episodes/{episode.episode_id}"
            assert client.get(path).status_code == 401
            authorized = client.get(
                path, headers={"authorization": "Bearer local-test-key"}
            )
            assert authorized.status_code == 200


def test_knowledge_graph_api_supports_inspection_edits_and_deletion(tmp_path):
    app = FastAPI()
    app.include_router(memory_router)
    app.state.config = SimpleNamespace(
        memory=SimpleNamespace(graph_db_path=str(tmp_path / "graph.db"))
    )
    client = TestClient(app)
    entity_a = client.post(
        "/v1/memory/graph/entities",
        json={"entity_id": "acme", "entity_type": "company", "name": "Acme"},
    )
    entity_b = client.post(
        "/v1/memory/graph/entities",
        json={"entity_id": "quartz", "entity_type": "product", "name": "Quartz"},
    )
    assert entity_a.status_code == entity_b.status_code == 200
    relation = client.post(
        "/v1/memory/graph/relations",
        json={
            "source_id": "acme",
            "target_id": "quartz",
            "relation_type": "current_version",
            "source_url": "https://example.org/acme",
            "evidence_quote": "Acme's current version is Quartz.",
            "valid_from": "2020-01-01",
            "valid_until": "2023-12-31",
        },
    )
    assert relation.status_code == 200
    relation_id = relation.json()["relation"]["id"]
    view = client.get("/v1/memory/graph?query=Acme")
    assert view.status_code == 200
    assert view.json()["counts"] == {"entities": 2, "relations": 1, "evidence": 1}
    assert (
        view.json()["relations"][0]["evidence"][0]["quote"]
        == "Acme's current version is Quartz."
    )
    assert view.json()["relations"][0]["evidence"][0]["valid_from"] == "2020-01-01"
    entity_c = client.post(
        "/v1/memory/graph/entities",
        json={"entity_id": "opal", "entity_type": "product", "name": "Opal"},
    )
    assert entity_c.status_code == 200
    competing = client.post(
        "/v1/memory/graph/relations",
        json={
            "source_id": "acme",
            "target_id": "opal",
            "relation_type": "current_version",
            "source_url": "https://example.org/opal",
            "evidence_quote": "Acme's current version is Opal.",
            "valid_from": "2022-01-01",
            "valid_until": "2024-12-31",
        },
    )
    assert competing.status_code == 200
    view = client.get("/v1/memory/graph?query=Acme")
    candidates = view.json()["possible_conflicts"]
    assert len(candidates) == 1
    assert candidates[0]["subject"]["name"] == "Acme"
    assert candidates[0]["kind"] == "different_targets"
    assert {claim["target"]["name"] for claim in candidates[0]["claims"]} == {
        "Quartz",
        "Opal",
    }

    edited = client.post(
        "/v1/memory/graph/entities",
        json={"entity_id": "acme", "entity_type": "company", "name": "Acme Inc."},
    )
    assert edited.json()["entity"]["name"] == "Acme Inc."
    revised = client.put(
        f"/v1/memory/graph/relations/{relation_id}",
        json={
            "source_id": "acme",
            "target_id": "quartz",
            "relation_type": "manufactures",
            "source_url": "https://example.org/acme",
            "evidence_quote": "Acme manufactures Quartz.",
            "status": "verified",
            "valid_from": "2024-01-01",
        },
    )
    assert revised.status_code == 200
    relation_id = revised.json()["relation"]["id"]
    assert revised.json()["relation"]["type"] == "manufactures"
    assert revised.json()["relation"]["evidence"][0]["valid_from"] == "2024-01-01"
    assert client.delete(f"/v1/memory/graph/relations/{relation_id}").status_code == 200
    assert client.delete("/v1/memory/graph/entities/quartz").status_code == 200
    remaining = client.get("/v1/memory/graph").json()
    assert remaining["counts"]["relations"] == 1
    assert remaining["relations"][0]["target"] == "opal"


def test_knowledge_graph_focus_pages_neighbors_outside_global_slice(tmp_path):
    from openjarvis.tools.storage.knowledge_graph import (
        Entity,
        KnowledgeGraphMemory,
        Relation,
    )

    path = tmp_path / "graph.db"
    with KnowledgeGraphMemory(path) as graph:
        graph.add_entity(Entity("center:ACE", "project", "ACE"))
        graph.add_entity(Entity("old", "software", "Old important neighbor"))
        graph.add_entity(Entity("new", "software", "New neighbor"))
        graph.add_entity(Entity("unrelated", "project", "Unrelated recent project"))
        graph.add_relation(Relation("old", "center:ACE", "informs"))
        graph.add_relation(Relation("center:ACE", "new", "uses"))
        graph.add_relation(Relation("center:ACE", "center:ACE", "documents"))

    app = FastAPI()
    app.include_router(memory_router)
    app.state.config = SimpleNamespace(memory=SimpleNamespace(graph_db_path=str(path)))
    with TestClient(app) as client:
        global_view = client.get("/v1/memory/graph", params={"limit": 1}).json()
        assert "old" not in {item["id"] for item in global_view["entities"]}

        params = {"focus_id": "center:ACE", "limit": 2}
        first = client.get("/v1/memory/graph", params=params)
        assert first.status_code == 200
        first_data = first.json()
        assert {entity["id"] for entity in first_data["entities"]} == {
            "center:ACE",
            "new",
        }
        assert first_data["entities"][0]["id"] == "center:ACE"
        assert {relation["type"] for relation in first_data["relations"]} == {
            "documents",
            "uses",
        }
        assert first_data["next_relation_id"] is not None

        second = client.get(
            "/v1/memory/graph",
            params={**params, "before_relation_id": first_data["next_relation_id"]},
        )
        assert second.status_code == 200
        second_data = second.json()
        assert [entity["id"] for entity in second_data["entities"]] == [
            "center:ACE",
            "old",
        ]
        assert second_data["relations"][0]["source"] == "old"
        assert second_data["relations"][0]["target"] == "center:ACE"
        assert second_data["next_relation_id"] is None

        isolated = client.get(
            "/v1/memory/graph", params={"focus_id": "unrelated"}
        ).json()
        assert len(isolated["entities"]) == 1
        assert not isolated["relations"]
        assert (
            client.get(
                "/v1/memory/graph", params={"focus_id": "nonexistent"}
            ).status_code
            == 404
        )
        assert (
            client.get(
                "/v1/memory/graph", params={**params, "before_relation_id": -1}
            ).status_code
            == 400
        )
        assert (
            client.get(
                "/v1/memory/graph", params={**params, "query": "ACE"}
            ).status_code
            == 400
        )


def test_knowledge_graph_status_update_keeps_exact_episode_evidence(tmp_path):
    from openjarvis.tools.storage.knowledge_graph import (
        Entity,
        KnowledgeGraphMemory,
        Relation,
    )

    graph_path = tmp_path / "graph.db"
    episode_id = "a" * 32
    with KnowledgeGraphMemory(graph_path) as graph:
        graph.add_entity(Entity("ace", "project", "ACE"))
        graph.add_entity(Entity("jarvis", "software", "OpenJarvis"))
        relation_id = graph.add_relation(
            Relation(
                "ace",
                "jarvis",
                "uses",
                weight=0.4,
                evidence=[
                    {
                        "source": "conversation",
                        "quote": "ACE uses OpenJarvis.",
                        "episode_id": episode_id,
                    }
                ],
            )
        )

    app = FastAPI()
    app.include_router(memory_router)
    app.state.config = SimpleNamespace(
        memory=SimpleNamespace(graph_db_path=str(graph_path))
    )
    client = TestClient(app)
    before = client.get("/v1/memory/graph").json()["relations"][0]
    response = client.put(
        f"/v1/memory/graph/relations/{relation_id}",
        json={
            "source_id": "ace",
            "target_id": "jarvis",
            "relation_type": "uses",
            "status": "verified",
            "weight": before["weight"],
            "confidence": before["confidence"],
        },
    )

    assert response.status_code == 200, response.text
    updated = response.json()["relation"]
    assert updated["status"] == "verified"
    assert updated["weight"] == 0.4
    assert updated["evidence"] == before["evidence"]
    assert len(updated["evidence"]) == 1
    assert updated["evidence"][0]["episode_id"] == episode_id


def test_knowledge_graph_api_exposes_evidence_review_candidates(tmp_path):
    from openjarvis.tools.storage.knowledge_graph import (
        Entity,
        KnowledgeGraphMemory,
        Relation,
    )

    graph_path = tmp_path / "graph.db"
    graph = KnowledgeGraphMemory(graph_path)
    graph.add_entity(Entity("ace", "project", "ACE"))
    graph.add_entity(Entity("model", "software", "ACE local model"))
    graph.add_relation(
        Relation(
            "ace",
            "model",
            "current_model",
            evidence=[
                {
                    "source": "document:old-config",
                    "quote": "ACE uses this local model.",
                    "observed_at": 1.0,
                },
                {
                    "source": "document:superseded-config",
                    "quote": "This later evidence was superseded.",
                    "observed_at": 3.0,
                    "status": "superseded",
                },
                {
                    "source": "conversation:latest-active-config",
                    "episode_id": "b" * 32,
                    "quote": (
                        "The latest active evidence still says ACE uses "
                        "this local model."
                    ),
                    "observed_at": 0.5,
                },
            ],
        )
    )
    with graph._conn:
        graph._conn.execute(
            "UPDATE relation_evidence SET created_at=? WHERE source=?",
            (1.0, "document:old-config"),
        )
        graph._conn.execute(
            "UPDATE relation_evidence SET created_at=? WHERE source=?",
            (2.0, "conversation:latest-active-config"),
        )
    graph.close()

    app = FastAPI()
    app.include_router(memory_router)
    app.state.config = SimpleNamespace(
        memory=SimpleNamespace(graph_db_path=str(graph_path))
    )
    response = TestClient(app).get("/v1/memory/graph?query=ACE")

    assert response.status_code == 200
    candidates = response.json()["evidence_review_candidates"]
    assert len(candidates) == 1
    assert candidates[0]["subject"]["name"] == "ACE"
    assert candidates[0]["target"]["name"] == "ACE local model"
    assert candidates[0]["relation"]["evidence"][0]["quote"] == (
        "This later evidence was superseded."
    )
    assert candidates[0]["last_evidence"]["quote"] == (
        "The latest active evidence still says ACE uses this local model."
    )
    assert candidates[0]["last_evidence"]["source"] == (
        "conversation:latest-active-config"
    )
    assert candidates[0]["last_evidence"]["episode_id"] == "b" * 32
    assert candidates[0]["last_evidence_at"] == 2.0
    assert candidates[0]["review_after_days"] == 90
    assert candidates[0]["age_days"] > candidates[0]["review_after_days"]


def test_knowledge_graph_event_history_is_bounded_and_paginated(tmp_path):
    from openjarvis.tools.storage.knowledge_graph import (
        Entity,
        KnowledgeGraphMemory,
        Relation,
    )

    graph_path = tmp_path / "graph.db"
    graph = KnowledgeGraphMemory(graph_path)
    graph.add_entity(Entity("ace", "project", "ACE", properties={"private": "omit"}))
    graph.add_entity(Entity("jarvis", "software", "OpenJarvis"))
    graph.add_relation(
        Relation(
            "ace",
            "jarvis",
            "uses",
            evidence=[
                {
                    "source": "manual",
                    "quote": "ACE uses OpenJarvis.",
                }
            ],
        )
    )
    graph.close()

    app = FastAPI()
    app.include_router(memory_router)
    app.state.config = SimpleNamespace(
        memory=SimpleNamespace(graph_db_path=str(graph_path))
    )
    client = TestClient(app)

    first = client.get("/v1/memory/graph/events?limit=2")
    assert first.status_code == 200
    first_page = first.json()
    assert len(first_page["events"]) == 2
    assert first_page["next_before_id"] == first_page["events"][-1]["id"]
    assert first_page["events"][0]["id"] > first_page["events"][1]["id"]
    assert first_page["events"][0]["after"]["evidence"][0]["quote"] == (
        "ACE uses OpenJarvis."
    )

    second = client.get(
        "/v1/memory/graph/events",
        params={"limit": 2, "before_id": first_page["next_before_id"]},
    )
    second_page = second.json()
    assert len(second_page["events"]) == 1
    assert second_page["events"][0]["event_type"] == "entity_added"
    assert second_page["events"][0]["after"]["property_keys"] == ["private"]
    assert second_page["next_before_id"] is None
    serialized_events = json.dumps([*first_page["events"], *second_page["events"]])
    assert '"omit"' not in serialized_events


def test_knowledge_graph_event_history_without_database_is_empty(tmp_path):
    app = FastAPI()
    app.include_router(memory_router)
    app.state.config = SimpleNamespace(
        memory=SimpleNamespace(graph_db_path=str(tmp_path / "missing.db"))
    )

    response = TestClient(app).get("/v1/memory/graph/events")
    assert response.status_code == 200
    assert response.json() == {"events": [], "next_before_id": None}


def test_graph_memory_purge_event_exposes_counts_without_episode_content():
    from openjarvis.server.api_routes import _graph_event_snapshot

    snapshot = _graph_event_snapshot(
        "memory_data_purged",
        {
            "entities_removed": 1,
            "entities_updated": 2,
            "relations_removed": 3,
            "relations_updated": 4,
            "evidence_removed": 5,
            "episode_id": "a" * 32,
            "quote": "private conversation text",
        },
    )

    assert snapshot == {
        "entities_removed": 1,
        "entities_updated": 2,
        "relations_removed": 3,
        "relations_updated": 4,
        "evidence_removed": 5,
    }


class TestAgentRoutes:
    def test_list_agents(self):
        client = TestClient(_make_app())
        resp = client.get("/v1/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "registered" in data
        assert "running" in data

    def test_create_agent(self):
        client = TestClient(_make_app())
        resp = client.post("/v1/agents", json={"agent_type": "simple"})
        # May succeed or fail depending on agent_tools availability
        assert resp.status_code in (200, 501)

    def test_kill_nonexistent(self):
        client = TestClient(_make_app())
        resp = client.delete("/v1/agents/nonexistent")
        assert resp.status_code in (404, 501)

    def test_admin_lifecycle_is_capability_gated_before_mutation(self):
        from openjarvis.tools.agent_tools import _SPAWNED_AGENTS

        app = _make_app()
        app.state.bus = EventBus(record_history=True)
        app.state.capability_policy = CapabilityPolicy(default_deny=True)
        app.state.rate_limiter = None
        _SPAWNED_AGENTS.pop("forbidden-api-agent", None)

        resp = TestClient(app).post(
            "/v1/agents",
            json={
                "agent_type": "simple",
                "agent_id": "forbidden-api-agent",
            },
        )

        assert resp.status_code == 403
        assert "forbidden-api-agent" not in _SPAWNED_AGENTS
        denied = [
            event
            for event in app.state.bus.history
            if event.event_type == EventType.CAPABILITY_DENIED
        ]
        assert denied[-1].data == {
            "agent_id": "server:api",
            "capability": "system:admin",
            "tool": "agent_spawn",
        }

    def test_agent_listing_is_capability_gated_before_disclosure(self):
        from openjarvis.tools.agent_tools import _SPAWNED_AGENTS

        app = _make_app()
        app.state.bus = EventBus(record_history=True)
        app.state.capability_policy = CapabilityPolicy(default_deny=True)
        app.state.rate_limiter = None
        _SPAWNED_AGENTS["private-api-agent"] = {
            "agent_id": "private-api-agent",
            "agent_type": "simple",
            "status": "running",
            "created_at": 0.0,
        }
        try:
            resp = TestClient(app).get("/v1/agents")
        finally:
            _SPAWNED_AGENTS.pop("private-api-agent", None)

        assert resp.status_code == 403
        assert "private-api-agent" not in resp.text
        assert any(
            event.event_type == EventType.CAPABILITY_DENIED
            and event.data
            == {
                "agent_id": "server:api",
                "capability": "system:admin",
                "tool": "agent_list",
            }
            for event in app.state.bus.history
        )

    def test_admin_lifecycle_is_rate_limited_before_mutation(self):
        from openjarvis.tools.agent_tools import _SPAWNED_AGENTS

        class _DenyLimiter:
            def __init__(self) -> None:
                self.keys = []

            def check(self, key: str):
                self.keys.append(key)
                return False, 2.5

        app = _make_app()
        app.state.bus = EventBus(record_history=True)
        policy = CapabilityPolicy(default_deny=True)
        policy.grant("server:api", "system:admin")
        limiter = _DenyLimiter()
        app.state.capability_policy = policy
        app.state.rate_limiter = limiter
        _SPAWNED_AGENTS.pop("limited-api-agent", None)

        resp = TestClient(app).post(
            "/v1/agents",
            json={"agent_type": "simple", "agent_id": "limited-api-agent"},
        )

        assert resp.status_code == 429
        assert "limited-api-agent" not in _SPAWNED_AGENTS
        assert limiter.keys == ["server:api:agent_spawn"]
        assert any(
            event.event_type == EventType.RATE_LIMITED
            and event.data["agent_id"] == "server:api"
            for event in app.state.bus.history
        )

    def test_agent_listing_is_rate_limited_before_disclosure(self):
        class _DenyLimiter:
            def __init__(self) -> None:
                self.keys = []

            def check(self, key: str):
                self.keys.append(key)
                return False, 2.5

        app = _make_app()
        app.state.bus = EventBus(record_history=True)
        policy = CapabilityPolicy(default_deny=True)
        policy.grant("server:api", "system:admin")
        limiter = _DenyLimiter()
        app.state.capability_policy = policy
        app.state.rate_limiter = limiter

        resp = TestClient(app).get("/v1/agents")

        assert resp.status_code == 429
        assert limiter.keys == ["server:api:agent_list"]
        assert any(
            event.event_type == EventType.RATE_LIMITED
            and event.data["agent_id"] == "server:api"
            and event.data["tool"] == "agent_list"
            for event in app.state.bus.history
        )


class TestMemoryRoutes:
    # These tests only assert route wiring. A missing FTS5 build may still
    # make the optional memory feature unavailable.
    _BACKEND_OPTIONAL = (200, 500, 503)

    def test_blocking_memory_handlers_use_starlette_threadpool(self):
        """Sync endpoints are automatically dispatched to worker threads."""
        for handler in (memory_store, memory_search, memory_stats, memory_index):
            assert not inspect.iscoroutinefunction(handler)

    def test_search(self):
        client = TestClient(_make_app())
        resp = client.post("/v1/memory/search", json={"query": "test"})
        # May fail if SQLite not set up, that's ok
        assert resp.status_code in self._BACKEND_OPTIONAL

    def test_stats(self):
        client = TestClient(_make_app())
        resp = client.get("/v1/memory/stats")
        assert resp.status_code in self._BACKEND_OPTIONAL

    def test_index_captures_document_graph_with_local_model_in_mixed_pool(
        self, tmp_path, monkeypatch
    ):
        from openjarvis.tools.storage.knowledge_graph import KnowledgeGraphMemory

        document = tmp_path / "ace-notes.md"
        document.write_text("ACE uses OpenJarvis.", encoding="utf-8")
        graph_path = tmp_path / "graph.db"
        document_text = "ACE uses OpenJarvis."
        chunks = [
            SimpleNamespace(
                source=str(document), content=document_text, metadata={"kind": "note"}
            )
        ]
        monkeypatch.delenv("OPENJARVIS_WORKSPACE", raising=False)
        ingest_options = {}

        def fake_ingest(_path, **kwargs):
            ingest_options.update(kwargs)
            return chunks

        monkeypatch.setattr("openjarvis.tools.storage.ingest.ingest_path", fake_ingest)

        class _Backend:
            def __init__(self):
                self.records = []

            def store(self, content, *, metadata):
                self.records.append((content, metadata))

        class _LocalEngine:
            is_local = True

            def __init__(self):
                self.calls = []

            def list_models(self):
                return ["local-model"]

            def generate(self, messages, **_kwargs):
                self.calls.append(messages)
                return {
                    "content": json.dumps(
                        {
                            "entities": [
                                {"name": "ACE", "type": "project", "quote": "ACE"},
                                {
                                    "name": "OpenJarvis",
                                    "type": "software",
                                    "quote": "OpenJarvis",
                                },
                            ],
                            "relations": [
                                {
                                    "subject": "ACE",
                                    "predicate": "uses",
                                    "object": "OpenJarvis",
                                    "quote": document_text,
                                    "confidence": 0.9,
                                }
                            ],
                        }
                    )
                }

        class _CloudEngine:
            is_cloud = True

            def list_models(self):
                return ["cloud-model"]

            def generate(self, *_args, **_kwargs):
                raise AssertionError("document extraction must remain local")

        local_engine = _LocalEngine()
        mixed_engine = SimpleNamespace(
            _engines=[("cloud", _CloudEngine()), ("custom_local", local_engine)]
        )
        backend = _Backend()
        monkeypatch.setattr(api_routes, "_get_memory_backend", lambda _request: backend)

        class _CleanScanner:
            def scan(self, _text):
                return SimpleNamespace(is_clean=True)

        monkeypatch.setattr(
            "openjarvis.memory.service._build_scanner", lambda: _CleanScanner()
        )
        monkeypatch.setattr(
            "openjarvis.memory.service._build_secret_scanner", lambda: _CleanScanner()
        )
        app = FastAPI()
        app.include_router(memory_router)
        app.state.config = SimpleNamespace(
            memory=SimpleNamespace(
                graph_db_path=str(graph_path), extraction_model="local-model"
            ),
            security=SimpleNamespace(secret_scanner=True),
        )
        app.state.engine = mixed_engine
        app.state.engine_name = "multi"
        app.state.model = "cloud-model"

        response = TestClient(app).post(
            "/v1/memory/index",
            json={
                "path": str(document),
                "ocr": True,
                "transcribe_media": True,
            },
        )

        assert response.status_code == 200
        result = response.json()
        assert result["chunks_indexed"] == 1
        assert result["ocr_requested"] is True
        assert result["transcription_requested"] is True
        assert ingest_options == {
            "ocr": True,
            "transcribe_media": True,
            "speech_config": None,
        }
        assert result["graph_capture"] == {
            "status": "complete",
            "documents_processed": 1,
            "documents_skipped": 0,
            "entities_stored": 2,
            "relations_stored": 1,
            "failures": 0,
        }
        assert backend.records == [
            (document_text, {"source": str(document), "kind": "note"})
        ]
        assert len(local_engine.calls) == 1
        graph = KnowledgeGraphMemory(graph_path)
        try:
            relation = graph.query_pattern(relation_type="uses").relations[0]
            assert relation.source == f"document:{document}"
            assert relation.evidence[0]["source"] == f"document:{document}"
            assert relation.evidence[0]["quote"] == document_text
        finally:
            graph.close()


class TestMemoryPythonFallback:
    """Default SQLite memory remains operational without the optional Rust wheel."""

    @staticmethod
    def _client(monkeypatch, tmp_path):
        # Force the same import failure as a venv without the compiled extension.
        def _boom():
            raise ImportError("No module named 'openjarvis_rust'")

        import openjarvis._rust_bridge as bridge
        import openjarvis.core.config as config

        monkeypatch.setattr(bridge, "get_rust_module", _boom)
        monkeypatch.setattr(config, "DEFAULT_CONFIG_DIR", tmp_path / "state")
        app = _make_app()
        state = config.JarvisConfig()
        state.memory.db_path = str(tmp_path / "state" / "memory.db")
        state.memory.graph_db_path = str(tmp_path / "state" / "graph.db")
        state.security.secret_scanner = False
        app.state.config = state
        return TestClient(app)

    def test_store_search_and_stats_use_python_fallback(self, monkeypatch, tmp_path):
        client = self._client(monkeypatch, tmp_path)
        stored = client.post(
            "/v1/memory/store",
            json={"content": "ACE fallback memory stores useful local notes"},
        )
        searched = client.post("/v1/memory/search", json={"query": "fallback notes"})
        stats = client.get("/v1/memory/stats")

        assert stored.status_code == 200
        assert searched.status_code == 200
        assert "fallback memory" in searched.json()["results"][0]["content"]
        assert stats.status_code == 200
        assert stats.json()["entries"] == 1

    def test_index_works_without_rust_extension(self, monkeypatch, tmp_path):
        documents = tmp_path / "documents"
        documents.mkdir()
        note = documents / "note.txt"
        note.write_text(" ".join(f"memoryword{i}" for i in range(100)))
        client = self._client(monkeypatch, tmp_path)
        resp = client.post("/v1/memory/index", json={"path": str(documents)})
        assert resp.status_code == 200, resp.text
        assert resp.json()["chunks_indexed"] >= 1

    def test_config_reports_available_without_rust(self, monkeypatch, tmp_path):
        client = self._client(monkeypatch, tmp_path)
        resp = client.get("/v1/memory/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert data["backend_type"] == "sqlite"
        assert data["detail"] is None


class TestBudgetRoutes:
    def test_get_budget(self):
        client = TestClient(_make_app())
        resp = client.get("/v1/budget")
        assert resp.status_code == 200
        data = resp.json()
        assert "limits" in data
        assert "usage" in data

    def test_set_limits(self):
        client = TestClient(_make_app())
        resp = client.put("/v1/budget/limits", json={"max_tokens_per_day": 100000})
        assert resp.status_code == 200
        assert resp.json()["limits"]["max_tokens_per_day"] == 100000


class TestMetricsRoute:
    def test_metrics_endpoint(self, tmp_path, monkeypatch):
        """An isolated empty config dir returns the exact no-data response.

        Pinning ``DEFAULT_CONFIG_DIR`` keeps the test independent of the
        machine's real telemetry database and makes any other response a
        regression rather than accepting all possible production states.
        """
        monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))

        client = TestClient(_make_app())
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.text == "# no telemetry data\n"
        assert resp.headers["content-type"].startswith("text/plain")

    def test_metrics_endpoint_reports_real_stats(self, tmp_path, monkeypatch):
        """Regression for #760: /metrics must report actual aggregated
        stats instead of silently falling back to the placeholder text.

        The handler called ``stats.get(...)`` on ``AggregatedStats``, a
        dataclass with no ``.get()`` method -- every real request raised
        ``AttributeError``, caught by the broad ``except Exception`` and
        papered over as "No metrics available", even with real telemetry
        data present in ``telemetry.db``.
        """
        import time

        from openjarvis.core.types import TelemetryRecord
        from openjarvis.telemetry.store import TelemetryStore

        monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))

        db_path = tmp_path / "telemetry.db"
        store = TelemetryStore(db_path)
        store.record(
            TelemetryRecord(
                timestamp=time.time(),
                model_id="test-model",
                engine="ollama",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                latency_seconds=2.0,
                cost_usd=0.001,
            )
        )
        store.close()

        client = TestClient(_make_app())
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "No metrics available" not in resp.text
        assert "openjarvis_requests_total 1" in resp.text
        assert "openjarvis_tokens_total 15" in resp.text
        assert "openjarvis_latency_avg_ms 2000" in resp.text

    def test_metrics_endpoint_empty_db_no_division_by_zero(self, tmp_path, monkeypatch):
        """An existing but empty telemetry.db (total_calls=0) must not
        raise ZeroDivisionError when computing average latency."""
        from openjarvis.telemetry.store import TelemetryStore

        monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))

        db_path = tmp_path / "telemetry.db"
        TelemetryStore(db_path).close()  # creates the schema, no records

        client = TestClient(_make_app())
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "No metrics available" not in resp.text
        assert "openjarvis_requests_total 0" in resp.text
        assert "openjarvis_latency_avg_ms 0" in resp.text

    def test_metrics_endpoint_uses_active_app_telemetry_path(self, tmp_path):
        import time

        from openjarvis.core.types import TelemetryRecord
        from openjarvis.telemetry.store import TelemetryStore

        db_path = tmp_path / "runtime" / "custom-telemetry.sqlite"
        store = TelemetryStore(db_path)
        store.record(
            TelemetryRecord(
                timestamp=time.time(),
                model_id="configured-model",
                engine="ollama",
                prompt_tokens=4,
                completion_tokens=3,
                total_tokens=7,
                latency_seconds=0.5,
                cost_usd=0.0,
            )
        )
        store.close()

        app = _make_app()
        app.state.config = SimpleNamespace(
            telemetry=SimpleNamespace(db_path=str(db_path))
        )
        resp = TestClient(app).get("/metrics")

        assert resp.status_code == 200
        assert "openjarvis_requests_total 1" in resp.text
        assert "openjarvis_tokens_total 7" in resp.text
        assert "openjarvis_latency_avg_ms 500" in resp.text


class TestSkillRoutes:
    def test_list_skills(self):
        client = TestClient(_make_app())
        resp = client.get("/v1/skills")
        assert resp.status_code == 200
        assert "skills" in resp.json()


class TestSessionRoutes:
    def test_list_sessions(self):
        client = TestClient(_make_app())
        resp = client.get("/v1/sessions")
        assert resp.status_code == 200


class TestTraceRoutes:
    def test_list_traces(self):
        client = TestClient(_make_app())
        resp = client.get("/v1/traces")
        assert resp.status_code == 200
