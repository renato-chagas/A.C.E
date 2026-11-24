from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from openjarvis.connectors.store import KnowledgeStore
from openjarvis.core.config import JarvisConfig
from openjarvis.memory.store import LocalFactStore
from openjarvis.server import cloud_router, routes, upload_router
from openjarvis.server.local_services import create_local_services_app

_WEBVIEW = {"Origin": "http://tauri.localhost"}


def _client(app, peer: str = "127.0.0.1") -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1", client=(peer, 50000))


def test_local_services_preserve_storage_without_an_inference_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    config = JarvisConfig()
    config.memory.db_path = str(tmp_path / "memory.db")
    config.memory.graph_db_path = str(tmp_path / "knowledge_graph.db")
    config.memory.facts_path = str(tmp_path / "facts.jsonl")
    store = LocalFactStore(config.memory.facts_path)
    assert store.add("ACE can remember offline.")
    monkeypatch.setattr(upload_router, "DEFAULT_CONFIG_DIR", tmp_path)
    app = create_local_services_app(config)

    with _client(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "mode": "local_services_only",
            "inference_available": False,
        }
        models = client.get("/v1/models")
        assert models.json()["data"] == []

        store_response = client.post(
            "/v1/memory/store",
            json={"content": "Oak is a local search fixture."},
            headers=_WEBVIEW,
        )
        assert store_response.status_code == 200, store_response.text
        data = client.post("/v1/memory/search", json={"query": "oak"}).json()
        assert data["results"][0]["content"] == "Oak is a local search fixture."
        assert client.get("/v1/memory/stats").json()["entries"] == 1

        facts = client.get("/v1/memory/facts", headers=_WEBVIEW)
        assert facts.headers["access-control-allow-origin"] == "http://tauri.localhost"
        assert facts.headers["cache-control"] == "no-store"
        assert facts.json()["facts"][0]["text"] == "ACE can remember offline."

        entity = client.post(
            "/v1/memory/graph/entities",
            json={"entity_id": "project:ace", "name": "ACE", "entity_type": "project"},
        )
        assert entity.status_code == 200, entity.text
        graph = client.get("/v1/memory/graph").json()
        assert graph["entities"][0]["name"] == "ACE"

        upload = client.post(
            "/v1/connectors/upload/ingest",
            json={"title": "Offline note", "content": "Ace stores a document."},
        )
        assert upload.status_code == 200, upload.text
        assert upload.json()["chunks_added"] == 1

    with KnowledgeStore(tmp_path / "knowledge.db") as knowledge:
        assert knowledge.count() == 1
    with pytest.raises(RuntimeError, match="memory backend is closed"):
        app.state.memory_backend.count()


def test_local_services_do_not_expose_external_or_automated_routes() -> None:
    with _client(create_local_services_app(JarvisConfig())) as client:
        for path in ("/v1/chat/completions", "/api/research", "/v1/speech/transcribe"):
            response = client.post(path, json={})
            assert response.status_code == 503
            assert "Inference is unavailable" in response.json()["detail"]
        assert client.post("/v1/memory/index", json={}).status_code == 503

        for path in (
            "/v1/models/pull",
            "/v1/cloud/reload",
            "/v1/managed-agents",
            "/v1/channels/send",
            "/v1/connectors/gmail/sync",
        ):
            assert client.post(path, json={}).status_code == 404


@pytest.mark.parametrize(
    ("peer", "headers"),
    [
        ("192.0.2.10", {}),
        ("127.0.0.1", {"Host": "external.example"}),
        ("127.0.0.1", {"Origin": "https://external.example"}),
        ("127.0.0.1", {"Origin": "http://127.0.0.1:9999"}),
        ("127.0.0.1", {"X-Forwarded-For": "127.0.0.1"}),
    ],
)
def test_non_local_requests_are_rejected_before_storage(peer, headers) -> None:
    app = create_local_services_app(JarvisConfig())
    with _client(app, peer=peer) as client:
        response = client.post(
            "/v1/memory/store", json={"content": "x"}, headers=headers
        )
        assert response.status_code == 403
        assert getattr(app.state, "memory_backend", None) is None


def test_configured_api_key_is_required_while_health_and_preflight_work() -> None:
    app = create_local_services_app(JarvisConfig(), api_key="test-key")
    with _client(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/info").status_code == 401
        response = client.options(
            "/v1/info",
            headers={
                **_WEBVIEW,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )
        assert response.status_code == 200
        response = client.get(
            "/v1/info", headers={**_WEBVIEW, "Authorization": "Bearer test-key"}
        )
        assert response.status_code == 200
        assert response.json()["inference_available"] is False


def test_local_inference_profile_keeps_cloud_named_stream_on_its_engine(
    monkeypatch,
) -> None:
    config = JarvisConfig()
    config.intelligence.provider = "local"
    config.agent.context_from_memory = False

    async def local_stream(messages, *, model, **kwargs):
        assert model == "gpt-local-fixture"
        yield "streamed locally"

    remote_stream = Mock(side_effect=AssertionError("unexpected cloud stream"))
    fallback_listing = AsyncMock(side_effect=AssertionError("unexpected fallback"))
    monkeypatch.setattr(cloud_router, "stream_cloud", remote_stream)
    monkeypatch.setattr(cloud_router, "list_local_models", fallback_listing)

    app = FastAPI()
    app.state.config = config
    app.state.engine = SimpleNamespace(
        engine_id="ollama",
        stream=local_stream,
        list_models=lambda: ["gpt-local-fixture"],
    )
    app.include_router(routes.router)

    with _client(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-local-fixture",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "streamed locally" in response.text
        response = client.get("/v1/models")
        assert [model["id"] for model in response.json()["data"]] == [
            "gpt-local-fixture"
        ]
        assert client.post("/v1/cloud/reload", json={}).status_code == 403
    fallback_listing.assert_not_called()
    remote_stream.assert_not_called()
