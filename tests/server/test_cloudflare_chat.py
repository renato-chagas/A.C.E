"""Check explicit Workers AI setup through the normal ACE conversation route."""

from __future__ import annotations

import importlib
import json

import httpx
import pytest
from click.testing import CliRunner

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
from fastapi.testclient import TestClient

from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import EngineRegistry
from openjarvis.engine.cloudflare import CLOUDFLARE_MODEL_ID, CloudflareEngine
from openjarvis.server import daemon

serve_module = importlib.import_module("openjarvis.cli.serve")
_ACCOUNT = "0" * 32
_TOKEN = "fake-token-for-ace-fixtures"
_REPLY = "Olá! A solução está pronta."
_REASONING = "hidden-reasoning-fixture"
_CHAT = "/v1/chat/completions"


@pytest.fixture
def packaged_chat(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", _ACCOUNT)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", _TOKEN)
    monkeypatch.setattr(serve_module, "_load_server_api_key", lambda: "")
    config = JarvisConfig()
    config.intelligence.provider = "local"
    # This local profile data is private, even when the user opts into chat.
    (tmp_path / "USER.md").write_text("secret-local-profile-fixture", encoding="utf-8")
    monkeypatch.setattr(serve_module, "load_config", lambda: config)
    EngineRegistry.register_value("cloudflare", CloudflareEngine)

    sync_client = httpx.Client
    async_client = httpx.AsyncClient
    received = []
    upstream = {"status": 200}

    def respond(request):
        received.append(request)
        assert request.url.host == "api.cloudflare.com"
        assert request.url.path == (
            f"/client/v4/accounts/{_ACCOUNT}/ai/run/{CLOUDFLARE_MODEL_ID}"
        )
        assert request.headers["Authorization"] == f"Bearer {_TOKEN}"
        if upstream["status"] != 200:
            body = {"success": False, "errors": [{"message": _TOKEN}]}
        else:
            body = {
                "success": True,
                "result": {
                    "choices": [
                        {
                            "message": {
                                "content": _REPLY,
                                "reasoning_content": _REASONING,
                            }
                        }
                    ]
                },
            }
        return httpx.Response(
            upstream["status"],
            headers={"Content-Type": "application/json"},
            content=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        )

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: sync_client(transport=transport, **kw)
    )
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: async_client(transport=transport, **kw)
    )

    started = []
    monkeypatch.setattr(
        daemon, "run_server", lambda app, **kw: started.append((app, kw))
    )
    return config, received, upstream, started


def test_selected_cloudflare_serves_normal_chat_and_keeps_profile_local(
    packaged_chat, monkeypatch, caplog
):
    local, received, _, started = packaged_chat

    def deny(*args, **kwargs):
        pytest.fail("unexpected direct cloud routing")

    from openjarvis.server import cloud_router
    from openjarvis.server.cloud_router import stream_cloud

    monkeypatch.setattr(cloud_router, "stream_cloud", deny)
    assert stream_cloud is not deny
    outcome = CliRunner().invoke(serve_module.serve, ["--engine", "cloudflare"])
    assert outcome.exit_code == 0, outcome.output
    app, server_options = started[0]
    assert server_options["host"] == "127.0.0.1"
    assert local.intelligence.provider == "local"
    assert local.agent.context_from_memory is True
    import os

    assert "CLOUDFLARE_API_TOKEN" not in os.environ
    assert "CLOUDFLARE_ACCOUNT_ID" not in os.environ

    conversation = [
        {"role": "user", "content": "Responda em português."},
        {"role": "assistant", "content": "Posso ajudar."},
        {"role": "user", "content": "A solução está pronta?"},
    ]
    payload = {"model": CLOUDFLARE_MODEL_ID, "messages": conversation, "max_tokens": 48}
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 51264)
    ) as client:
        health = client.get("/health")
        assert health.json() == {"status": "ok"}
        assert health.headers["x-ace-inference-source"] == "cloudflare"
        assert client.get("/v1/info").json()["engine"] == "cloudflare"
        assert client.get("/v1/models").json()["data"][0]["id"] == CLOUDFLARE_MODEL_ID
        assert not received

        blocked = client.post(
            _CHAT, json=payload, headers={"Origin": "https://outside.test"}
        )
        assert blocked.status_code == 403
        assert (
            client.post(_CHAT, json={**payload, "model": "unselected"}).status_code
            == 400
        )
        assert (
            client.post(
                _CHAT, json={**payload, "tools": [{"type": "function"}]}
            ).status_code
            == 400
        )
        assert not received

        response = client.post(_CHAT, json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == _REPLY
        assert _REASONING not in response.text
        stream = client.post(_CHAT, json={**payload, "stream": True})
        assert stream.status_code == 200
        chunks = [
            json.loads(line[6:])
            for line in stream.text.splitlines()
            if line.startswith("data: {")
        ]
        assert (
            "".join(
                chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks
            )
            == _REPLY
        )
        assert "data: [DONE]" in stream.text
        assert _REASONING not in stream.text

    assert len(received) == 2
    for request in received:
        content = json.loads(request.content)
        assert content["messages"][1:] == conversation
        assert content["messages"][0]["role"] == "system"
        assert "secret-local-profile-fixture" not in request.content.decode()
        assert content["max_completion_tokens"] == 48
        assert content["store"] is False
        assert content["stream"] is False
    assert _TOKEN not in caplog.text


@pytest.mark.parametrize(
    ("upstream_status", "api_status"), [(401, 403), (429, 429), (500, 502)]
)
def test_upstream_failures_are_safe_in_nonstream_and_ui_stream(
    packaged_chat, upstream_status, api_status, caplog
):
    _, _, upstream, started = packaged_chat
    upstream["status"] = upstream_status
    assert (
        CliRunner().invoke(serve_module.serve, ["--engine", "cloudflare"]).exit_code
        == 0
    )
    payload = {
        "model": CLOUDFLARE_MODEL_ID,
        "messages": [{"role": "user", "content": "Olá"}],
    }
    with TestClient(
        started[0][0], base_url="http://127.0.0.1", client=("127.0.0.1", 51264)
    ) as client:
        response = client.post(_CHAT, json=payload)
        assert response.status_code == api_status
        assert "Cloudflare" in response.json()["detail"]
        stream = client.post(_CHAT, json={**payload, "stream": True})
        assert response.json()["detail"] in stream.text
        assert _TOKEN not in stream.text
        assert _TOKEN not in response.text
    assert _TOKEN not in caplog.text


def test_local_services_do_not_load_cloudflare_even_with_env(packaged_chat):
    config, received, _, started = packaged_chat
    config.intelligence.provider = "cloudflare"
    outcome = CliRunner().invoke(serve_module.serve, ["--local-services-only"])
    assert outcome.exit_code == 0, outcome.output
    with TestClient(
        started[0][0], base_url="http://127.0.0.1", client=("127.0.0.1", 51264)
    ) as client:
        response = client.get("/health")
        assert response.json()["inference_available"] is False
        assert "x-ace-inference-source" not in response.headers
        assert (
            client.post(_CHAT, json={"model": CLOUDFLARE_MODEL_ID}).status_code == 503
        )
    assert not received


def test_local_inference_only_rejects_explicit_cloud_selection(packaged_chat):
    _, received, _, started = packaged_chat
    outcome = CliRunner().invoke(
        serve_module.serve, ["--local-inference-only", "--engine", "cloudflare"]
    )
    assert outcome.exit_code == 2
    assert "seleção online explícita" in outcome.output
    assert not received and not started


def test_cloudflare_requires_env_and_refuses_lan_bind(packaged_chat, monkeypatch):
    _, received, _, started = packaged_chat
    lan = CliRunner().invoke(
        serve_module.serve, ["--engine", "cloudflare", "--host", "0.0.0.0"]
    )
    assert lan.exit_code == 2 and "loopback" in lan.output
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    missing = CliRunner().invoke(serve_module.serve, ["--engine", "cloudflare"])
    assert missing.exit_code == 1
    assert "CLOUDFLARE_ACCOUNT_ID" in missing.output
    assert "CLOUDFLARE_API_TOKEN" in missing.output
    assert not received and not started
