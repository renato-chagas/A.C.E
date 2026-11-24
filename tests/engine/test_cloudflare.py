"""Fixture-only coverage for fixed-endpoint Workers AI inference."""

from __future__ import annotations

import json

import httpx
import pytest

import openjarvis.engine._discovery as discovery
from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import EngineRegistry
from openjarvis.core.types import Message, Role
from openjarvis.engine.cloudflare import (
    CLOUDFLARE_MODEL_ID,
    CloudflareEngine,
    CloudflareInferenceError,
)

_ACCOUNT = "0" * 32
_TOKEN = "fixture-only-token"
_PATH = f"/client/v4/accounts/{_ACCOUNT}/ai/run/{CLOUDFLARE_MODEL_ID}"
_MESSAGES = [Message(Role.USER, "Você pode falar português?")]


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", _ACCOUNT)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", _TOKEN)
    return CloudflareEngine()


@pytest.fixture
def transport_factory(monkeypatch):
    sync_client, async_client = httpx.Client, httpx.AsyncClient

    def install(handler):
        transport = httpx.MockTransport(handler)

        def options(kwargs):
            assert kwargs["base_url"] == "https://api.cloudflare.com"
            assert kwargs["follow_redirects"] is False
            assert kwargs["trust_env"] is False
            timeout = kwargs["timeout"].as_dict()
            assert all(0 < seconds <= 90 for seconds in timeout.values())
            return {**kwargs, "transport": transport}

        monkeypatch.setattr(
            httpx, "Client", lambda **kwargs: sync_client(**options(kwargs))
        )
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kwargs: async_client(**options(kwargs))
        )

    return install


@pytest.mark.asyncio
async def test_sync_and_async_inference_use_exact_json_endpoint(
    configured, monkeypatch, transport_factory
):
    calls = []

    def answer(request):
        calls.append(request)
        assert request.method == "POST"
        assert request.url == f"https://api.cloudflare.com{_PATH}"
        assert request.headers["authorization"] == f"Bearer {_TOKEN}"
        content = request.content.decode("utf-8")
        assert "português" in content
        assert json.loads(content) == {
            "messages": [{"role": "user", "content": _MESSAGES[0].text}],
            "temperature": 0.7,
            "max_completion_tokens": 1024,
            "stream": False,
            "store": False,
        }
        return httpx.Response(
            200,
            json={
                "success": True,
                "result": {
                    "response": "Wrong response field",
                    "choices": [
                        {
                            "message": {
                                "content": "Sim, falo português.",
                                "reasoning_content": "Private deliberation",
                                "tool_calls": [{"id": "not-allowed"}],
                            }
                        },
                        {"message": {"content": "Wrong alternate choice"}},
                    ],
                },
            },
        )

    transport_factory(answer)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID")
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    assert configured.health() and configured.can_serve(CLOUDFLARE_MODEL_ID)
    assert configured.list_models() == [CLOUDFLARE_MODEL_ID]
    assert calls == []
    reply = configured.generate(_MESSAGES, model=CLOUDFLARE_MODEL_ID)
    assert reply == {
        "content": "Sim, falo português.",
        "usage": {},
        "model": CLOUDFLARE_MODEL_ID,
    }
    assert [
        fragment
        async for fragment in configured.stream(_MESSAGES, model=CLOUDFLARE_MODEL_ID)
    ] == [reply["content"]]
    full = [
        chunk
        async for chunk in configured.stream_full(_MESSAGES, model=CLOUDFLARE_MODEL_ID)
    ]
    assert full[0].content == reply["content"]
    assert full[0].tool_calls is None
    assert full[1].finish_reason == "stop"
    assert full[1].finish_reason_reported is False
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("status", "expected"),
    [(302, 502), (401, 403), (402, 429), (403, 403), (429, 429), (500, 502)],
)
def test_vendor_failures_never_expose_bodies_or_follow_redirects(
    configured, transport_factory, status, expected
):
    attempts = []

    def answer(request):
        attempts.append(request.url)
        return httpx.Response(
            status,
            headers={"Location": "https://redirect.invalid"},
            text=f"Sensitive response containing {_TOKEN}",
        )

    transport_factory(answer)
    with pytest.raises(CloudflareInferenceError) as caught:
        configured.generate(_MESSAGES, model=CLOUDFLARE_MODEL_ID)
    assert caught.value.status_code == expected
    assert _TOKEN not in str(caught.value)
    assert "Sensitive response" not in str(caught.value)
    assert attempts == [httpx.URL(f"https://api.cloudflare.com{_PATH}")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"), [(httpx.ReadTimeout, 504), (httpx.ConnectError, 502)]
)
async def test_transport_failure_is_sanitized(
    configured, transport_factory, error, expected
):
    def fail(request):
        raise error(f"Sensitive detail containing {_TOKEN}", request=request)

    transport_factory(fail)
    with pytest.raises(CloudflareInferenceError) as caught:
        async for _ in configured.stream(_MESSAGES, model=CLOUDFLARE_MODEL_ID):
            pytest.fail("a failed request yielded text")
    assert caught.value.status_code == expected
    assert caught.value.__suppress_context__
    assert _TOKEN not in str(caught.value)


@pytest.mark.parametrize(
    "result",
    [
        {"response": _TOKEN},
        {"choices": []},
        {"choices": [{"message": {"reasoning_content": _TOKEN}}]},
        {"choices": [{"message": {"content": " \n "}}]},
        {"choices": [{"message": {"content": [{"text": _TOKEN}]}}]},
    ],
)
def test_only_the_exact_first_text_choice_is_accepted(
    configured, transport_factory, result
):
    transport_factory(lambda _request: httpx.Response(200, json={"result": result}))
    with pytest.raises(CloudflareInferenceError) as caught:
        configured.generate(_MESSAGES, model=CLOUDFLARE_MODEL_ID)
    assert caught.value.status_code == 502
    assert _TOKEN not in str(caught.value)


@pytest.mark.asyncio
async def test_alternative_models_use_cloudflare_chat_completion(
    configured, transport_factory
):
    selected = "@cf/openai/gpt-oss-20b"

    def answer(request):
        expected_path = f"/client/v4/accounts/{_ACCOUNT}/ai/v1/chat/completions"
        assert request.url.path == expected_path
        assert request.headers["authorization"] == f"Bearer {_TOKEN}"
        assert json.loads(request.content) == {
            "model": selected,
            "messages": [{"role": "user", "content": _MESSAGES[0].text}],
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False,
            "store": False,
        }
        return httpx.Response(
            200, json={"choices": [{"message": {
                "content": "Até logo!", "reasoning_content": "Fixture only",
            }}]}
        )

    transport_factory(answer)
    assert configured.can_serve(selected)
    reply = configured.generate(_MESSAGES, model=selected)
    assert reply["model"] == selected and reply["content"] == "Até logo!"
    assert [
        text async for text in configured.stream(_MESSAGES, model=selected)
    ] == ["Até logo!"]


def test_unapproved_inputs_and_missing_credential_are_rejected_before_http(
    configured, monkeypatch, transport_factory
):
    def unexpected_request(_request):
        pytest.fail("unexpected Cloudflare request")

    transport_factory(unexpected_request)
    for model in ("alternate/model", "@cf/qwen/../escape", "@cf/qwen/foo?", ""):
        with pytest.raises(CloudflareInferenceError) as wrong_model:
            configured.generate(_MESSAGES, model=model)
        assert wrong_model.value.status_code == 400
    with pytest.raises(CloudflareInferenceError) as tools:
        configured.generate(
            _MESSAGES, model=CLOUDFLARE_MODEL_ID, tools=[{"type": "function"}]
        )
    assert tools.value.status_code == 400
    with pytest.raises(CloudflareInferenceError) as images:
        configured.generate(
            [Message(Role.USER, "Image", images=["fixture"])],
            model=CLOUDFLARE_MODEL_ID,
        )
    assert images.value.status_code == 400

    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    missing = CloudflareEngine()
    assert not missing.health()
    assert missing.list_models() == []
    with pytest.raises(CloudflareInferenceError) as missing_token:
        missing.generate(_MESSAGES, model=CLOUDFLARE_MODEL_ID)
    assert missing_token.value.status_code == 403

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "../../alternate")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", _TOKEN)
    assert not CloudflareEngine().health()


def test_cloudflare_discovery_requires_explicit_choice(configured, monkeypatch):
    EngineRegistry.register_value("cloudflare", CloudflareEngine)
    monkeypatch.setattr(
        discovery, "_maybe_register_mining_sidecar_engine", lambda: None
    )
    config = JarvisConfig()
    config.engine.default = config.intelligence.preferred_engine = "cloudflare"
    for provider in ("", "local", "openai"):
        config.intelligence.provider = provider
        assert discovery.discover_engines(config) == []
        assert discovery.get_configured_engine(config, pin_default=True) is None
    config.intelligence.provider = "local"
    assert discovery.get_engine(config, "cloudflare") is None

    config.intelligence.provider = "openai"
    explicit = discovery.get_engine(config, "cloudflare", CLOUDFLARE_MODEL_ID)
    assert explicit is not None and explicit[0] == "cloudflare"

    class UnexpectedEngine:
        def __init__(self, **_kwargs):
            pytest.fail("explicit Cloudflare profile probed an unrelated engine")

    EngineRegistry.register_value("ollama", UnexpectedEngine)
    config.engine.default = config.intelligence.preferred_engine = "ollama"
    config.intelligence.provider = "cloudflare"
    assert discovery.discover_models(discovery.discover_engines(config)) == {
        "cloudflare": [CLOUDFLARE_MODEL_ID]
    }
    selected = discovery.get_configured_engine(config, pin_default=True)
    assert selected is not None and selected[0] == "cloudflare"
    assert (
        discovery.get_engine(
            config, model="other/model", fallback_models=[CLOUDFLARE_MODEL_ID]
        )
        is None
    )
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    assert discovery.get_engine(config) is None
