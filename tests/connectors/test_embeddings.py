"""Tests for openjarvis.connectors.embeddings.default_embedder."""

from __future__ import annotations

import pytest
import requests

from openjarvis.connectors import embeddings


def test_default_embedder_returns_instance_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embeddings.OllamaEmbedder, "is_available", lambda self: True)
    emb = embeddings.default_embedder()
    assert isinstance(emb, embeddings.OllamaEmbedder)
    assert emb.model_version == f"ollama:{embeddings.DEFAULT_EMBED_MODEL}"


def test_default_embedder_uses_configured_host(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class AvailableEmbedder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def is_available(self):
            return True

    monkeypatch.setattr(embeddings, "OllamaEmbedder", AvailableEmbedder)

    assert (
        embeddings.default_embedder(local_only=True, host="http://127.0.0.1:12434")
        is not None
    )
    assert captured["host"] == "http://127.0.0.1:12434"
    assert captured["trust_env"] is False


def test_default_embedder_returns_none_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embeddings.OllamaEmbedder, "is_available", lambda self: False)
    assert embeddings.default_embedder() is None


def test_local_only_default_embedder_rejects_remote_ollama_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "https://remote.example:11434")
    monkeypatch.setattr(
        embeddings.OllamaEmbedder,
        "is_available",
        lambda self: pytest.fail("remote host must not be probed"),
    )

    assert embeddings.default_embedder(local_only=True) is None


def test_local_only_default_embedder_disables_proxies_and_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.setattr(embeddings.OllamaEmbedder, "is_available", lambda self: True)
    embedder = embeddings.default_embedder(local_only=True)
    assert embedder is not None
    try:
        assert embedder._trust_env is False
        assert embedder._session.trust_env is False
        assert embedder._allow_redirects is False
    finally:
        embedder._session.close()


def test_embedder_host_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit host > OLLAMA_HOST env > built-in default, like OllamaEngine."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert embeddings.OllamaEmbedder()._host == embeddings.DEFAULT_OLLAMA_HOST

    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434/")
    assert embeddings.OllamaEmbedder()._host == "http://127.0.0.1:11434"

    explicit = embeddings.OllamaEmbedder(host="http://gpu-box:11434")
    assert explicit._host == "http://gpu-box:11434"


def test_embedder_can_ignore_environment_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    embedder = embeddings.OllamaEmbedder(trust_env=False)
    try:
        assert embedder._session.trust_env is False
    finally:
        embedder._session.close()


def test_embedder_disables_proxy_for_loopback_host_by_default() -> None:
    local = embeddings.OllamaEmbedder(host="http://127.0.0.1:11434")
    remote = embeddings.OllamaEmbedder(host="http://gpu-box:11434")
    try:
        assert local._session.trust_env is False
        assert remote._session.trust_env is True
    finally:
        local._session.close()
        remote._session.close()


class _FakeResponse:
    def __init__(self, payload=None, *, status_code=200):
        self._payload = payload or {}
        self.status_code = status_code
        self.is_redirect = False

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self):
        return self._payload


class _FakeSession:
    trust_env = False

    def __init__(self, responder):
        self._responder = responder
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self._responder(url, kwargs)


def test_embed_batch_uses_chunked_ollama_embed_endpoint():
    session = _FakeSession(
        lambda _url, kwargs: _FakeResponse(
            {
                "embeddings": [
                    [float(index), 1.0]
                    for index, _text in enumerate(kwargs["json"]["input"])
                ]
            }
        )
    )
    embedder = embeddings.OllamaEmbedder()
    embedder._session = session

    vectors = embedder.embed_batch(["first", "", "second", "third", "fourth"])

    assert len(session.calls) == 1
    assert session.calls[0][0].endswith("/api/embed")
    assert session.calls[0][1]["json"]["input"] == [
        "first",
        "second",
        "third",
        "fourth",
    ]
    assert vectors[1] is None
    assert all(vector is not None for index, vector in enumerate(vectors) if index != 1)
    assert embedder.dim == 2


def test_embed_batch_falls_back_to_legacy_endpoint_when_unsupported():
    def respond(url, kwargs):
        if url.endswith("/api/embed"):
            return _FakeResponse(status_code=404)
        text = kwargs["json"]["prompt"]
        return _FakeResponse({"embedding": [float(len(text)), 1.0]})

    session = _FakeSession(respond)
    embedder = embeddings.OllamaEmbedder()
    embedder._session = session

    vectors = embedder.embed_batch(["one", "two"])

    assert len(session.calls) == 3
    assert all(vector is not None for vector in vectors)


def test_embed_batch_network_failure_does_not_fan_out_single_requests():
    def fail(_url, _kwargs):
        raise requests.ConnectionError("local runtime unavailable")

    session = _FakeSession(fail)
    embedder = embeddings.OllamaEmbedder()
    embedder._session = session

    vectors = embedder.embed_batch(["one", "two", "three"])

    assert len(session.calls) == 1
    assert vectors == [None, None, None]


def test_embed_batch_chunks_large_inputs_at_fixed_limit():
    batch_sizes = []

    def respond(_url, kwargs):
        count = len(kwargs["json"]["input"])
        batch_sizes.append(count)
        return _FakeResponse({"embeddings": [[1.0, 0.0]] * count})

    session = _FakeSession(respond)
    embedder = embeddings.OllamaEmbedder()
    embedder._session = session

    vectors = embedder.embed_batch([f"fact {index}" for index in range(35)])

    assert batch_sizes == [32, 3]
    assert len(vectors) == 35
    assert all(vector is not None for vector in vectors)


def test_episode_embedder_ignores_proxy_and_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openjarvis.memory.episodes as episodes

    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    episodes._EMBEDDER_CACHE.clear()
    monkeypatch.setattr(embeddings.OllamaEmbedder, "is_available", lambda self: True)
    embedder = episodes._local_embedder("local-test-model")
    assert embedder is not None
    try:
        assert embedder._session.trust_env is False
        assert embedder._allow_redirects is False
    finally:
        embedder._session.close()
        episodes._EMBEDDER_CACHE.clear()
