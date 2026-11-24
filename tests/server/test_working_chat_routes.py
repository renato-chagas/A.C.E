"""Shared working chats through the normal and no-inference APIs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from openjarvis.core.config import JarvisConfig
from openjarvis.server.app import create_app
from openjarvis.server.local_services import create_local_services_app
from openjarvis.sessions import working_chat
from openjarvis.sessions.working_chat import WorkingChatStore

_PATH = "/v1/working-chats"
_WEBVIEW = {"Origin": "http://tauri.localhost"}


@pytest.fixture(params=["normal", "services"])
def app_and_store(request, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))
    path = tmp_path / "working_chats.db"
    monkeypatch.setattr(working_chat, "working_chat_path", lambda: path)
    config = JarvisConfig()
    if request.param == "services":
        return create_local_services_app(config), path
    return create_app(object(), "test-model", config=config), path


def _client(app, peer: str = "127.0.0.1") -> TestClient:
    return TestClient(app, base_url="http://localhost", client=(peer, 50000))


def _assert_uncached(response, status: int) -> None:
    assert response.status_code == status, response.text
    assert response.headers.get("Cache-Control") == "no-store"


def test_reads_do_not_create_and_open_shares_terminal_history(app_and_store):
    app, path = app_and_store
    with _client(app) as client:
        response = client.get(_PATH, headers=_WEBVIEW)
        _assert_uncached(response, 200)
        assert response.headers["access-control-allow-origin"] == _WEBVIEW["Origin"]
        assert response.json() == {"chats": []}
        _assert_uncached(client.post(_PATH + "/read", json={"name": "ACE"}), 404)
        assert not path.exists()

        response = client.post(_PATH + "/open", json={"name": " ACE "})
        _assert_uncached(response, 200)
        assert response.json() == {
            "name": "ACE",
            "session_id": response.json()["session_id"],
            "revision": 0,
            "turns": [],
        }
        store = WorkingChatStore(path)
        chat = store.read("ACE")
        assert chat is not None
        changed = store.append(chat, "Question from CLI", "Reply from CLI")

        response = client.post(_PATH + "/read", json={"name": "ACE"})
        _assert_uncached(response, 200)
        assert response.json() == {
            "name": "ACE",
            "session_id": changed.session_id,
            "revision": changed.revision,
            "turns": [["Question from CLI", "Reply from CLI"]],
        }
        assert (
            client.post(_PATH + "/open", json={"name": "ACE"}).json() == response.json()
        )
        assert client.get(_PATH).json() == {"chats": ["ACE"]}


def test_forget_rejects_stale_revision_and_recreated_sessions(app_and_store):
    app, path = app_and_store
    store = WorkingChatStore(path)
    original = store.open("Shared")
    changed = store.append(original, "Question", "Reply")
    with _client(app) as client:

        def forget(chat):
            return client.post(
                _PATH + "/forget",
                json={
                    "name": chat.name,
                    "session_id": chat.session_id,
                    "revision": chat.revision,
                },
            )

        _assert_uncached(forget(original), 409)
        assert store.read("Shared") == changed
        del_response = forget(changed)
        _assert_uncached(del_response, 200)
        assert del_response.json() == {"deleted": True}
        assert store.read("Shared") is None

        recreated = store.open("Shared")
        _assert_uncached(forget(original), 409)
        assert store.read("Shared") == recreated


@pytest.mark.parametrize(
    ("peer", "headers"),
    [
        ("192.0.2.15", {}),
        ("127.0.0.1", {"Host": "public.example"}),
        ("127.0.0.1", {"Origin": "https://public.example"}),
        ("127.0.0.1", {"Forwarded": "for=127.0.0.1"}),
    ],
)
def test_proxy_or_external_request_does_not_parse_or_create(
    app_and_store,
    peer,
    headers,
):
    app, path = app_and_store
    with _client(app, peer) as client:
        response = client.post(
            _PATH + "/open",
            content='{"name": "secret-name"',
            headers={**headers, "Content-Type": "application/json"},
        )
        _assert_uncached(response, 403)
        assert "secret-name" not in response.text
    assert not path.exists()


@pytest.mark.parametrize(
    ("action", "payload"),
    [
        ("open", {"name": 123}),
        ("open", {"name": "a\nb"}),
        ("open", {"name": "private", "text": "user-secret"}),
        ("forget", {"name": "private", "session_id": "bad", "revision": 0}),
        ("forget", {"name": "private", "session_id": "a" * 32, "revision": True}),
        ("forget", {"name": "private", "session_id": "a" * 32, "revision": "1"}),
    ],
)
def test_invalid_payloads_do_not_echo_content(app_and_store, action, payload):
    app, path = app_and_store
    with _client(app) as client:
        response = client.post(f"{_PATH}/{action}", json=payload)
        _assert_uncached(response, 422)
        assert response.json() == {"detail": "Invalid working chat request."}
    assert not path.exists()


@pytest.mark.parametrize("streamed", [False, True])
def test_bodies_are_bounded_before_json_validation(app_and_store, streamed):
    app, path = app_and_store
    payload = b'{"name":"private","padding":"' + b"x" * 32768 + b'"}'
    with _client(app) as client:
        response = client.post(
            _PATH + "/open",
            content=iter([payload[:2048], payload[2048:]]) if streamed else payload,
            headers={"Content-Type": "application/json"},
        )
        _assert_uncached(response, 413)
        assert response.json() == {"detail": "Working chat request is too large."}
    assert not path.exists()


def test_bad_store_returns_generic_error(app_and_store):
    app, path = app_and_store
    path.write_text("not a database: do not echo me", encoding="utf-8")
    with _client(app) as client:
        response = client.post(_PATH + "/read", json={"name": "ACE"})
        _assert_uncached(response, 503)
        assert response.json() == {"detail": "Working chat storage is unavailable."}


def test_local_services_describes_inference_unavailability():
    with _client(create_local_services_app(JarvisConfig())) as client:
        response = client.post(_PATH + "/turn", json={"message": "not saved"})
        _assert_uncached(response, 503)
        assert "Inference is unavailable" in response.json()["detail"]
