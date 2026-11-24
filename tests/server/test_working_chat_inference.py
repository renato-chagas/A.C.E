"""Exercise shared conversation generation and compare-and-swap persistence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from openjarvis.core.config import JarvisConfig
from openjarvis.core.events import EventBus, EventType
from openjarvis.engine.multi import MultiEngine
from openjarvis.server import working_chat_inference
from openjarvis.server.app import create_app
from openjarvis.sessions import working_chat
from openjarvis.sessions.working_chat import WorkingChatStore


class TextEngine:
    engine_id = "ollama"
    is_local = True
    is_cloud = False

    def __init__(self):
        self.calls = []
        self.before_answer = lambda: None
        self.finish_reason = "stop"

    def list_models(self):
        return ["local-demo"]

    def generate(self, messages, **kwargs):
        self.calls.append(
            [(message.role.value, message.content) for message in messages]
        )
        assert kwargs["model"] == "local-demo"
        assert "tools" not in kwargs
        self.before_answer()
        return {
            "content": "A saved desktop answer.",
            "finish_reason": self.finish_reason,
        }


@pytest.fixture
def shared_chat(tmp_path, monkeypatch):
    monkeypatch.setattr(working_chat, "get_config_dir", lambda: tmp_path)
    store = WorkingChatStore()
    session = store.open("Project planning")
    snapshot = store.append(session, "A terminal question.", "A terminal answer.")
    config = JarvisConfig()
    config.agent.context_from_memory = False
    config.telemetry.enabled = False
    config.analytics.enabled = False
    engine = TextEngine()
    bus = EventBus(record_history=True)
    app = create_app(engine, model="local-demo", config=config, bus=bus)
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 47777)
    ) as client:
        yield app, client, engine, bus, store, snapshot


def _payload(chat, text="Continue from our plan."):
    return {
        "name": chat.name,
        "session_id": chat.session_id,
        "revision": chat.revision,
        "message": text,
    }


def test_desktop_uses_and_extends_terminal_history(shared_chat):
    _, client, engine, bus, store, after_terminal = shared_chat
    response = client.post("/v1/working-chats/turn", json=_payload(after_terminal))
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    body = response.json()
    assert body["saved"]
    assert body["chat"]["revision"] == after_terminal.revision + 1
    assert engine.calls[0][-3:] == [
        ("user", "A terminal question."),
        ("assistant", "A terminal answer."),
        ("user", "Continue from our plan."),
    ]
    resumed_in_terminal = store.open(after_terminal.name)
    assert resumed_in_terminal.turns[-1] == (
        "Continue from our plan.",
        "A saved desktop answer.",
    )
    events = [
        event
        for event in bus.history
        if event.event_type == EventType.CHAT_EXCHANGE_COMPLETED
    ]
    assert len(events) == 1
    assert events[0].data["session_id"] == after_terminal.session_id


def test_turn_with_stale_snapshot_does_not_infer(shared_chat):
    _, client, engine, _, store, snapshot = shared_chat
    store.append(snapshot, "A newer turn.", "The newer reply.")
    reply = client.post("/v1/working-chats/turn", json=_payload(snapshot))
    assert reply.status_code == 409
    assert not engine.calls


def test_concurrent_terminal_write_retains_its_turn(shared_chat):
    _, client, engine, bus, store, snapshot = shared_chat
    engine.before_answer = lambda: store.append(
        snapshot, "Concurrent question.", "Concurrent reply."
    )
    response = client.post("/v1/working-chats/turn", json=_payload(snapshot))
    assert response.status_code == 200
    assert response.json() == {
        "chat": None,
        "answer": "A saved desktop answer.",
        "saved": False,
        "reason": "changed",
    }
    assert store.read(snapshot.name).turns[-1] == (
        "Concurrent question.",
        "Concurrent reply.",
    )
    assert not any(
        event.event_type == EventType.CHAT_EXCHANGE_COMPLETED for event in bus.history
    )


@pytest.mark.parametrize("reason", ["incomplete", "scanner"])
def test_partial_or_flagged_text_is_not_persisted(shared_chat, monkeypatch, reason):
    _, client, engine, bus, store, snapshot = shared_chat
    if reason == "incomplete":
        engine.finish_reason = "length"
    else:
        scanner = SimpleNamespace(scan=lambda _text: SimpleNamespace(is_clean=False))
        monkeypatch.setattr(
            working_chat_inference, "_build_secret_scanner", lambda: scanner
        )

    response = client.post("/v1/working-chats/turn", json=_payload(snapshot))
    assert response.status_code == 200
    assert response.json()["reason"] == reason
    assert not response.json()["saved"]
    assert store.read(snapshot.name) == snapshot
    assert not any(
        event.event_type == EventType.CHAT_EXCHANGE_COMPLETED for event in bus.history
    )


@pytest.mark.parametrize("route", ["remote", "mixed"])
def test_private_history_does_not_enter_unverified_routing(shared_chat, route):
    app, client, engine, _, _, snapshot = shared_chat
    remote = TextEngine()
    remote._host = "http://remote.invalid:1234"
    if route == "remote":
        app.state.engine = remote
    else:
        remote.list_models = lambda: ["other-model"]
        app.state.engine = MultiEngine([("ollama", engine), ("remote", remote)])

    response = client.post("/v1/working-chats/turn", json=_payload(snapshot))
    assert response.status_code == 503
    assert engine.calls == remote.calls == []


def test_router_with_only_local_engines_can_resume_the_chat(shared_chat):
    app, client, engine, _, store, snapshot = shared_chat
    app.state.engine = MultiEngine([("ollama", engine)])
    response = client.post("/v1/working-chats/turn", json=_payload(snapshot))
    assert response.status_code == 200
    assert response.json()["saved"]
    assert len(engine.calls) == 1
    assert store.read(snapshot.name).revision == snapshot.revision + 1


def test_invalid_input_does_not_echo_private_text_or_infer(shared_chat):
    _, client, engine, _, _, snapshot = shared_chat
    message = "private input " * 600
    response = client.post("/v1/working-chats/turn", json=_payload(snapshot, message))
    assert response.status_code == 422
    assert "private input" not in response.text
    assert not engine.calls
