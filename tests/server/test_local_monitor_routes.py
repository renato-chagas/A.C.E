"""Tests for the loopback-only local resource monitor controls."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from openjarvis.core.config import JarvisConfig
from openjarvis.proactive.desktop_monitor import DesktopMonitorConflict
from openjarvis.proactive.local_monitor import LocalMonitorStore
from openjarvis.server.app import create_app
from openjarvis.server.local_services import create_local_services_app

_ROUTE = "/v1/local-monitor"
_AUTH = {"Authorization": "Bearer local-test-key"}
_WEBVIEW = {"Origin": "http://tauri.localhost"}
_DATABASE = "proactive_events.db"


class FakeController:
    def __init__(self) -> None:
        self.enabled = False
        self.running = False
        self.external = False
        self.notifications_enabled = False
        self.calls: list[tuple[bool, bool]] = []

    def status(self) -> tuple[bool, bool, bool, bool]:
        return self.enabled, self.running, self.external, self.notifications_enabled

    def set_enabled(
        self, enabled: bool, *, notifications_enabled: bool = False
    ) -> None:
        if self.external:
            raise DesktopMonitorConflict("private conflict detail")
        self.calls.append((enabled, notifications_enabled))
        self.enabled = enabled
        self.running = enabled
        self.notifications_enabled = notifications_enabled


@pytest.fixture(params=["normal", "local_services"])
def monitor(request, monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))
    config = JarvisConfig()
    config.security.enabled = False
    app = (
        create_app(object(), "model", config=config, api_key="local-test-key")
        if request.param == "normal"
        else create_local_services_app(config, api_key="local-test-key")
    )
    controller = FakeController()
    app.state.local_monitor_controller = controller
    return app, controller, tmp_path / _DATABASE


def _client(app, *, peer: str = "127.0.0.1") -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1", client=(peer, 50000))


def _uncached(response, code: int) -> None:
    assert response.status_code == code, response.text
    assert response.headers["Cache-Control"] == "no-store"


def test_monitor_status_has_no_write_side_effects_and_patch_is_an_explicit_opt_in(
    monitor,
):
    app, controller, database = monitor
    with _client(app) as client:
        status = client.get(_ROUTE, headers=_AUTH)
        _uncached(status, 200)
        assert status.json() == {
            "enabled": False,
            "running": False,
            "externally_managed": False,
            "notifications_enabled": False,
            "history_available": True,
            "incidents": [],
        }
        assert not database.exists()
        assert controller.calls == []

        preflight = client.options(
            _ROUTE,
            headers={
                **_WEBVIEW,
                "Access-Control-Request-Method": "PATCH",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert preflight.status_code == 200
        assert "PATCH" in preflight.headers["Access-Control-Allow-Methods"]
        assert controller.calls == []

        start = client.patch(_ROUTE, json={"enabled": True}, headers=_AUTH | _WEBVIEW)
        _uncached(start, 200)
        assert start.headers["Access-Control-Allow-Origin"] == _WEBVIEW["Origin"]
        assert start.json()["enabled"] is True
        assert start.json()["running"] is True
        assert start.json()["notifications_enabled"] is False

        end = client.patch(_ROUTE, json={"enabled": False}, headers=_AUTH)
        _uncached(end, 200)
        assert end.json()["enabled"] is False
        assert end.json()["running"] is False
        assert end.json()["notifications_enabled"] is False
        assert controller.calls == [(True, False), (False, False)]


def test_monitor_notification_choice_is_reported_on_inspection(monitor):
    app, controller, _ = monitor
    with _client(app) as client:
        start = client.patch(
            _ROUTE, json={"enabled": True, "notifications_enabled": True}, headers=_AUTH
        )
        _uncached(start, 200)
        assert start.json()["notifications_enabled"] is True
        status = client.get(_ROUTE, headers=_AUTH)
        _uncached(status, 200)
        assert status.json()["notifications_enabled"] is True
    assert controller.calls == [(True, True)]


def test_monitor_rejects_unsafe_requests_without_applying_changes(monitor):
    app, controller, _ = monitor
    with _client(app) as client:
        assert client.get(_ROUTE).status_code == 401
        assert client.patch(_ROUTE, json={"enabled": True}).status_code == 401

        for headers in (
            {"Host": "remote.example"},
            {"Origin": "https://remote.example"},
            {"X-Forwarded-For": "127.0.0.1"},
        ):
            _uncached(client.get(_ROUTE, headers=_AUTH | headers), 403)
            _uncached(
                client.patch(_ROUTE, json={"enabled": True}, headers=_AUTH | headers),
                403,
            )
        _uncached(
            client.patch(
                _ROUTE,
                content=b'{"enabled": true}',
                headers=_AUTH | {"Content-Type": "text/plain"},
            ),
            415,
        )
        for value in (
            {"enabled": "true"},
            {"enabled": True, "extra": "private"},
            {"enabled": True, "notifications_enabled": "true"},
            {"enabled": True, "notifications_enabled": None},
            {"enabled": False, "notifications_enabled": True},
        ):
            invalid = client.patch(_ROUTE, json=value, headers=_AUTH)
            _uncached(invalid, 422)
            assert "private" not in invalid.text
        _uncached(
            client.patch(
                _ROUTE,
                content=b'{"enabled": true}' + b" " * 1024,
                headers=_AUTH | {"Content-Type": "application/json"},
            ),
            413,
        )

    with _client(app, peer="192.0.2.10") as client:
        _uncached(client.patch(_ROUTE, json={"enabled": True}, headers=_AUTH), 403)
    assert controller.calls == []


def test_monitor_history_projects_incidents_without_raw_details_or_writes(monitor):
    app, _, database = monitor
    store = LocalMonitorStore(database)
    assert store.observe(
        "cpu_high", "warning", "High CPU", {"process": "private-name.exe"}
    )
    incident_id = store.recent()[0]["id"]
    assert store.save_assessment(incident_id, "Local explanation")
    assert store.acknowledge(incident_id)
    assert store.resolve("cpu_high")

    previous = database.read_bytes()
    with _client(app) as client:
        status = client.get(_ROUTE, headers=_AUTH)
    _uncached(status, 200)
    payload = status.json()
    assert payload["history_available"] is True
    incident = payload["incidents"][0]
    assert incident["id"] == incident_id
    assert incident["status"] == "resolved"
    assert incident["resolved_at"]
    assert incident["acknowledged_at"]
    assert incident["episodes_last_30d"] == 1
    assert incident["assessments"][0]["analysis"] == "Local explanation"
    assert "private-name.exe" not in status.text
    assert "details" not in incident
    assert "memory_captured_at" not in incident
    assert database.read_bytes() == previous


def test_legacy_monitor_history_can_be_read_without_schema_migration(monitor):
    app, _, database = monitor
    with closing(sqlite3.connect(database)) as db:
        db.executescript(
            """
            CREATE TABLE local_monitor_events (
                id INTEGER PRIMARY KEY, metric TEXT, severity TEXT,
                status TEXT, summary TEXT, details_json TEXT, first_seen TEXT,
                last_seen TEXT, resolved_at TEXT, observations INTEGER
            );
            INSERT INTO local_monitor_events VALUES (
                1, 'cpu_high', 'warning', 'active', 'High CPU',
                '{}', '2026-01-01', '2026-01-01', NULL, 1
            );
            """
        )
    previous = database.read_bytes()
    with _client(app) as client:
        response = client.get(_ROUTE, headers=_AUTH)
    _uncached(response, 200)
    assert response.json()["history_available"] is True
    incident = response.json()["incidents"][0]
    assert incident["acknowledged_at"] is None
    assert incident["assessments"] == []
    assert database.read_bytes() == previous


def test_monitor_can_be_disabled_if_history_is_unreadable(monitor):
    app, controller, database = monitor
    database.write_bytes(b"This is not SQLite")
    controller.enabled = controller.running = True
    with _client(app) as client:
        broken = client.get(_ROUTE, headers=_AUTH)
        _uncached(broken, 200)
        assert broken.json()["history_available"] is False
        assert broken.json()["incidents"] == []
        end = client.patch(_ROUTE, json={"enabled": False}, headers=_AUTH)
    _uncached(end, 200)
    assert end.json()["history_available"] is False
    assert end.json()["running"] is False
    assert controller.calls == [(False, False)]


def test_externally_managed_monitor_conflict_is_safe_and_uncached(monitor):
    app, controller, _ = monitor
    controller.external = controller.running = True
    with _client(app) as client:
        status = client.get(_ROUTE, headers=_AUTH)
        _uncached(status, 200)
        assert status.json()["externally_managed"] is True
        conflict = client.patch(_ROUTE, json={"enabled": False}, headers=_AUTH)
    _uncached(conflict, 409)
    assert "private conflict detail" not in conflict.text
    assert controller.calls == []


def test_incident_acknowledgement_is_idempotent_and_does_not_resolve(monitor):
    app, controller, database = monitor
    with _client(app) as client:
        missing = client.post(
            f"{_ROUTE}/incidents/1/acknowledge", json={}, headers=_AUTH
        )
        _uncached(missing, 404)
        assert not database.exists()

        store = LocalMonitorStore(database)
        assert store.observe("cpu_high", "warning", "High CPU", {"private": "detail"})
        incident_id = store.recent()[0]["id"]
        route = f"{_ROUTE}/incidents/{incident_id}/acknowledge"
        for headers in (
            {"Host": "remote.example"},
            {"Origin": "https://remote.example"},
            {"X-Forwarded-For": "127.0.0.1"},
        ):
            _uncached(client.post(route, json={}, headers=_AUTH | headers), 403)
        assert client.post(route, json={}).status_code == 401
        _uncached(client.post(route, json={"status": "resolved"}, headers=_AUTH), 422)
        _uncached(
            client.post(
                route, content="{}", headers=_AUTH | {"Content-Type": "text/plain"}
            ),
            415,
        )
        _uncached(
            client.post(
                route,
                content=b"{}" + b" " * 1024,
                headers=_AUTH | {"Content-Type": "application/json"},
            ),
            413,
        )
        assert store.recent()[0]["acknowledged_at"] is None

        response = client.post(route, json={}, headers=_AUTH | _WEBVIEW)
        _uncached(response, 200)
        incident = response.json()["incidents"][0]
        assert incident["status"] == "active"
        assert incident["acknowledged_at"]
        assert incident["resolved_at"] is None
        assert "details" not in incident
        again = client.post(route, json={}, headers=_AUTH)
        _uncached(again, 200)
        assert again.json()["incidents"][0] == incident
    assert controller.calls == []


def _observed_cpu(database):
    store = LocalMonitorStore(database)
    assert store.observe(
        "cpu_high",
        "warning",
        "summary includes do-not-share.exe",
        {"cpu_percent": 96.0, "limit_percent": 95.0, "user_path": "fixture-private"},
    )
    incident = store.recent(1)[0]
    return store, incident["id"], f"{_ROUTE}/incidents/{incident['id']}/analyze"


def test_analyze_measurements_for_recovered_incident_is_saved_once(monitor):
    app, _, database = monitor
    store, incident_id, route = _observed_cpu(database)
    assert store.resolve("cpu_high")
    with _client(app) as client:
        first = client.post(route, json={}, headers=_AUTH)
        _uncached(first, 200)
        assert first.json()["mode"] == "measurements"
        assert first.json()["incident_id"] == incident_id
        assert "96.0%" in first.json()["assessment"]["analysis"]
        assert "pending for local-model analysis" not in first.text
        assert "do-not-share.exe" not in first.text
        repeat = client.post(route, json={}, headers=_AUTH)
        assert repeat.json() == first.json()
        incident = client.get(_ROUTE, headers=_AUTH).json()["incidents"][0]
        assert incident["status"] == "resolved"
        assert incident["acknowledged_at"] is None
        assert incident["assessments"] == [first.json()["assessment"]]


def test_invalid_analysis_cannot_create_history_or_contact_an_engine(monitor):
    app, _, database = monitor
    route = f"{_ROUTE}/incidents/1/analyze"
    with _client(app) as client:
        _uncached(client.post(route, json={}, headers=_AUTH), 404)
        _uncached(client.post(route, json={"use_local_model": 1}, headers=_AUTH), 422)
        _uncached(
            client.post(
                route,
                json={"use_local_model": True},
                headers=_AUTH | {"Origin": "https://external.test"},
            ),
            403,
        )
        assert not database.exists()


def test_explicit_cloudflare_runtime_falls_back_to_measurements(monitor, monkeypatch):
    from openjarvis.engine.cloudflare import CloudflareEngine
    from openjarvis.server import local_monitor_routes

    app, _, database = monitor
    _, _, route = _observed_cpu(database)
    app.state.engine = CloudflareEngine()
    app.state.model = "@cf/google/gemma-4-26b-a4b-it"

    def unexpected(*args, **kwargs):
        pytest.fail("Assessment called a cloud engine.")

    monkeypatch.setattr(app.state.engine, "generate", unexpected)
    with _client(app) as client:
        response = client.post(route, json={"use_local_model": True}, headers=_AUTH)
        _uncached(response, 200)
        assert response.json()["mode"] == "measurements"
        with local_monitor_routes._ANALYSIS_SLOT:
            response = client.post(route, json={"use_local_model": True}, headers=_AUTH)
            _uncached(response, 429)
            _uncached(client.post(route, json={}, headers=_AUTH), 200)


@pytest.mark.parametrize("sample_changes", [False, True])
def test_model_receives_only_pinned_aggregate_facts(
    monitor, monkeypatch, sample_changes
):
    import json

    import httpx

    from openjarvis.engine.ollama import OllamaEngine

    app, _, database = monitor
    store, _, route = _observed_cpu(database)
    engine = OllamaEngine(host="http://localhost:11713")
    init = OllamaEngine.__init__
    requests = []

    def serve(request):
        body = json.loads(request.read())
        requests.append(request.url.path)
        assert request.url.host == "127.0.0.1"
        assert body["model"] == "local-fixture:latest:local"
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion"]})
        assert request.url.path == "/api/chat"
        assert "tools" not in body
        prompt = body["messages"][-1]["content"]
        assert "cpu_percent" in prompt and "96.0" in prompt
        assert "do-not-share.exe" not in prompt and "fixture-private" not in prompt
        if sample_changes:
            store.observe(
                "cpu_high",
                "critical",
                "new severity",
                {"cpu_percent": 99.0, "limit_percent": 95.0},
            )
        return httpx.Response(
            200, json={"message": {"content": "Reveja a carga de CPU por aplicativo."}}
        )

    def open_pinned(self, *args, **options):
        assert options["trust_env"] is False
        init(self, *args, **options)
        self._client.close()
        self._client = httpx.Client(
            base_url=self._host, transport=httpx.MockTransport(serve)
        )

    monkeypatch.setattr(OllamaEngine, "__init__", open_pinned)
    app.state.engine = engine
    app.state.model = "local-fixture:latest"
    try:
        with _client(app) as client:
            result = client.post(route, json={"use_local_model": True}, headers=_AUTH)
            assert requests == ["/api/show", "/api/chat"]
            saved = client.get(_ROUTE, headers=_AUTH).json()["incidents"][0]
            if sample_changes:
                _uncached(result, 409)
                assert saved["severity"] == "critical"
                assert all(
                    "Reveja a carga" not in item["analysis"]
                    for item in saved["assessments"]
                )
            else:
                _uncached(result, 200)
                assert result.json()["mode"] == "local_model"
                assert "Reveja a carga" in result.json()["assessment"]["analysis"]
                assert saved["assessments"][0] == result.json()["assessment"]
    finally:
        engine.close()
