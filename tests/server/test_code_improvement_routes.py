"""Tests for the local code improvement proposal and review API."""

from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from openjarvis.core.config import JarvisConfig
from openjarvis.learning.code_improvement import CodeImprovementManager
from openjarvis.server import code_improvement_routes
from openjarvis.server.app import create_app
from openjarvis.server.local_services import create_local_services_app

_ROUTE = "/v1/code-improvements"
_OLDER = "abc000000001"
_NEWER = "abc000000002"
_MISSING = "abc000000003"
_PATCH = "diff --git a/example.py b/example.py\n-old\n+new\n"
_PRIVATE_REPOSITORY = "X:/private/source"


@pytest.fixture(params=["normal", "services"])
def api(request, monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))
    monkeypatch.setattr(code_improvement_routes, "get_config_dir", lambda: tmp_path)

    def bad_constructor(*args, **kwargs):
        pytest.fail("Review must never instantiate the write-side manager.")

    monkeypatch.setattr(CodeImprovementManager, "__init__", bad_constructor)
    config = JarvisConfig()
    app = (
        create_app(object(), "test-model", config=config)
        if request.param == "normal"
        else create_local_services_app(config)
    )
    return app, tmp_path / "improvements.db"


def _client(app, peer: str = "127.0.0.1") -> TestClient:
    return TestClient(app, base_url="http://localhost", client=(peer, 50000))


def _schema(db, *, legacy=False):
    db.executescript(
        """
        CREATE TABLE code_improvement_proposals (
            proposal_id TEXT PRIMARY KEY, title TEXT, status TEXT, created_at REAL,
            files_json TEXT, repository TEXT, patch TEXT, validation TEXT
        );
        """
    )
    if not legacy:
        db.executescript(
            """
            ALTER TABLE code_improvement_proposals ADD COLUMN validation_receipt TEXT;
            CREATE TABLE code_improvement_apply_journals (proposal_id TEXT PRIMARY KEY);
            """
        )


def _insert(db, proposal, status, created_at):
    db.execute(
        "INSERT INTO code_improvement_proposals (proposal_id, title, status, "
        "created_at, files_json, repository, patch, validation) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            proposal,
            "Improve example",
            status,
            created_at,
            json.dumps(["example.py"]),
            _PRIVATE_REPOSITORY,
            _PATCH,
            "Syntax check passed.",
        ),
    )


def _assert_uncached(response, status):
    assert response.status_code == status, response.text
    assert response.headers["Cache-Control"] == "no-store"


def test_empty_profile_stays_empty(api):
    app, database = api
    with _client(app) as client:
        response = client.get(_ROUTE)
        _assert_uncached(response, 200)
        assert response.json() == {"proposals": []}
        _assert_uncached(client.get(f"{_ROUTE}/{_MISSING}"), 404)
    assert not database.exists()


def test_reads_committed_wal_without_writing_or_loading_manager(api):
    app, database = api
    with closing(sqlite3.connect(database)) as db:
        assert db.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        db.execute("PRAGMA wal_autocheckpoint=0")
        _schema(db)
        _insert(db, _OLDER, "approved", 6.0)
        _insert(db, _NEWER, "applied", 12.0)
        db.execute("UPDATE code_improvement_proposals SET validation_receipt='receipt'")
        db.executemany(
            "INSERT INTO code_improvement_apply_journals VALUES (?)",
            ((_OLDER,), (_NEWER,)),
        )
        db.commit()

        wal = Path(str(database) + "-wal")
        before = (database.read_bytes(), wal.read_bytes())
        with _client(app) as client:
            response = client.get(_ROUTE, params={"limit": 1})
            _assert_uncached(response, 200)
            expected = {
                "proposal_id": _NEWER,
                "title": "Improve example",
                "status": "applied",
                "created_at": 12.0,
                "files": ["example.py"],
                "rollback_available": True,
            }
            assert response.json() == {"proposals": [expected]}
            assert _PRIVATE_REPOSITORY not in response.text
            summaries = client.get(_ROUTE).json()["proposals"]
            assert summaries[1]["proposal_id"] == _OLDER
            assert not summaries[1]["rollback_available"]

            detail = client.get(
                f"{_ROUTE}/{_NEWER}", headers={"Origin": "http://tauri.localhost"}
            )
            _assert_uncached(detail, 200)
            assert (
                detail.headers["Access-Control-Allow-Origin"]
                == "http://tauri.localhost"
            )
            assert detail.json() == {
                **expected,
                "repository": _PRIVATE_REPOSITORY,
                "patch": _PATCH,
                "validation": "Syntax check passed.",
                "validation_receipt": "receipt",
            }
            assert client.post(f"{_ROUTE}/{_NEWER}").status_code == 405
        assert (database.read_bytes(), wal.read_bytes()) == before


def test_rolling_back_is_resumable_only_with_a_persisted_journal(api):
    app, database = api
    with closing(sqlite3.connect(database)) as db:
        _schema(db)
        _insert(db, _OLDER, "rolling_back", 6.0)
        _insert(db, _NEWER, "rolling_back", 12.0)
        db.execute("UPDATE code_improvement_proposals SET validation_receipt='receipt'")
        db.execute("INSERT INTO code_improvement_apply_journals VALUES (?)", (_NEWER,))
        db.commit()

    with _client(app) as client:
        listing = client.get(_ROUTE)
        _assert_uncached(listing, 200)
        assert [
            (row["proposal_id"], row["rollback_available"])
            for row in listing.json()["proposals"]
        ] == [(_NEWER, True), (_OLDER, False)]
        detail = client.get(f"{_ROUTE}/{_NEWER}")
        _assert_uncached(detail, 200)
        assert detail.json()["status"] == "rolling_back"
        assert detail.json()["rollback_available"] is True


def test_legacy_database_can_be_read_without_migration(api):
    app, database = api
    with closing(sqlite3.connect(database)) as db:
        _schema(db, legacy=True)
        _insert(db, _OLDER, "applied", 6.0)
        db.commit()
    before = database.read_bytes()
    with _client(app) as client:
        response = client.get(f"{_ROUTE}/{_OLDER}")
        _assert_uncached(response, 200)
        assert response.json()["validation_receipt"] == ""
        assert not response.json()["rollback_available"]
        assert response.json()["patch"] == _PATCH
    assert database.read_bytes() == before


@pytest.mark.parametrize(
    ("peer", "headers"),
    [
        ("192.0.2.15", {}),
        ("127.0.0.1", {"Host": "public.example"}),
        ("127.0.0.1", {"Origin": "https://public.example"}),
        ("127.0.0.1", {"Forwarded": "for=127.0.0.1"}),
    ],
)
def test_external_access_is_denied_before_query_parsing(api, peer, headers):
    app, database = api
    with _client(app, peer) as client:
        response = client.get(_ROUTE, params={"limit": "private"}, headers=headers)
        _assert_uncached(response, 403)
        assert "private" not in response.text
    assert not database.exists()


def test_invalid_input_and_corruption_do_not_echo_private_data(api):
    app, database = api
    with _client(app) as client:
        for url in (
            _ROUTE + "?limit=private",
            _ROUTE + "/private",
            _ROUTE + "?limit=201",
        ):
            response = client.get(url)
            _assert_uncached(response, 422)
            assert response.json() == {"detail": "Invalid code improvement request."}
        database.write_text("private corruption", encoding="utf-8")
        response = client.get(_ROUTE)
        _assert_uncached(response, 503)
        assert response.json() == {"detail": "Code improvement storage is unavailable."}


@pytest.mark.skipif(shutil.which("git") is None, reason="Git required")
def test_creates_validated_local_draft_without_editing_checkout(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setenv("OPENJARVIS_HOME", str(profile))
    repository = tmp_path / "source"
    (repository / ".git").mkdir(parents=True)
    source = repository / "example.py"
    source.write_bytes(b"value = 1\n")
    (repository / "unselected.py").write_bytes(b"UNSELECTED = True\n")
    patch = "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    calls = []

    class SelectedModel:
        def generate(self, messages, **kwargs):
            assert not kwargs.get("tools")
            calls.append(kwargs["model"])
            assert "value = 1" in messages[1].text
            assert "UNSELECTED" not in messages[1].text
            return {"content": patch}

    @contextmanager
    def pinned(_state, *, capability):
        assert capability == "completion"
        yield SelectedModel(), "local-model"

    def unexpected_recovery(*_args):
        pytest.fail("A desktop draft must not recover a prior apply or rollback.")

    monkeypatch.setattr(code_improvement_routes, "pinned_local_inference", pinned)
    monkeypatch.setattr(
        CodeImprovementManager, "_recover_interrupted_applies", unexpected_recovery
    )
    with _client(create_app(object(), "local-model")) as client:
        response = client.post(
            _ROUTE,
            headers={"Origin": "http://tauri.localhost"},
            json={
                "repository": str(repository),
                "task": "Change the value",
                "files": ["example.py"],
            },
        )
        _assert_uncached(response, 201)
        draft = response.json()
        assert draft["status"] == "validated"
        assert draft["validation_receipt"]
        assert draft["files"] == ["example.py"]
        assert client.get(f"{_ROUTE}/{draft['proposal_id']}").json() == draft
    assert source.read_bytes() == b"value = 1\n"
    assert calls == ["local-model"]


def test_post_refuses_nonlocal_model_and_invalid_input_before_writing(api):
    app, database = api
    body = {
        "repository": str(database.parent),
        "task": "private request",
        "files": ["example.py"],
    }
    with _client(app) as client:
        _assert_uncached(
            client.post(
                _ROUTE, json=body, headers={"Origin": "https://public.example"}
            ),
            403,
        )
        for response, status in (
            (client.post(_ROUTE, json=body), 503),
            (client.post(_ROUTE, json={**body, "files": []}), 422),
            (client.post(_ROUTE, json={**body, "apply": True}), 422),
            (client.post(_ROUTE, content=json.dumps(body)), 415),
            (client.post(_ROUTE, json={**body, "task": "x" * (140 * 1024)}), 413),
        ):
            _assert_uncached(response, status)
            assert "private request" not in response.text
    assert not database.exists()
