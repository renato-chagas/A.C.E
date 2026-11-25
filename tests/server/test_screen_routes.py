"""Exercise the private screen request boundary without using the live desktop."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from openjarvis.cli import _screen
from openjarvis.core.config import JarvisConfig
from openjarvis.engine.cloudflare import (
    CLOUDFLARE_MODEL_ID,
    CloudflareEngine,
    CloudflareInferenceError,
)
from openjarvis.engine.ollama import OllamaEngine
from openjarvis.server import screen_routes
from openjarvis.server.app import create_app
from openjarvis.server.local_services import create_local_services_app
from openjarvis.tools.storage.ocr import OcrUnavailableError

_PATH = "/v1/screen/inspect"
_IMAGE = b"\x89PNG\r\n\x1a\nscreen fixture"


def _client(app, peer="127.0.0.1"):
    return TestClient(app, base_url="http://127.0.0.1", client=(peer, 52178))


@pytest.fixture(autouse=True)
def _no_live_screen(monkeypatch):
    monkeypatch.setattr(screen_routes, "_capture_supported", lambda: True)
    monkeypatch.setattr(
        _screen,
        "capture_screen_bytes",
        lambda: pytest.fail("Unexpected desktop capture in a route test"),
    )


def test_local_services_reads_screen_text_once_without_inference(monkeypatch):
    captures = []
    monkeypatch.setattr(
        _screen, "capture_screen_bytes", lambda: captures.append(_IMAGE) or _IMAGE
    )
    monkeypatch.setattr(
        screen_routes, "_prepare_ocr", lambda: lambda image: image[-14:].decode()
    )
    app = create_local_services_app(JarvisConfig())
    with _client(app) as client:
        assert client.get(_PATH).json() == {
            "available": False,
            "text_available": True,
            "text_question_available": False,
            "cloud_vision_available": False,
        }
        assert captures == []
        response = client.post(
            _PATH, json={"mode": "text"}, headers={"Origin": "http://tauri.localhost"}
        )
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json() == {"answer": "screen fixture"}
        assert (
            client.post(_PATH, json={"question": "What is visible?"}).status_code == 503
        )
        assert captures == [_IMAGE]


@pytest.mark.parametrize(
    ("peer", "headers", "status"),
    [
        ("192.0.2.55", {}, 403),
        ("127.0.0.1", {"Host": "external.test"}, 403),
        ("127.0.0.1", {"Origin": "https://external.test"}, 403),
        ("127.0.0.1", {"X-Forwarded-For": "127.0.0.1"}, 403),
        ("127.0.0.1", {}, 401),
    ],
)
def test_nonlocal_and_unauthorized_requests_cannot_capture(peer, headers, status):
    app = create_local_services_app(JarvisConfig(), api_key="profile-test-key")
    with _client(app, peer) as client:
        response = client.post(_PATH, headers=headers, json={"mode": "text"})
        assert response.status_code == status


@pytest.mark.parametrize(
    "body",
    [
        {"mode": "text", "question": "unexpected field"},
        {"mode": "other"},
        {"mode": "vision"},
        {"mode": "vision", "question": "   "},
        {"mode": "vision", "question": "x" * 2001},
        {"mode": "text-question", "question": "   "},
        {"mode": "cloud-vision", "question": "Look"},
        {"mode": "cloud-vision", "question": "Look", "confirm_upload": False},
        {"mode": "cloud-vision", "question": "Look", "confirm_upload": 1},
        {"question": 23},
    ],
)
def test_invalid_request_cannot_capture(body):
    with _client(create_local_services_app(JarvisConfig())) as client:
        response = client.post(_PATH, json=body)
        assert response.status_code == 422
        assert response.json() == {"detail": "Invalid screen inspection request."}
        assert response.headers["cache-control"] == "no-store"


def test_bad_media_type_and_excessive_body_cannot_capture():
    with _client(create_local_services_app(JarvisConfig())) as client:
        response = client.post(_PATH, content='{"mode":"text"}')
        assert response.status_code == 415
        response = client.post(_PATH, json={"question": "x" * (40 * 1024)})
        assert response.status_code == 413
        assert response.headers["cache-control"] == "no-store"


def test_ocr_availability_and_busy_requests_fail_before_capture(monkeypatch):
    from openjarvis.tools.storage import ocr

    def unavailable():
        raise OcrUnavailableError("private path")

    monkeypatch.setattr(ocr, "screen_ocr_reader", unavailable)
    with _client(create_local_services_app(JarvisConfig())) as client:
        assert client.get(_PATH).json() == {
            "available": False,
            "text_available": False,
            "text_question_available": False,
            "cloud_vision_available": False,
        }
        response = client.post(_PATH, json={"mode": "text"})
        assert response.status_code == 503
        assert "private path" not in response.text
        with screen_routes._CAPTURE_LOCK:
            assert client.get(_PATH).status_code == 200
            assert client.post(_PATH, json={"mode": "text"}).status_code == 409


def _mock_ollama(monkeypatch, show_result, *, chat_status=200):
    primary = OllamaEngine(host="http://localhost:11434")
    original_init = OllamaEngine.__init__
    calls = []

    def handle(request):
        calls.append((str(request.url), json.loads(request.read())))
        if request.url.path == "/api/show":
            if show_result is None:
                return httpx.Response(
                    307, headers={"Location": "https://external.test/show"}
                )
            return httpx.Response(200, json=show_result)
        if chat_status == 307:
            return httpx.Response(
                307, headers={"Location": "https://external.test/chat"}
            )
        return httpx.Response(
            200, json={"message": {"content": "Synthetic local reply"}}
        )

    def init(self, *args, **kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["host"] == "http://127.0.0.1:11434"
        original_init(self, *args, **kwargs)
        self._client.close()
        self._client = httpx.Client(
            base_url=self._host, transport=httpx.MockTransport(handle)
        )

    monkeypatch.setattr(OllamaEngine, "__init__", init)
    return primary, calls


@pytest.mark.parametrize(
    "show_result",
    [
        None,
        {"capabilities": ["completion"]},
        {"capabilities": ["vision"], "remote_host": "example.test"},
        {"capabilities": ["vision"], "remote_model": "remote tag"},
    ],
)
def test_vision_refuses_redirect_remote_or_text_only_before_capture(
    monkeypatch, show_result
):
    primary, calls = _mock_ollama(monkeypatch, show_result)
    try:
        with _client(
            create_app(primary, "fixture:latest", engine_name="ollama")
        ) as client:
            assert client.post(_PATH, json={"question": "Look"}).status_code == 503
        assert calls == [
            ("http://127.0.0.1:11434/api/show", {"model": "fixture:latest:local"})
        ]
    finally:
        primary.close()


@pytest.mark.parametrize("chat_status", [200, 307])
def test_vision_sends_one_image_to_the_pinned_local_model(monkeypatch, chat_status):
    primary, calls = _mock_ollama(
        monkeypatch, {"capabilities": ["vision"]}, chat_status=chat_status
    )
    monkeypatch.setattr(screen_routes, "_prepare_ocr", lambda: None)
    captures = []
    monkeypatch.setattr(
        _screen, "capture_screen_bytes", lambda: captures.append(_IMAGE) or _IMAGE
    )
    try:
        with _client(
            create_app(primary, "fixture:latest", engine_name="ollama")
        ) as client:
            assert client.get(_PATH).json() == {
                "available": True,
                "text_available": False,
                "text_question_available": False,
                "cloud_vision_available": False,
            }
            assert not captures
            response = client.post(
                _PATH, json={"mode": "vision", "question": "  Look  "}
            )
            assert len(captures) == 1
        assert [url for url, _ in calls] == [
            "http://127.0.0.1:11434/api/show",
            "http://127.0.0.1:11434/api/show",
            "http://127.0.0.1:11434/api/chat",
        ]
        assert calls[0][1]["model"] == "fixture:latest:local"
        payload = calls[-1][1]
        assert payload["model"] == "fixture:latest:local"
        assert "tools" not in payload
        assert payload["messages"][-1]["content"] == "Look"
        assert payload["messages"][-1]["images"] == [base64.b64encode(_IMAGE).decode()]
        if chat_status == 200:
            assert response.json() == {"answer": "Synthetic local reply"}
        else:
            assert response.status_code == 502
            assert "external.test" not in response.text
    finally:
        primary.close()



def test_text_question_sends_only_bounded_ocr_after_local_route_check(monkeypatch):
    primary, calls = _mock_ollama(monkeypatch, {"capabilities": ["completion"]})
    text = "Visible untrusted text. " * 200
    captures = []

    def capture():
        assert [url for url, _ in calls] == [
            "http://127.0.0.1:11434/api/show",
            "http://127.0.0.1:11434/api/show",
            "http://127.0.0.1:11434/api/show",
        ]
        captures.append(_IMAGE)
        return _IMAGE

    monkeypatch.setattr(_screen, "capture_screen_bytes", capture)
    monkeypatch.setattr(screen_routes, "_prepare_ocr", lambda: lambda image: text)
    try:
        with _client(
            create_app(primary, "fixture:latest", engine_name="ollama")
        ) as client:
            assert client.get(_PATH).json() == {
                "available": False,
                "text_available": True,
                "text_question_available": True,
                "cloud_vision_available": False,
            }
            assert not captures
            response = client.post(
                _PATH, json={"mode": "text-question", "question": "  What is here? "}
            )
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
            assert response.json() == {"answer": "Synthetic local reply"}
        assert captures == [_IMAGE]
        assert calls[-1][0] == "http://127.0.0.1:11434/api/chat"
        payload = calls[-1][1]
        assert "tools" not in payload
        assert "untrusted OCR data" in payload["messages"][0]["content"]
        assert "images" not in payload["messages"][-1]
        assert json.loads(payload["messages"][-1]["content"]) == {
            "question": "What is here?",
            "screen_text": text[:4000],
            "screen_text_truncated": True,
        }
    finally:
        primary.close()


def test_text_question_refuses_remote_model_before_capture(monkeypatch):
    primary, calls = _mock_ollama(
        monkeypatch, {"capabilities": ["completion"], "remote_host": "example.test"}
    )
    monkeypatch.setattr(
        screen_routes,
        "_prepare_ocr",
        lambda: pytest.fail("OCR must not be prepared for an unverified model"),
    )
    try:
        with _client(
            create_app(primary, "fixture:latest", engine_name="ollama")
        ) as client:
            response = client.post(
                _PATH, json={"mode": "text-question", "question": "What is here?"}
            )
        assert response.status_code == 503
        assert calls == [
            ("http://127.0.0.1:11434/api/show", {"model": "fixture:latest:local"})
        ]
    finally:
        primary.close()


@pytest.mark.parametrize("expected_status", [200, 424, 429])
def test_cloud_vision_requires_consent_and_supported_model(
    monkeypatch, expected_status
):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a" * 32)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "synthetic-token")
    captures = []
    questions = []

    def capture():
        captures.append(_IMAGE)
        return _IMAGE

    def generate(self, messages, *, model, **kwargs):
        assert model == CLOUDFLARE_MODEL_ID
        assert "untrusted data" in messages[0].content
        assert messages[-1].images == [base64.b64encode(_IMAGE).decode()]
        questions.append(messages[-1].content)
        if expected_status == 424:
            raise CloudflareInferenceError(403, "Cloudflare denied access.")
        if expected_status == 429:
            raise CloudflareInferenceError(429, "Cloudflare reached a usage limit.")
        return {"content": "Cloud analysis fixture"}

    monkeypatch.setattr(_screen, "capture_screen_bytes", capture)
    monkeypatch.setattr(screen_routes, "_prepare_ocr", lambda: None)
    monkeypatch.setattr(CloudflareEngine, "generate_vision", generate)
    app = create_local_services_app(
        JarvisConfig(), cloudflare_engine=CloudflareEngine()
    )
    with _client(app) as client:
        assert client.get(_PATH).json() == {
            "available": False,
            "text_available": False,
            "text_question_available": False,
            "cloud_vision_available": True,
        }
        body = {"mode": "cloud-vision", "question": "  Look once  "}
        assert client.post(_PATH, json=body).status_code == 422
        body["confirm_upload"] = True
        assert client.post(
            _PATH, json=body, headers={"Origin": "https://external.test"}
        ).status_code == 403
        app.state.model = "@cf/moonshotai/kimi-k2.7-code"
        assert not client.get(_PATH).json()["cloud_vision_available"]
        assert client.post(_PATH, json=body).status_code == 503
        app.state.model = CLOUDFLARE_MODEL_ID
        assert not captures
        assert not questions
        response = client.post(_PATH, json=body)
        assert response.status_code == expected_status
        assert response.headers["cache-control"] == "no-store"
        if expected_status == 200:
            assert response.json() == {"answer": "Cloud analysis fixture"}
    assert captures == [_IMAGE]
    assert questions == ["Look once"]
