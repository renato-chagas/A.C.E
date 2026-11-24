"""Exercise the public text reader with simulated transports and DNS records."""

from __future__ import annotations

import io
import socket

import pytest

from openjarvis.security import public_web, ssrf


class _Response:
    def __init__(self, status: int, body: bytes = b"", **headers: str) -> None:
        self.status = status
        self.headers = {name.lower(): value for name, value in headers.items()}
        self.body = io.BytesIO(body)
        self.read_calls = 0

    def getheader(self, name: str, default: str | None = None) -> str | None:
        return self.headers.get(name.lower().replace("-", "_"), default)

    def read1(self, size: int) -> bytes:
        self.read_calls += 1
        return self.body.read(size)


@pytest.fixture
def transport(monkeypatch):
    responses: dict[str, _Response] = {}
    requested = []
    resolved = []

    def resolve(name, port, *_args):
        resolved.append((name, port))
        address = "127.0.0.1" if name == "private.test" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    class Connection:
        def __init__(self, host, port, *, pinned_ip, timeout):
            assert pinned_ip == "93.184.216.34"
            assert timeout > 0
            self.host = host
            self.port = port

        def request(self, method, path, *, headers):
            assert method == "GET"
            assert headers["Host"] == self.host
            assert headers["Accept-Encoding"] == "identity"
            assert (
                not {"Authorization", "Cookie", "Proxy-Authorization"} & headers.keys()
            )
            self.path = path
            requested.append((self.host, self.port, path))

        def getresponse(self):
            return responses[self.path]

        def close(self):
            pass

    # Simulate the preceding DNS policy and ensure the pinned resolution still
    # rejects private addresses before initializing an HTTP connection.
    monkeypatch.setattr(ssrf, "check_ssrf", lambda _url: None)
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(public_web, "_PinnedHTTPConnection", Connection)
    monkeypatch.setattr(public_web, "_PinnedHTTPSConnection", Connection)
    return responses, requested, resolved


def test_public_redirects_use_pinned_addresses_and_extract_literal_text(transport):
    responses, requested, resolved = transport
    responses["/start"] = _Response(
        302, location="https://second.test/p;section?q=caf%C3%A9"
    )
    responses["/p;section?q=caf%C3%A9"] = _Response(
        200,
        b"<head><title>metadata</title></head><p>Caf\xe9 &amp; p\xe3o</p>"
        b"<script>script payload</script><p>Agora</p><style>CSS</style>",
        content_type="text/html; charset=iso-8859-1",
    )
    assert (
        public_web.fetch_public_text("https://first.test/start") == "Café & pão Agora"
    )
    assert resolved == [("first.test", 443), ("second.test", 443)]
    assert requested == [
        ("first.test", 443, "/start"),
        ("second.test", 443, "/p;section?q=caf%C3%A9"),
    ]


def test_rejects_private_redirect_without_reading_either_body(transport):
    responses, requested, resolved = transport
    jump = responses["/redirect"] = _Response(302, location="http://private.test/")
    with pytest.raises(public_web.PublicWebPolicyError, match="public HTTP"):
        public_web.fetch_public_text("https://first.test/redirect")
    assert resolved == [("first.test", 443), ("private.test", 80)]
    assert requested == [("first.test", 443, "/redirect")]
    assert jump.read_calls == 0


@pytest.mark.parametrize(
    "url",
    ["https://user@first.test/", "file:///tmp/content", "https://first.test:0/"],
)
def test_rejects_urls_with_credentials_or_invalid_destinations(transport, url):
    _responses, requested, resolved = transport
    with pytest.raises(public_web.PublicWebPolicyError):
        public_web.fetch_public_text(url)
    assert not requested
    assert not resolved


@pytest.mark.parametrize(
    ("status", "headers"),
    [
        (401, {"content_type": "text/plain"}),
        (200, {"content_type": "image/png"}),
        (200, {"content_type": "text/plain", "content_encoding": "gzip"}),
        (200, {}),
    ],
)
def test_rejects_auth_and_unsupported_media_before_read(transport, status, headers):
    responses, *_rest = transport
    response = responses["/"] = _Response(status, b"unread content", **headers)
    with pytest.raises(RuntimeError):
        public_web.fetch_public_text("https://first.test/")
    assert response.read_calls == 0


def test_bounds_response_bytes_and_returns_plain_text(transport, monkeypatch):
    responses, *_rest = transport
    monkeypatch.setattr(public_web, "_MAX_BYTES", 12)
    responses["/"] = _Response(200, b"0123456789abc", content_type="text/plain")
    with pytest.raises(RuntimeError, match="response byte limit"):
        public_web.fetch_public_text("https://first.test/")
    responses["/"] = _Response(200, b"<b>Literal", content_type="text/plain")
    assert public_web.fetch_public_text("https://first.test/", max_chars=6) == "<b>Lit"


@pytest.mark.parametrize("location", ["https://fi\trst.test/", "https://[invalid"])
def test_rejects_malformed_redirect_locations(transport, location):
    responses, requested, _resolved = transport
    responses["/"] = _Response(302, location=location)
    with pytest.raises(public_web.PublicWebPolicyError):
        public_web.fetch_public_text("https://first.test/")
    assert requested == [("first.test", 443, "/")]


def test_incomplete_response_is_not_saved_as_a_full_page(transport):
    responses, *_rest = transport
    responses["/"] = _Response(
        200, b"part", content_type="text/plain", content_length="10"
    )
    with pytest.raises(RuntimeError, match="response was incomplete"):
        public_web.fetch_public_text("https://first.test/")
