"""Read public pages with the DNS-pinned transport used by the RSS connector."""

from __future__ import annotations

import http.client
import time
from email.message import Message
from html.parser import HTMLParser
from urllib.parse import quote, urljoin, urlsplit

from openjarvis.connectors.news_rss import (
    _PinnedHTTPConnection,
    _PinnedHTTPSConnection,
    _validate_feed_url,
)

_MAX_BYTES = 2 * 1024 * 1024
_MAX_REDIRECTS = 5
_TIMEOUT = 20.0
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_PLAIN_TYPES = frozenset({"text/plain", "text/markdown"})
_SKIP_TAGS = frozenset({"head", "script", "style", "template", "noscript", "svg"})
_BREAK_TAGS = frozenset(
    {
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }
)


class PublicWebPolicyError(ValueError):
    """A supplied URL or redirect destination cannot be fetched safely."""


class _PageText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skipped: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skipped.append(tag)
        elif not self._skipped and tag in _BREAK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._skipped:
            if tag == self._skipped[-1]:
                self._skipped.pop()
        elif tag in _BREAK_TAGS:
            self.parts.append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self._skipped and tag in _BREAK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._skipped:
            self.parts.append(data)


def _seconds_remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("Public page fetch timed out.")
    return remaining


def _read_text_response(
    response: http.client.HTTPResponse,
    deadline: float,
    max_chars: int,
) -> str:
    content_type = response.getheader("Content-Type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type not in _HTML_TYPES | _PLAIN_TYPES:
        raise RuntimeError("The public page response is not supported text.")
    encoding = response.getheader("Content-Encoding", "").strip().lower()
    if encoding not in {"", "identity"}:
        raise RuntimeError("The public page returned unsupported compressed content.")
    length = response.getheader("Content-Length")
    if length is not None:
        try:
            declared_size = int(length)
        except ValueError:
            raise RuntimeError(
                "The public page returned an invalid response size."
            ) from None
        if declared_size < 0 or declared_size > _MAX_BYTES:
            raise RuntimeError("The public page exceeded the response byte limit.")
    else:
        declared_size = None

    body = bytearray()
    while True:
        _seconds_remaining(deadline)
        piece = response.read1(min(64 * 1024, _MAX_BYTES + 1 - len(body)))
        if not piece:
            break
        body.extend(piece)
        if len(body) > _MAX_BYTES:
            raise RuntimeError("The public page exceeded the response byte limit.")
    if declared_size is not None and len(body) != declared_size:
        raise RuntimeError("The public page response was incomplete.")

    media = Message()
    media["Content-Type"] = content_type
    charset = media.get_content_charset() or "utf-8-sig"
    try:
        text = body.decode(charset, errors="replace")
    except (LookupError, UnicodeError):
        text = body.decode("utf-8-sig", errors="replace")
    if media_type in _HTML_TYPES:
        parser = _PageText()
        try:
            parser.feed(text)
            parser.close()
        except (AssertionError, ValueError):
            raise RuntimeError("The public page HTML could not be read.") from None
        text = "".join(parser.parts)
    return " ".join(text.split())[:max_chars].rstrip()


def _reject_url_controls(url: str) -> None:
    # Check before urlsplit and urljoin silently remove embedded controls.
    if any(ord(char) < 32 or ord(char) == 127 for char in url) or "\\" in url:
        raise PublicWebPolicyError("The public page URL contains invalid characters.")


def _fetch_hop(
    url: str,
    deadline: float,
    max_chars: int,
) -> tuple[str | None, str]:
    _reject_url_controls(url)
    try:
        url_parts = urlsplit(url)
        if url_parts.port == 0:
            raise ValueError("Invalid port")
        target = _validate_feed_url(url)
        # Preserve path parameters and percent escapes while encoding Unicode.
        request_target = quote(url_parts.path or "/", safe="/;:@!$&'()*+,=%")
        if url_parts.query:
            request_target += "?" + quote(url_parts.query, safe="/?;:@!$&'()*+,=%")
    except ValueError:
        raise PublicWebPolicyError(
            "The URL is not a reachable public HTTP(S) address."
        ) from None

    connection_class = (
        _PinnedHTTPSConnection if target.scheme == "https" else _PinnedHTTPConnection
    )
    for address in target.addresses:
        connection = connection_class(
            target.hostname,
            target.port,
            pinned_ip=address,
            timeout=_seconds_remaining(deadline),
        )
        try:
            connection.request(
                "GET",
                request_target,
                headers={
                    "Host": target.host_header,
                    "Accept": "text/html, application/xhtml+xml, "
                    "text/plain, text/markdown",
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                    "User-Agent": "OpenJarvis Public Research/1.0",
                },
            )
            response = connection.getresponse()
            if response.status in _REDIRECT_STATUSES:
                location = response.getheader("Location", "")
                _reject_url_controls(location)
                if not location.strip():
                    raise RuntimeError(
                        "The public page redirected without a destination."
                    )
                return location.strip(), ""
            if response.status != 200:
                raise RuntimeError(f"The public page returned HTTP {response.status}.")
            return None, _read_text_response(response, deadline, max_chars)
        except (OSError, http.client.HTTPException):
            continue
        finally:
            connection.close()
    raise RuntimeError("Could not connect to the verified public page.")


def fetch_public_text(url: str, max_chars: int = 9000) -> str:
    """Fetch plain text directly from a public HTTP(S) source or safe redirect."""
    if not url.strip():
        raise PublicWebPolicyError("A public page URL is required.")
    if max_chars < 1:
        raise ValueError("A positive text limit is required.")
    current = url.strip()
    deadline = time.monotonic() + _TIMEOUT
    for _ in range(_MAX_REDIRECTS + 1):
        _seconds_remaining(deadline)
        location, text = _fetch_hop(current, deadline, max_chars)
        if location is None:
            return text
        try:
            current = urljoin(current, location)
        except ValueError:
            raise PublicWebPolicyError("The redirect URL is invalid.") from None
    raise RuntimeError("The public page exceeded the redirect limit.")


__all__ = ["PublicWebPolicyError", "fetch_public_text"]
