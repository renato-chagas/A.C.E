"""Local-first desktop access to ACE's shared web-source archive."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from threading import BoundedSemaphore
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Message

from openjarvis.connectors.store import KnowledgeStore
from openjarvis.core.paths import ConfigurationError, get_config_dir
from openjarvis.engine._pinned_inference import pinned_local_inference
from openjarvis.memory.graph_extractor import DocumentGraphExtractor
from openjarvis.memory.service import _build_scanner, _build_secret_scanner
from openjarvis.security.public_web import PublicWebPolicyError
from openjarvis.server.api_routes import _allows_private_memory_access
from openjarvis.tools.web_research import WebResearchTool, _canonical_url
from openjarvis.tools.web_search import WebSearchTool

_NO_STORE = {"Cache-Control": "no-store"}
_INVALID = "Invalid web research request."
_UNAVAILABLE = "The local web-source archive is unavailable."
_MAX_QUERY = 2048
_MAX_BODY = 16 * 1024
_RESEARCH_SLOTS = BoundedSemaphore(2)
_EXTRACTION_SLOT = BoundedSemaphore(1)
_STALE_CAPTURE = "The saved page has changed. Reload the source before extracting."
_URL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def _bounded_body(request: Request) -> Request:
    length = request.headers.get("content-length")
    if length is not None:
        try:
            size = int(length)
        except ValueError:
            size = -1
        if size < 0:
            raise HTTPException(400, _INVALID)
        if size > _MAX_BODY:
            raise HTTPException(413, _INVALID)

    received = 0

    async def receive() -> Message:
        nonlocal received
        message = await request.receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > _MAX_BODY:
                raise HTTPException(413, _INVALID)
        return message

    return Request(request.scope, receive)


class _PrivateWebLibraryRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        handler = super().get_route_handler()

        async def guarded(request: Request) -> Response:
            try:
                if not _allows_private_memory_access(request):
                    raise HTTPException(403, "Direct local access is required.")
                if request.method == "POST":
                    media_type = request.headers.get("content-type", "").split(";", 1)[
                        0
                    ]
                    if media_type.strip().lower() != "application/json":
                        raise HTTPException(415, "Use an application/json request.")
                    request = _bounded_body(request)
                response = await handler(request)
            except RequestValidationError:
                raise HTTPException(422, _INVALID, headers=_NO_STORE) from None
            except StarletteHTTPException as exc:
                exc.headers = {**(exc.headers or {}), **_NO_STORE}
                raise
            response.headers.update(_NO_STORE)
            return response

        return guarded


class WebResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=_MAX_QUERY)


class WebSource(BaseModel):
    url: str
    revision: str
    title: str
    saved_at: str
    text: str
    capture: Literal["page", "snippet", "unknown"]


class WebExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    url: str = Field(min_length=1, max_length=_MAX_QUERY)
    revision: str = Field(min_length=1, max_length=128)


class WebExtractionResult(BaseModel):
    url: str
    saved_at: str
    entities_stored: int
    relations_stored: int


class WebLibrary(BaseModel):
    sources: list[WebSource]


class WebResearchResult(WebLibrary):
    offline_fallback: bool


def _database_paths(request: Request) -> tuple[Path, Path]:
    root = get_config_dir()
    config = getattr(request.app.state, "config", None)
    storage = config.tools.storage if config is not None else None
    knowledge = getattr(storage, "knowledge_db_path", "") or root / "knowledge.db"
    graph = getattr(storage, "graph_db_path", "") or root / "knowledge_graph.db"
    return Path(knowledge).expanduser(), Path(graph).expanduser()


def _load_captures(
    path: Path, query: str = "", urls: tuple[str, ...] = ()
) -> list[WebSource]:
    if not urls and _URL_SCHEME.match(query):
        try:
            urls = (_canonical_url(query),)
        except ValueError:
            return []
    rows = KnowledgeStore.read_web_captures(path, query=query, urls=urls)
    return [WebSource.model_validate(row) for row in rows]


router = APIRouter(
    prefix="/v1/web-library",
    tags=["web research"],
    route_class=_PrivateWebLibraryRoute,
)


@router.get("")
def read_web_library(
    request: Request, q: Annotated[str, Query(max_length=_MAX_QUERY)] = ""
) -> WebLibrary:
    try:
        path, _ = _database_paths(request)
        return WebLibrary(sources=_load_captures(path, q.strip()))
    except (ConfigurationError, OSError, sqlite3.Error, ValueError):
        raise HTTPException(503, _UNAVAILABLE) from None


@router.post("")
def search_and_save(body: WebResearchRequest, request: Request) -> WebResearchResult:
    if _URL_SCHEME.match(body.query) and not body.query.lower().startswith(
        ("http://", "https://")
    ):
        raise HTTPException(422, "Enter a public HTTP or HTTPS source URL.")
    if not _RESEARCH_SLOTS.acquire(blocking=False):
        raise HTTPException(429, "Web research is already in progress.")
    try:
        knowledge, graph = _database_paths(request)
        try:
            result = WebResearchTool(
                knowledge_db_path=str(knowledge),
                graph_db_path=str(graph),
                direct_public_capture=True,
            ).execute(query=body.query, max_results=3)
        except PublicWebPolicyError:
            # A previously archived URL can still be opened when DNS validation
            # cannot complete. No connection is made to the rejected destination.
            requested = WebSearchTool._extract_url(body.query) or body.query
            archived = _load_captures(knowledge, requested)
            if archived:
                return WebResearchResult(sources=archived, offline_fallback=True)
            raise HTTPException(
                422, "Enter a supported public web source URL."
            ) from None
        urls = tuple(result.metadata.get("source_urls", ()))
        offline = not result.success or not urls
        fallback_url = result.metadata.get("lookup_url")
        if offline and fallback_url:
            urls = (fallback_url,)
        sources = _load_captures(knowledge, body.query if offline else "", urls)
        return WebResearchResult(sources=sources, offline_fallback=offline)
    except (ConfigurationError, OSError, sqlite3.Error, ValueError, RuntimeError):
        raise HTTPException(503, _UNAVAILABLE) from None
    finally:
        _RESEARCH_SLOTS.release()


@router.post("/extract")
def extract_web_source(
    body: WebExtractionRequest, request: Request
) -> WebExtractionResult:
    """Read one archived revision and add its quoted claims to the local graph."""
    try:
        url = _canonical_url(body.url)
    except ValueError:
        raise HTTPException(422, "Choose a saved web source.") from None
    if not _EXTRACTION_SLOT.acquire(blocking=False):
        raise HTTPException(429, "Another source is being analyzed.")
    try:
        knowledge, graph = _database_paths(request)
        rows = KnowledgeStore.read_web_captures(knowledge, urls=(url,), limit=1)
        if not rows:
            raise HTTPException(404, "The saved source could not be found.")
        source = rows[0]
        if source["revision"] != body.revision:
            raise HTTPException(409, _STALE_CAPTURE)
        if source["capture"] != "page" or not source["text"].strip():
            raise HTTPException(422, "Save the page text before extracting claims.")
        try:
            captured = datetime.fromisoformat(source["saved_at"])
            if captured.tzinfo is None:
                captured = captured.replace(tzinfo=timezone.utc)
            observed_at = captured.timestamp()
        except (OSError, OverflowError, TypeError, ValueError):
            raise HTTPException(422, "This capture has no usable date.") from None

        scanners = (_build_secret_scanner(), _build_scanner())
        if any(scanner is None for scanner in scanners):
            raise HTTPException(503, "Local content safeguards are unavailable.")

        def is_safe(text: str) -> bool:
            try:
                return all(scanner.scan(text).is_clean is True for scanner in scanners)
            except Exception:
                return False

        if not is_safe(source["text"]):
            raise HTTPException(422, "Local content safeguards rejected this capture.")

        with pinned_local_inference(
            request.app.state, capability="completion"
        ) as runtime:
            if runtime is None:
                raise HTTPException(503, "A supported local text model is required.")
            extractor = DocumentGraphExtractor(*runtime)
            try:
                extraction = extractor.extract(source["text"], strict=True)
            except (RuntimeError, ValueError):
                raise HTTPException(
                    502, "The local model could not analyze this capture."
                ) from None
        latest = KnowledgeStore.read_web_captures(knowledge, urls=(url,), limit=1)
        if not latest or latest[0]["revision"] != body.revision:
            raise HTTPException(409, _STALE_CAPTURE)
        entities, relations = extractor.persist(
            extraction,
            graph,
            source_id=f"web:{body.revision}",
            source_kind="web",
            source_url=url,
            observed_at=observed_at,
            quote_is_safe=is_safe,
        )
        return WebExtractionResult(
            url=url,
            saved_at=source["saved_at"],
            entities_stored=entities,
            relations_stored=relations,
        )
    except (ConfigurationError, OSError, sqlite3.Error, ValueError, RuntimeError):
        raise HTTPException(503, _UNAVAILABLE) from None
    finally:
        _EXTRACTION_SLOT.release()
