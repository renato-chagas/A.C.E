"""Loopback routes for creating and reviewing code improvement proposals."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Awaitable, Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path as FilePath
from threading import Lock
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Message

from openjarvis.core.paths import ConfigurationError, get_config_dir
from openjarvis.engine._pinned_inference import pinned_local_inference
from openjarvis.learning.code_improvement import (
    CodeImprovementManager,
    assert_no_secrets,
    generate_local_patch,
)
from openjarvis.server.api_routes import _allows_private_memory_access

_NO_STORE = {"Cache-Control": "no-store"}
_UNAVAILABLE = "Code improvement storage is unavailable."
_INVALID = "Invalid code improvement request."
_BODY_LIMIT = 128 * 1024
_PROPOSAL_LOCK = Lock()


class _PrivateReviewRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        handler = super().get_route_handler()

        async def restricted(request: Request) -> Response:
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

        return restricted


def _bounded_body(request: Request) -> Request:
    try:
        size = int(request.headers.get("content-length", "0"))
    except ValueError:
        size = -1
    if size < 0:
        raise HTTPException(400, _INVALID)
    if size > _BODY_LIMIT:
        raise HTTPException(413, _INVALID)

    received = 0

    async def receive() -> Message:
        nonlocal received
        message = await request.receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > _BODY_LIMIT:
                raise HTTPException(413, _INVALID)
        return message

    return Request(request.scope, receive)


class CodeImprovementRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    repository: str = Field(min_length=1, max_length=1024)
    task: str = Field(min_length=1, max_length=4000)
    files: list[Annotated[str, StringConstraints(min_length=1, max_length=512)]] = (
        Field(min_length=1, max_length=24)
    )


class CodeImprovementSummary(BaseModel):
    proposal_id: str
    title: str
    status: str
    created_at: float = Field(allow_inf_nan=False)
    files: list[str]
    rollback_available: bool


class CodeImprovementDetail(CodeImprovementSummary):
    repository: str
    patch: str
    validation: str
    validation_receipt: str


class CodeImprovementList(BaseModel):
    proposals: list[CodeImprovementSummary]


@contextmanager
def _read_proposals() -> Iterator[sqlite3.Connection | None]:
    try:
        path = get_config_dir() / "improvements.db"
        if not path.exists():
            yield None
            return
        # Keep SQLite WAL visibility and forbid initialization or recovery.
        uri = path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=2.0)) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN")
            yield db
    except (ConfigurationError, OSError, sqlite3.Error, ValueError):
        raise HTTPException(503, _UNAVAILABLE, headers=_NO_STORE) from None


def _select_proposals(db: sqlite3.Connection, *, detail: bool = False) -> str | None:
    columns = {
        row["name"]
        for row in db.execute("PRAGMA table_info(code_improvement_proposals)")
    }
    if not columns:
        return None
    journal_exists = db.execute(
        "SELECT 1 FROM sqlite_schema "
        "WHERE type='table' AND name='code_improvement_apply_journals'"
    ).fetchone()
    rollback = (
        "p.status IN ('applied', 'rolling_back') AND EXISTS (SELECT 1 FROM "
        "code_improvement_apply_journals AS j WHERE j.proposal_id=p.proposal_id)"
        if journal_exists is not None
        else "0"
    )
    fields = (
        "p.proposal_id, p.title, p.status, p.created_at, p.files_json, "
        f"({rollback}) AS rollback_available"
    )
    if detail:
        fields += ", p.repository, p.patch"
        for column in ("validation", "validation_receipt"):
            fields += f", p.{column}" if column in columns else f", '' AS {column}"
    return f"SELECT {fields} FROM code_improvement_proposals AS p"


def _summary(row: sqlite3.Row) -> CodeImprovementSummary:
    return CodeImprovementSummary(
        proposal_id=row["proposal_id"],
        title=row["title"],
        status=row["status"],
        created_at=row["created_at"],
        files=json.loads(row["files_json"]),
        rollback_available=bool(row["rollback_available"]),
    )


router = APIRouter(
    prefix="/v1/code-improvements",
    tags=["code improvements"],
    route_class=_PrivateReviewRoute,
)


@router.post("", status_code=201)
def create_code_improvement(
    body: CodeImprovementRequest, request: Request
) -> CodeImprovementDetail:
    root = FilePath(body.repository)
    if not root.is_absolute() or root.drive.startswith(("\\", "//")):
        raise HTTPException(422, "Enter the absolute path to a local Git checkout.")
    if not _PROPOSAL_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Another code proposal is already being prepared.")
    try:
        with pinned_local_inference(
            request.app.state, capability="completion"
        ) as model:
            if model is None:
                raise HTTPException(503, "Select and start a supported local model.")
            try:
                if not (root / ".git").exists():
                    raise HTTPException(
                        422, "Enter the path to an existing Git checkout."
                    )
                assert_no_secrets(body.task)
                patch, hashes = generate_local_patch(
                    body.task,
                    tuple(dict.fromkeys(body.files)),
                    root,
                    pinned_model=model,
                )
                # Preparing a draft must not resume a previous apply or rollback.
                manager = CodeImprovementManager(recover=False)
                proposal = manager.propose(
                    root, body.task, patch, expected_source_hashes=hashes
                )
            except ValueError:
                raise HTTPException(
                    422, "Check the selected source files and request a smaller change."
                ) from None
            except (OSError, RuntimeError, sqlite3.Error):
                raise HTTPException(
                    503, "Could not prepare a proposal with the local model and Git."
                ) from None
        try:
            proposal = manager.validate(proposal.proposal_id)
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            # Keep the saved diff as a draft when isolated validation fails.
            pass
        return CodeImprovementDetail(rollback_available=False, **asdict(proposal))
    finally:
        _PROPOSAL_LOCK.release()


@router.get("")
def list_code_improvements(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> CodeImprovementList:
    with _read_proposals() as db:
        query = _select_proposals(db) if db is not None else None
        if db is None or query is None:
            return CodeImprovementList(proposals=[])
        rows = db.execute(
            query + " ORDER BY p.created_at DESC, p.proposal_id LIMIT ?", (limit,)
        )
        return CodeImprovementList(proposals=[_summary(row) for row in rows])


@router.get("/{proposal_id}")
def read_code_improvement(
    proposal_id: Annotated[str, Path(pattern=r"^[a-f0-9]{12}$")],
) -> CodeImprovementDetail:
    with _read_proposals() as db:
        query = _select_proposals(db, detail=True) if db is not None else None
        if db is not None and query is not None:
            row = db.execute(
                query + " WHERE p.proposal_id = ?", (proposal_id,)
            ).fetchone()
            if row is not None:
                return CodeImprovementDetail(
                    **_summary(row).model_dump(),
                    repository=row["repository"],
                    patch=row["patch"],
                    validation=row["validation"],
                    validation_receipt=row["validation_receipt"],
                )
    raise HTTPException(404, "Code improvement proposal not found.", headers=_NO_STORE)
