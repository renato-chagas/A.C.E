"""Local-only endpoints for the shared CLI and desktop working chat store."""

from __future__ import annotations

import sqlite3
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Message

from openjarvis.core.paths import ConfigurationError
from openjarvis.server.api_routes import _allows_private_memory_access
from openjarvis.sessions.working_chat import (
    SavedChat,
    WorkingChatStore,
    validate_chat_name,
)

_WORKING_CHAT_BODY_LIMIT = 32 * 1024
_NO_STORE = {"Cache-Control": "no-store"}
_UNAVAILABLE = "Working chat storage is unavailable."


class WorkingChatRoute(APIRoute):
    """Guard private routes before parsing and avoid exposing submitted text."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        handler = super().get_route_handler()

        async def guarded(request: Request) -> Response:
            if not _allows_private_memory_access(request):
                return JSONResponse(
                    {"detail": "Direct local access is required."},
                    status_code=403,
                    headers=_NO_STORE,
                )

            try:
                response = await handler(_bounded_request(request))
            except RequestValidationError:
                return JSONResponse(
                    {"detail": "Invalid working chat request."},
                    status_code=422,
                    headers=_NO_STORE,
                )
            except ResponseValidationError:
                return JSONResponse(
                    {"detail": _UNAVAILABLE}, status_code=503, headers=_NO_STORE
                )
            except StarletteHTTPException as exc:
                exc.headers = {**(exc.headers or {}), **_NO_STORE}
                raise
            response.headers.update(_NO_STORE)
            return response

        return guarded


def _bounded_request(request: Request) -> Request:
    raw_length = request.headers.get("content-length")
    if raw_length is not None:
        try:
            length = int(raw_length)
        except ValueError:
            length = -1
        if length < 0:
            raise HTTPException(400, detail="Invalid working chat request.")
        if length > _WORKING_CHAT_BODY_LIMIT:
            raise HTTPException(413, detail="Working chat request is too large.")

    received = 0

    async def receive() -> Message:
        nonlocal received
        message = await request.receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > _WORKING_CHAT_BODY_LIMIT:
                raise HTTPException(413, detail="Working chat request is too large.")
        return message

    return Request(request.scope, receive)


class WorkingChatNameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def valid_name(cls, name: str) -> str:
        return validate_chat_name(name)


class WorkingChatVersionRequest(WorkingChatNameRequest):
    session_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    revision: int = Field(ge=0, le=2**63 - 1)


class WorkingChatSnapshot(BaseModel):
    name: str
    session_id: str
    revision: int
    turns: list[tuple[str, str]]


def chat_snapshot(chat: SavedChat) -> dict[str, str | int | list[list[str]]]:
    return {
        "name": chat.name,
        "session_id": chat.session_id,
        "revision": chat.revision,
        "turns": [[question, answer] for question, answer in chat.turns],
    }


@contextmanager
def _chat_store(*, create: bool = False) -> Iterator[WorkingChatStore]:
    try:
        yield WorkingChatStore(create=create)
    except (ConfigurationError, OSError, sqlite3.Error, ValueError) as exc:
        raise HTTPException(503, detail=_UNAVAILABLE) from exc


router = APIRouter(
    prefix="/v1/working-chats", tags=["working chats"], route_class=WorkingChatRoute
)


@router.get("")
def list_working_chats() -> dict[str, list[str]]:
    with _chat_store() as store:
        return {"chats": store.list_names()}


@router.post("/read", response_model=WorkingChatSnapshot)
def read_working_chat(
    payload: WorkingChatNameRequest,
) -> dict[str, str | int | list[list[str]]]:
    with _chat_store() as store:
        chat = store.read(payload.name)
    if chat is None:
        raise HTTPException(404, detail="Working chat not found.")
    return chat_snapshot(chat)


@router.post("/open", response_model=WorkingChatSnapshot)
def open_working_chat(
    payload: WorkingChatNameRequest,
) -> dict[str, str | int | list[list[str]]]:
    with _chat_store(create=True) as store:
        return chat_snapshot(store.open(payload.name))


@router.post("/forget")
def forget_working_chat(
    payload: WorkingChatVersionRequest,
) -> dict[str, Literal[True]]:
    snapshot = SavedChat(payload.name, payload.session_id, payload.revision, ())
    with _chat_store() as store:
        deleted = store.forget_if(snapshot)
    if not deleted:
        raise HTTPException(409, detail="Working chat changed. Read it again.")
    return {"deleted": True}
