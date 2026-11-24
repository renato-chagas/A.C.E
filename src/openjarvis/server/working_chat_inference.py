"""Local model turns backed by the terminal and desktop shared chat store."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from openjarvis.core.paths import ConfigurationError
from openjarvis.core.types import Message, Role
from openjarvis.engine._locality import is_local_engine, is_local_engine_for_model
from openjarvis.memory.service import _build_secret_scanner, is_complete_text_reply
from openjarvis.server.routes import _ensure_identity_prompt, _record_completed_exchange
from openjarvis.server.working_chat_routes import (
    WorkingChatRoute,
    WorkingChatVersionRequest,
    chat_snapshot,
)
from openjarvis.sessions.working_chat import (
    SavedChat,
    SavedChatChanged,
    WorkingChatStore,
)

router = APIRouter(
    prefix="/v1/working-chats", tags=["working chats"], route_class=WorkingChatRoute
)


class WorkingChatTurnRequest(WorkingChatVersionRequest):
    message: str = Field(min_length=1, max_length=8_000)


def _messages_for_turn(chat: SavedChat, text: str, state: Any) -> list[Message]:
    messages = [
        message
        for question, answer in chat.turns
        for message in (
            Message(role=Role.USER, content=question),
            Message(role=Role.ASSISTANT, content=answer),
        )
    ]
    messages.append(Message(role=Role.USER, content=text))
    config = getattr(state, "config", None)
    messages = _ensure_identity_prompt(messages, config)
    if config is None or not config.agent.context_from_memory:
        return messages

    try:
        from openjarvis.tools.storage.context import ContextConfig, inject_context

        service = getattr(state, "memory_service", None)
        return inject_context(
            text,
            messages,
            getattr(state, "memory_backend", None),
            config=ContextConfig.from_memory_config(config.memory),
            facts=service.list_facts() if service is not None else [],
        )
    except Exception:
        # The retained working history is usable when optional recall is down.
        return messages


def _can_save(text: str, answer: str) -> bool:
    try:
        scanner = _build_secret_scanner()
        return (
            scanner is not None and scanner.scan(f"{text}\n{answer}").is_clean is True
        )
    except Exception:
        return False


@router.post("/turn")
def working_chat_turn(body: WorkingChatTurnRequest, request: Request) -> dict:
    """Complete a text turn locally and save it only against its current revision."""
    text = body.message.strip()
    if not text:
        raise HTTPException(422, "Enter a message.")

    state = request.app.state
    engine = getattr(state, "engine", None)
    engine_key = getattr(state, "engine_name", "") or ""
    model = getattr(state, "model", "") or ""
    # Verify the whole engine, as well as the model's route. A mixed router
    # may refresh its model map concurrently and route old history elsewhere.
    if (
        engine is None
        or not model
        or not is_local_engine(engine, engine_key)
        or not is_local_engine_for_model(engine, model, engine_key)
    ):
        raise HTTPException(503, "A verified local model is required for shared chats.")

    try:
        store = WorkingChatStore(create=False)
        chat = store.read(body.name)
    except (ConfigurationError, OSError, sqlite3.Error, ValueError):
        raise HTTPException(503, "Could not read the saved chat.") from None
    if chat is None or (chat.session_id, chat.revision) != (
        body.session_id,
        body.revision,
    ):
        raise HTTPException(409, "The saved chat changed. Reopen it before sending.")

    prefs = getattr(getattr(state, "config", None), "intelligence", None)
    try:
        messages = _messages_for_turn(chat, text, state)
        result = engine.generate(
            messages,
            model=model,
            temperature=getattr(prefs, "temperature", 0.7),
            max_tokens=getattr(prefs, "max_tokens", 1_024),
        )
    except Exception:
        raise HTTPException(502, "The local model could not respond.") from None

    answer = result.get("content", "") if isinstance(result, dict) else ""
    if not isinstance(answer, str) or not answer.strip():
        raise HTTPException(502, "The local model returned no text.")

    if not is_complete_text_reply(result):
        return {
            "chat": chat_snapshot(chat),
            "answer": answer,
            "saved": False,
            "reason": "incomplete",
        }
    if not _can_save(text, answer):
        return {
            "chat": chat_snapshot(chat),
            "answer": answer,
            "saved": False,
            "reason": "scanner",
        }
    try:
        updated = store.append(chat, text, answer)
    except SavedChatChanged:
        return {"chat": None, "answer": answer, "saved": False, "reason": "changed"}
    except (OSError, sqlite3.Error, ValueError):
        return {"chat": None, "answer": answer, "saved": False, "reason": "storage"}

    _record_completed_exchange(
        getattr(state, "memory_service", None),
        text,
        answer,
        bus=getattr(state, "bus", None),
        source="desktop.shared_chat",
        session_id=updated.session_id,
    )
    return {"chat": chat_snapshot(updated), "answer": answer, "saved": True}
