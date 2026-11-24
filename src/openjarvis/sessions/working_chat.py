"""Bounded, explicitly named chat history on the local filesystem."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from openjarvis.core.paths import get_config_dir
from openjarvis.security.file_utils import secure_create

_MAX_TURNS = 12
_MAX_MESSAGE_CHARS = 3_000
_MAX_HISTORY_CHARS = 8_000
_TRUNCATED = "\n[Saved text shortened.]"

Turn = tuple[str, str]


def validate_chat_name(name: str) -> str:
    """Allow printable session names that are never used as file paths."""
    name = name.strip()
    if not name or len(name) > 80 or not name.isprintable():
        raise ValueError("Use a single line of 1 to 80 printable characters.")
    return name


@dataclass(frozen=True, slots=True)
class SavedChat:
    name: str
    session_id: str
    revision: int
    turns: tuple[Turn, ...]


class SavedChatChanged(RuntimeError):
    """The chat was updated or deleted in another terminal."""


def working_chat_path() -> Path:
    return get_config_dir() / "working_chats.db"


class WorkingChatStore:
    """Store text-only complete turns, independently of the memory archive."""

    def __init__(self, path: Path | None = None, *, create: bool = True) -> None:
        self.path = path if path is not None else working_chat_path()
        if create:
            secure_create(self.path)
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS working_chats (
                        name TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        revision INTEGER NOT NULL DEFAULT 0,
                        turns_json TEXT NOT NULL DEFAULT '[]',
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.commit()

    def _connect(self) -> sqlite3.Connection:
        # rw also prevents accidental creation if a management command races
        # with removal of the database.
        uri = self.path.resolve().as_uri() + "?mode=rw"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _read_chat(row: sqlite3.Row) -> SavedChat:
        turns = json.loads(row["turns_json"])
        if (
            not isinstance(turns, list)
            or len(turns) > _MAX_TURNS
            or any(
                not isinstance(turn, list)
                or len(turn) != 2
                or any(
                    not isinstance(text, str) or len(text) > _MAX_MESSAGE_CHARS
                    for text in turn
                )
                for turn in turns
            )
            or sum(len(user) + len(answer) for user, answer in turns)
            > _MAX_HISTORY_CHARS
        ):
            raise ValueError("Invalid saved chat history.")
        return SavedChat(
            row["name"],
            row["session_id"],
            row["revision"],
            tuple((user, answer) for user, answer in turns),
        )

    def open(self, name: str) -> SavedChat:
        name = validate_chat_name(name)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR IGNORE INTO working_chats(name, session_id, updated_at) "
                "VALUES (?, ?, ?)",
                (name, uuid.uuid4().hex, time.time()),
            )
            row = conn.execute(
                "SELECT * FROM working_chats WHERE name=?", (name,)
            ).fetchone()
            return self._read_chat(row)

    def read(self, name: str) -> SavedChat | None:
        """Read a named chat without creating or modifying it."""
        name = validate_chat_name(name)
        if not self.path.is_file():
            return None
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM working_chats WHERE name=?", (name,)
            ).fetchone()
        return self._read_chat(row) if row is not None else None

    def list_names(self) -> list[str]:
        if not self.path.is_file():
            return []
        with closing(self._connect()) as conn:
            return [
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM working_chats ORDER BY updated_at DESC, name"
                )
            ]

    def forget(self, name: str) -> bool:
        return self._forget(validate_chat_name(name))

    def forget_if(self, snapshot: SavedChat) -> bool:
        """Delete only the exact session and revision given by the caller."""
        return self._forget(
            validate_chat_name(snapshot.name), (snapshot.session_id, snapshot.revision)
        )

    def _forget(self, name: str, version: tuple[str, int] | None = None) -> bool:
        if not self.path.is_file():
            return False
        query = "DELETE FROM working_chats WHERE name=?"
        values: tuple[str | int, ...] = (name,)
        if version is not None:
            query += " AND session_id=? AND revision=?"
            values += version
        with closing(self._connect()) as conn, conn:
            conn.execute("PRAGMA secure_delete=ON")
            return conn.execute(query, values).rowcount == 1

    def append(self, chat: SavedChat, user: str, answer: str) -> SavedChat:
        def clip(text: str) -> str:
            if len(text) <= _MAX_MESSAGE_CHARS:
                return text
            return text[: _MAX_MESSAGE_CHARS - len(_TRUNCATED)] + _TRUNCATED

        new_turn = (clip(user), clip(answer))
        turns = (*chat.turns, new_turn)[-_MAX_TURNS:]
        while sum(len(question) + len(reply) for question, reply in turns) > (
            _MAX_HISTORY_CHARS
        ):
            turns = turns[1:]
        return self._write(chat, chat.session_id, turns)

    def clear(self, chat: SavedChat) -> SavedChat:
        return self._write(chat, uuid.uuid4().hex, ())

    def _write(
        self, chat: SavedChat, session_id: str, turns: tuple[Turn, ...]
    ) -> SavedChat:
        revision = chat.revision + 1
        with closing(self._connect()) as conn, conn:
            conn.execute("PRAGMA secure_delete=ON")
            updated = conn.execute(
                "UPDATE working_chats "
                "SET session_id=?, revision=?, turns_json=?, updated_at=? "
                "WHERE name=? AND session_id=? AND revision=?",
                (
                    session_id,
                    revision,
                    json.dumps(turns, ensure_ascii=False),
                    time.time(),
                    chat.name,
                    chat.session_id,
                    chat.revision,
                ),
            )
            if updated.rowcount != 1:
                raise SavedChatChanged("Saved chat changed in another terminal.")
        return SavedChat(chat.name, session_id, revision, turns)
