"""SQLite-backed persistence for scheduled tasks and run logs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

_CREATE_TASKS_TABLE = """\
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id              TEXT PRIMARY KEY,
    prompt          TEXT    NOT NULL,
    schedule_type   TEXT    NOT NULL,
    schedule_value  TEXT    NOT NULL,
    context_mode    TEXT    NOT NULL DEFAULT 'isolated',
    status          TEXT    NOT NULL DEFAULT 'active',
    next_run        TEXT,
    last_run        TEXT,
    agent           TEXT    NOT NULL DEFAULT 'simple',
    tools           TEXT    NOT NULL DEFAULT '',
    metadata        TEXT    NOT NULL DEFAULT '{}'
);
"""

_CREATE_LOGS_TABLE = """\
CREATE TABLE IF NOT EXISTS task_run_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT    NOT NULL,
    started_at  TEXT    NOT NULL,
    finished_at TEXT,
    success     INTEGER NOT NULL DEFAULT 0,
    result      TEXT    NOT NULL DEFAULT '',
    error       TEXT    NOT NULL DEFAULT '',
    status      TEXT    NOT NULL DEFAULT ''
);
"""

_INSERT_TASK = """\
INSERT OR REPLACE INTO scheduled_tasks
    (id, prompt, schedule_type, schedule_value, context_mode,
     status, next_run, last_run, agent, tools, metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_LOG = """\
INSERT INTO task_run_logs
    (task_id, started_at, finished_at, success, result, error, status)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""


class SchedulerStore:
    """SQLite CRUD store for scheduled tasks and their run logs."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            from openjarvis.security.file_utils import secure_create

            secure_create(Path(self._db_path))
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(_CREATE_TASKS_TABLE)
            self._conn.execute(_CREATE_LOGS_TABLE)
            columns = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(task_run_logs)")
            }
            if "status" not in columns:
                self._conn.execute(
                    "ALTER TABLE task_run_logs "
                    "ADD COLUMN status TEXT NOT NULL DEFAULT ''"
                )
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS task_run_running "
                "ON task_run_logs (task_id) WHERE status = 'running'"
            )

    @property
    def db_path(self) -> str:
        """Return the persistent database path used to coordinate schedulers."""
        return self._db_path

    # -- Task CRUD -----------------------------------------------------------

    def save_task(self, task: Dict[str, Any]) -> None:
        """Insert or replace a scheduled task record."""
        self._conn.execute(
            _INSERT_TASK,
            (
                task["id"],
                task["prompt"],
                task["schedule_type"],
                task["schedule_value"],
                task.get("context_mode", "isolated"),
                task.get("status", "active"),
                task.get("next_run"),
                task.get("last_run"),
                task.get("agent", "simple"),
                task.get("tools", ""),
                json.dumps(task.get("metadata", {})),
            ),
        )
        self._conn.commit()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single task by ID, or ``None`` if not found."""
        row = self._conn.execute(
            "SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all tasks, optionally filtered by *status*."""
        if status is not None:
            rows = self._conn.execute(
                "SELECT * FROM scheduled_tasks WHERE status = ?", (status,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM scheduled_tasks").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_due_tasks(self, now_iso: str) -> List[Dict[str, Any]]:
        """Return active tasks whose ``next_run`` is at or before *now_iso*."""
        rows = self._conn.execute(
            "SELECT * FROM scheduled_tasks WHERE status = 'active' "
            "AND next_run IS NOT NULL AND next_run <= ? "
            "AND NOT EXISTS (SELECT 1 FROM task_run_logs "
            "WHERE task_id = scheduled_tasks.id AND status = 'running') "
            "ORDER BY next_run, id",
            (now_iso,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_task(self, task: Dict[str, Any]) -> None:
        """Update an existing task (same as save_task — uses INSERT OR REPLACE)."""
        self.save_task(task)

    def transition_task(
        self,
        expected: Dict[str, Any],
        *,
        status: str,
        next_run: str | None,
    ) -> None:
        """Change task state without overwriting a concurrent run or decision."""
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE scheduled_tasks SET status=?, next_run=? "
                "WHERE id=? AND status=? AND next_run IS ? AND last_run IS ?",
                (
                    status,
                    next_run,
                    expected["id"],
                    expected["status"],
                    expected["next_run"],
                    expected["last_run"],
                ),
            )
            if not cursor.rowcount:
                raise RuntimeError("Task changed; review its current status and retry.")

    def delete_task(self, task_id: str) -> None:
        """Delete a task by ID."""
        self._conn.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        self._conn.commit()

    # -- Run logs ------------------------------------------------------------

    def has_running_run(self, task_id: str) -> bool:
        """Return whether an occurrence is still claimed."""
        return (
            self._conn.execute(
                "SELECT 1 FROM task_run_logs WHERE task_id=? AND status='running'",
                (task_id,),
            ).fetchone()
            is not None
        )

    def start_run(
        self, task_id: str, scheduled_at: str | None, started_at: str
    ) -> tuple[int, Dict[str, Any]] | None:
        """Claim the expected task occurrence before invoking its runtime."""
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            task = self._conn.execute(
                "SELECT * FROM scheduled_tasks WHERE id=? AND status='active' "
                "AND next_run IS ? AND NOT EXISTS (SELECT 1 FROM task_run_logs "
                "WHERE task_id=? AND status='running')",
                (task_id, scheduled_at, task_id),
            ).fetchone()
            if task is None:
                return None
            cursor = self._conn.execute(
                "INSERT INTO task_run_logs (task_id, started_at, status) "
                "VALUES (?, ?, 'running')",
                (task_id, started_at),
            )
            assert cursor.lastrowid is not None
            return cursor.lastrowid, self._row_to_dict(task)

    def finish_run(
        self,
        task_id: str,
        run_id: int,
        *,
        scheduled_at: str | None,
        finished_at: str,
        next_run: str | None,
        success: bool,
        result: str,
        error: str,
    ) -> None:
        """Commit the outcome and advance the schedule in the same transaction."""
        with self._conn:
            finished = self._conn.execute(
                "UPDATE task_run_logs SET finished_at=?, success=?, result=?, "
                "error=?, status=? WHERE id=? AND task_id=? AND status='running'",
                (
                    finished_at,
                    int(success),
                    result,
                    error,
                    "succeeded" if success else "failed",
                    run_id,
                    task_id,
                ),
            )
            if not finished.rowcount:
                raise RuntimeError("The scheduled run is no longer claimed.")
            self._conn.execute(
                "UPDATE scheduled_tasks SET last_run=?, next_run=?, "
                "status=CASE WHEN status='paused' THEN status ELSE ? END "
                "WHERE id=? AND status IN ('active', 'paused') AND next_run IS ?",
                (
                    finished_at,
                    next_run,
                    "active" if next_run else "completed",
                    task_id,
                    scheduled_at,
                ),
            )

    def recover_interrupted_runs(self, detected_at: str) -> int:
        """Record unknown outcomes. Call only after acquiring the scheduler lease."""
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(
                "UPDATE scheduled_tasks SET status='review_required', next_run=NULL "
                "WHERE status IN ('active', 'paused') AND EXISTS "
                "(SELECT 1 FROM task_run_logs WHERE task_id=scheduled_tasks.id "
                "AND status='running')"
            )
            cursor = self._conn.execute(
                "UPDATE task_run_logs SET status='interrupted', finished_at=?, "
                "error='Outcome unknown after scheduler interruption' "
                "WHERE status='running'",
                (detected_at,),
            )
            return cursor.rowcount

    def log_run(
        self,
        task_id: str,
        started_at: str,
        finished_at: str,
        success: bool,
        result: str = "",
        error: str = "",
    ) -> None:
        """Record a single execution of a task."""
        self._conn.execute(
            _INSERT_LOG,
            (
                task_id,
                started_at,
                finished_at,
                int(success),
                result,
                error,
                "succeeded" if success else "failed",
            ),
        )
        self._conn.commit()

    def get_run_logs(self, task_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent run logs for *task_id*."""
        rows = self._conn.execute(
            "SELECT * FROM task_run_logs WHERE task_id = ? ORDER BY id DESC LIMIT ?",
            (task_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        if "metadata" in d and isinstance(d["metadata"], str):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = {}
        return d


__all__ = ["SchedulerStore"]
