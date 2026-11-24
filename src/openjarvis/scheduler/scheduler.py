"""Task scheduler — cron/interval/once execution with background polling."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from openjarvis.scheduler.instance_lock import SchedulerInstanceLock
from openjarvis.scheduler.store import SchedulerStore

logger = logging.getLogger(__name__)

# Event type strings (avoids editing core EventType enum)
SCHEDULER_TASK_START = "scheduler_task_start"
SCHEDULER_TASK_END = "scheduler_task_end"


def _result_text(result: Any) -> str:
    """Convert a system response to the text stored in a scheduled run log."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, str):
            return content
    return str(result)


@dataclass(slots=True)
class ScheduledTask:
    """A task scheduled for future or recurring execution."""

    id: str
    prompt: str
    schedule_type: str  # "cron" | "interval" | "once"
    schedule_value: str  # cron expression, interval seconds, ISO datetime
    context_mode: str = "isolated"
    status: str = "active"
    next_run: Optional[str] = None
    last_run: Optional[str] = None
    agent: str = "simple"
    tools: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for store persistence."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "schedule_type": self.schedule_type,
            "schedule_value": self.schedule_value,
            "context_mode": self.context_mode,
            "status": self.status,
            "next_run": self.next_run,
            "last_run": self.last_run,
            "agent": self.agent,
            "tools": self.tools,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ScheduledTask:
        """Deserialize from a plain dict."""
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            schedule_type=d["schedule_type"],
            schedule_value=d["schedule_value"],
            context_mode=d.get("context_mode", "isolated"),
            status=d.get("status", "active"),
            next_run=d.get("next_run"),
            last_run=d.get("last_run"),
            agent=d.get("agent", "simple"),
            tools=d.get("tools", ""),
            metadata=d.get("metadata", {}),
        )


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class TaskScheduler:
    """Scheduler that polls for due tasks and executes them.

    Parameters
    ----------
    store:
        The persistence backend.
    system:
        Optional ``JarvisSystem`` instance for executing prompts.
    poll_interval:
        Seconds between poll cycles (default 60).
    bus:
        Optional event bus for publishing scheduler events.
    """

    def __init__(
        self,
        store: SchedulerStore,
        system: Any = None,
        *,
        poll_interval: int = 60,
        bus: Any = None,
    ) -> None:
        self._store = store
        self._system = system
        self._poll_interval = poll_interval
        self._bus = bus
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        database_path = getattr(store, "db_path", "")
        self._instance_lock = (
            SchedulerInstanceLock(database_path)
            if database_path and database_path != ":memory:"
            else None
        )

    # -- Public API ----------------------------------------------------------

    def set_system(self, system: Any) -> None:
        """Attach the JarvisSystem used to execute due tasks.

        Bind before :meth:`start`; changing the runtime of a live scheduler is
        rejected so one task cannot observe a partially replaced system.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError(
                    "Cannot replace the system while the scheduler is running"
                )
            self._system = system

    def start(self) -> None:
        """Start the background polling daemon thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._instance_lock is not None and not self._instance_lock.acquire():
                raise RuntimeError(
                    "Another scheduler is already running for this database."
                )
            try:
                if self._instance_lock is not None:
                    self._instance_lock.clear_stop_request()
                self._recover_interrupted_runs()
                self._stop_event.clear()
                self._thread = threading.Thread(
                    target=self._poll_loop, daemon=True, name="jarvis-scheduler"
                )
                self._thread.start()
            except Exception:
                self._thread = None
                if self._instance_lock is not None:
                    self._instance_lock.release()
                raise
        logger.info("Scheduler started (poll_interval=%ds)", self._poll_interval)

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._poll_interval + 5)
            if thread.is_alive():
                logger.error(
                    "Scheduler thread did not stop; retaining its process lock"
                )
                return
            self._thread = None
        if self._instance_lock is not None:
            self._instance_lock.release()
        logger.info("Scheduler stopped")

    @property
    def external_stop_requested(self) -> bool:
        """Whether another process requested graceful scheduler shutdown."""
        return bool(
            self._instance_lock is not None and self._instance_lock.stop_requested()
        )

    def clear_external_stop_request(self) -> None:
        """Clear a completed external shutdown request."""
        if self._instance_lock is not None:
            self._instance_lock.clear_stop_request()

    def create_task(
        self,
        prompt: str,
        schedule_type: str,
        schedule_value: str,
        *,
        requires_approval: bool = False,
        **kwargs: Any,
    ) -> ScheduledTask:
        """Create and persist a new scheduled task."""
        raw_metadata = kwargs.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        if requires_approval:
            # Agent-created tasks must execute with the exact tool scope shown
            # to the user during approval, including an explicit empty scope.
            metadata["_openjarvis_explicit_tool_scope"] = True
        task = ScheduledTask(
            id=uuid.uuid4().hex[:16],
            prompt=prompt,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            agent=kwargs.get("agent", "simple"),
            tools=kwargs.get("tools", ""),
            context_mode=kwargs.get("context_mode", "isolated"),
            status="pending_approval" if requires_approval else "active",
            metadata=metadata,
        )
        task.next_run = self._compute_next_run(task)
        with self._lock:
            self._store.save_task(task.to_dict())
        return task

    def list_tasks(self, *, status: Optional[str] = None) -> List[ScheduledTask]:
        """Return tasks, optionally filtered by *status*."""
        with self._lock:
            rows = self._store.list_tasks(status=status)
        return [ScheduledTask.from_dict(r) for r in rows]

    def _existing_task(self, task_id: str) -> Dict[str, Any]:
        data = self._store.get_task(task_id)
        if data is None:
            raise KeyError(f"Task not found: {task_id}")
        return data

    def pause_task(self, task_id: str) -> None:
        """Pause an active task while preserving any occurrence in progress."""
        with self._lock:
            data = self._existing_task(task_id)
            if data["status"] == "pending_approval":
                raise RuntimeError(
                    "Task requires explicit approval; use 'jarvis scheduler approve'"
                )
            if data["status"] not in {"active", "paused"}:
                raise RuntimeError(f"Cannot pause a task with status {data['status']}.")
            self._store.transition_task(
                data, status="paused", next_run=data["next_run"]
            )

    def resume_task(self, task_id: str, *, reviewed_interruption: bool = False) -> None:
        """Resume a paused task or an explicitly reviewed recurring task."""
        with self._lock:
            data = self._existing_task(task_id)
            status = data["status"]
            if status == "pending_approval":
                raise RuntimeError(
                    "Task requires explicit approval; use 'jarvis scheduler approve'"
                )
            if status == "review_required":
                if not reviewed_interruption:
                    raise RuntimeError(
                        "Review the interrupted run in 'jarvis scheduler logs' "
                        "and use 'jarvis scheduler resume' to confirm."
                    )
                if data["schedule_type"] == "once":
                    raise RuntimeError(
                        "The one-time run might have taken effect. Review its outcome "
                        "and create a new task if it should run again."
                    )
            elif status != "paused":
                raise RuntimeError(f"Cannot resume a task with status {status}.")
            if self._store.has_running_run(task_id):
                raise RuntimeError(
                    "Wait for the current run to finish before resuming."
                )
            next_run = self._compute_next_run(ScheduledTask.from_dict(data))
            self._store.transition_task(
                data, status="active" if next_run else "completed", next_run=next_run
            )

    def approve_task(self, task_id: str) -> None:
        """Activate an agent-created task after an explicit user decision."""
        with self._lock:
            data = self._existing_task(task_id)
            if data["status"] != "pending_approval":
                raise RuntimeError("Task is not waiting for approval")
            next_run = self._compute_next_run(ScheduledTask.from_dict(data))
            self._store.transition_task(data, status="active", next_run=next_run)

    def cancel_task(self, task_id: str) -> None:
        """Cancel a task, preventing future dispatch."""
        with self._lock:
            data = self._existing_task(task_id)
            self._store.transition_task(data, status="cancelled", next_run=None)

    def _recover_interrupted_runs(self) -> None:
        recovered = self._store.recover_interrupted_runs(_now_iso())
        if recovered:
            logger.warning(
                "%d scheduled run(s) interrupted with unknown outcome; "
                "review them with 'jarvis scheduler logs'",
                recovered,
            )

    def has_claimed_run(self, task_id: str) -> bool:
        """Return whether a recorded attempt may still take effect."""
        with self._lock:
            return self._store.has_running_run(task_id)

    def run_task_now(self, task_id: str) -> tuple[bool, str] | None:
        """Run one occurrence immediately using the same lease as the daemon."""
        database_path = self._store.db_path
        lease = (
            SchedulerInstanceLock(database_path)
            if database_path != ":memory:"
            else None
        )
        if lease is not None and not lease.acquire():
            raise RuntimeError(
                "Another scheduler is already running for this database."
            )
        try:
            with self._lock:
                if lease is not None:
                    self._recover_interrupted_runs()
                data = self._existing_task(task_id)
                if data["status"] != "active":
                    raise RuntimeError(f"Task {task_id} has status {data['status']}.")
            return self._execute_task(ScheduledTask.from_dict(data))
        finally:
            if lease is not None:
                lease.release()

    # -- Background loop -----------------------------------------------------

    def _poll_loop(self) -> None:
        """Poll for due tasks and execute them until stopped."""
        while not self._stop_event.is_set():
            try:
                now = _now_iso()
                with self._lock:
                    due = self._store.get_due_tasks(now)
                for task_dict in due:
                    if self._stop_event.is_set() or self.external_stop_requested:
                        self._stop_event.set()
                        break
                    task = ScheduledTask.from_dict(task_dict)
                    self._execute_task(task)
            except Exception:
                logger.exception("Scheduler poll error")
            self._stop_event.wait(timeout=self._poll_interval)

    def _execute_task(self, task: ScheduledTask) -> tuple[bool, str] | None:
        """Claim a task occurrence before calling tools and commit its outcome."""
        with self._lock:
            claimed = self._store.start_run(task.id, task.next_run, _now_iso())
        if claimed is None:
            return None
        run_id, data = claimed
        task = ScheduledTask.from_dict(data)

        # Publish start event
        if self._bus is not None:
            self._bus.publish(
                SCHEDULER_TASK_START,
                {"task_id": task.id, "prompt": task.prompt},
            )

        success = False
        result_text = ""
        error_text = ""

        try:
            if self._system is not None:
                raw_tools = (
                    task.tools
                    if isinstance(task.tools, list)
                    else task.tools.split(",")
                )
                tools_list = (
                    [t.strip() for t in raw_tools if t.strip()] if task.tools else []
                )
                meta = task.metadata or {}
                ask_kwargs: Dict[str, Any] = {
                    "agent": task.agent,
                    "tools": (
                        tools_list
                        if tools_list or meta.get("_openjarvis_explicit_tool_scope")
                        else None
                    ),
                }
                if meta.get("operator_id"):
                    ask_kwargs["system_prompt"] = meta.get("system_prompt", "")
                    ask_kwargs["operator_id"] = meta["operator_id"]
                response = self._system.ask(
                    task.prompt,
                    **ask_kwargs,
                )
                result_text = _result_text(response)
            else:
                result_text = f"[dry-run] Would execute: {task.prompt}"
            success = True
        except Exception as exc:
            error_text = str(exc)
            logger.error("Task %s failed: %s", task.id, exc)

        finished_at = _now_iso()

        scheduled_at = task.next_run
        task.last_run = finished_at
        after = datetime.fromisoformat(finished_at)
        if scheduled_at and task.schedule_type != "once":
            scheduled = datetime.fromisoformat(scheduled_at)
            if scheduled.tzinfo is None:
                scheduled = scheduled.replace(tzinfo=timezone.utc)
            after = max(after, scheduled.astimezone(timezone.utc))
        next_run = self._compute_next_run(task, after=after)
        with self._lock:
            self._store.finish_run(
                task.id,
                run_id,
                scheduled_at=scheduled_at,
                finished_at=finished_at,
                next_run=next_run,
                success=success,
                result=result_text,
                error=error_text,
            )

        # Publish end event
        if self._bus is not None:
            self._bus.publish(
                SCHEDULER_TASK_END,
                {
                    "task_id": task.id,
                    "success": success,
                    "result": result_text,
                    "error": error_text,
                },
            )

        return success, result_text if success else error_text

    def _compute_next_run(
        self, task: ScheduledTask, *, after: datetime | None = None
    ) -> Optional[str]:
        """Compute the next run time for a task.

        Returns an ISO 8601 string, or ``None`` if the task should not run again.
        """
        now = after if after is not None else datetime.now(timezone.utc)

        if task.schedule_type == "once":
            # If already run, no more runs
            if task.last_run is not None:
                return None
            # Otherwise the schedule_value is the target ISO datetime
            return task.schedule_value

        if task.schedule_type == "interval":
            seconds = float(task.schedule_value)
            next_time = now + timedelta(seconds=seconds)
            return next_time.isoformat()

        if task.schedule_type == "cron":
            timezone_name = str((task.metadata or {}).get("timezone") or "") or None
            return self._compute_next_cron(task.schedule_value, now, timezone_name)

        return None

    @staticmethod
    def _compute_next_cron(
        cron_expr: str,
        now: datetime,
        timezone_name: str | None = None,
    ) -> Optional[str]:
        """Compute the next run time from a cron expression.

        Evaluate in the configured IANA timezone and persist the result in UTC.
        Missing cron support is fatal instead of silently changing semantics.
        """
        try:
            from croniter import croniter  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "croniter is required for cron schedules; reinstall OpenJarvis"
            ) from exc

        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        local_now = now.astimezone(ZoneInfo(timezone_name)) if timezone_name else now
        next_local = croniter(cron_expr, local_now).get_next(datetime)
        if next_local.tzinfo is None:
            next_local = next_local.replace(tzinfo=local_now.tzinfo)
        return next_local.astimezone(timezone.utc).isoformat()


__all__ = [
    "SCHEDULER_TASK_END",
    "SCHEDULER_TASK_START",
    "ScheduledTask",
    "TaskScheduler",
]
