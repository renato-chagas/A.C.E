"""Detached lifecycle helpers for ACE's local resource monitor."""

from __future__ import annotations

import copy
import json
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

from openjarvis.core.paths import get_config_dir
from openjarvis.security.file_utils import (
    secure_create,
    secure_mkdir,
    secure_write_json,
)

_WORKER_TOKEN_ENV = "OPENJARVIS_MONITOR_LAUNCH_TOKEN"
_VALID_STATUSES = {"starting", "running", "stopping"}
_MAX_LOG_BYTES = 4 * 1024 * 1024
_LOG_BACKUPS = 3
_MAX_SCREEN_ASSESSMENT_LOG_BYTES = 256 * 1024
_REASONER_RETRY_SECONDS = 60.0


class MonitorStateConflict(RuntimeError):
    """An active or stopping worker belongs to a different operation."""


def _state_path() -> Path:
    return get_config_dir() / "proactive_monitor.json"


def _lock_path() -> Path:
    return get_config_dir() / "proactive_monitor.control.lock"


def _singleton_lock_path() -> Path:
    return get_config_dir() / "proactive_monitor.singleton.lock"


def _rotate_log_if_needed(path: Path) -> None:
    if not path.exists() or path.stat().st_size < _MAX_LOG_BYTES:
        return
    oldest = Path(f"{path}.{_LOG_BACKUPS}")
    oldest.unlink(missing_ok=True)
    for index in range(_LOG_BACKUPS - 1, 0, -1):
        source = Path(f"{path}.{index}")
        if source.exists():
            os.replace(source, Path(f"{path}.{index + 1}"))
    os.replace(path, Path(f"{path}.1"))


def _worker_command(*, screen_monitor_enabled: bool) -> list[str]:
    """Build the detached worker command with explicit screen opt-in."""
    command = [
        sys.executable,
        "-m",
        "openjarvis.cli",
        "monitor",
        "--daemon-worker",
    ]
    if screen_monitor_enabled:
        command.append("--monitor-screen")
    return command


@contextmanager
def _control_lock() -> Iterator[None]:
    """Serialize monitor start, stop, and state updates across processes."""
    path = _lock_path()
    secure_create(path)
    with path.open("r+b") as handle:
        if os.name == "nt":
            import msvcrt

            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def acquire_local_monitor_lock() -> BinaryIO | None:
    """Acquire the machine-local monitor lease without waiting."""
    path = _singleton_lock_path()
    secure_create(path)
    handle = path.open("r+b")
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def release_local_monitor_lock(handle: BinaryIO | None) -> None:
    """Release a lease acquired by :func:`acquire_local_monitor_lock`."""
    if handle is None:
        return
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _read_state_unlocked() -> dict[str, Any] | None:
    try:
        payload = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        type(payload.get("pid")) is not int
        or payload["pid"] <= 0
        or not isinstance(payload.get("process_created"), (float, int))
        or not isinstance(payload.get("launch_token"), str)
        or not payload["launch_token"]
        or payload.get("status") not in _VALID_STATUSES
        or not isinstance(payload.get("started_at"), str)
        or not isinstance(payload.get("log_file"), str)
        or type(payload.get("screen_monitor_enabled", False)) is not bool
        or type(payload.get("desktop_resource_only", False)) is not bool
        or type(payload.get("desktop_notifications_enabled", False)) is not bool
        or (
            payload.get("desktop_resource_only", False)
            and payload.get("screen_monitor_enabled", False)
        )
        or (
            payload.get("desktop_notifications_enabled", False)
            and not payload.get("desktop_resource_only", False)
        )
    ):
        return None
    return payload


def _process_matches(state: dict[str, Any]) -> bool | None:
    """Check both process creation time and the private worker command line."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        process = psutil.Process(int(state["pid"]))
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
        if abs(process.create_time() - float(state["process_created"])) > 0.05:
            return False
        arguments = process.cmdline()
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, psutil.Error, OSError, ValueError):
        return None
    return "--daemon-worker" in arguments and "openjarvis.cli" in arguments


def _live_state_unlocked(*, clean_stale: bool = True) -> dict[str, Any] | None:
    state = _read_state_unlocked()
    if state is None:
        if clean_stale:
            _state_path().unlink(missing_ok=True)
        return None
    match = _process_matches(state)
    if match is True:
        return state
    if match is None:
        raise RuntimeError(
            "Could not verify the local monitor process. Check permissions and "
            "install psutil before managing the background monitor."
        )
    if clean_stale:
        _state_path().unlink(missing_ok=True)
    return None


def start_monitor_daemon(
    interval_seconds: int,
    *,
    screen_monitor_enabled: bool = False,
    desktop_resource_only: bool = False,
    desktop_notifications_enabled: bool = False,
) -> dict[str, Any]:
    """Start one detached worker and return its state record."""
    if desktop_resource_only and screen_monitor_enabled:
        raise ValueError("Desktop resource monitoring does not take screen captures.")
    if desktop_notifications_enabled and not desktop_resource_only:
        raise ValueError("Dashboard notifications require a dashboard worker.")
    base = get_config_dir()
    secure_mkdir(base)
    state_path = _state_path()
    log_path = base / "proactive_monitor.log"
    command = _worker_command(screen_monitor_enabled=screen_monitor_enabled)
    if desktop_resource_only:
        command[1:1] = ["-I", "-B"]
    token = secrets.token_urlsafe(32)
    env = os.environ.copy()
    env[_WORKER_TOKEN_ENV] = token
    spawn_options: dict[str, Any] = {}
    if sys.platform == "win32":
        spawn_options["creationflags"] = (
            subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        spawn_options["start_new_session"] = True

    with _control_lock():
        existing = _live_state_unlocked()
        if existing is not None:
            if desktop_resource_only:
                if (
                    existing.get("desktop_resource_only", False)
                    and existing["status"] != "stopping"
                ):
                    if existing.get("desktop_notifications_enabled", False) != (
                        desktop_notifications_enabled
                    ):
                        raise MonitorStateConflict(
                            "Stop the dashboard worker before changing notifications."
                        )
                    return existing
                raise MonitorStateConflict(
                    "Another monitor owns the background worker."
                )
            raise RuntimeError(
                f"ACE background monitor is already {existing['status']} "
                f"(PID {existing['pid']})."
            )
        secure_create(log_path)
        _rotate_log_if_needed(log_path)
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                close_fds=True,
                env=env,
                **spawn_options,
            )
        try:
            import psutil

            created = psutil.Process(process.pid).create_time()
            state = {
                "pid": process.pid,
                "process_created": created,
                "launch_token": token,
                "status": "starting",
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "interval_seconds": int(interval_seconds),
                "screen_monitor_enabled": bool(screen_monitor_enabled),
                "desktop_resource_only": desktop_resource_only,
                "desktop_notifications_enabled": desktop_notifications_enabled,
                "log_file": str(log_path),
            }
            secure_write_json(state_path, state)
        except Exception:
            try:
                process.terminate()
            except OSError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
            raise

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        active = get_monitor_daemon_status()
        if active is None:
            try:
                detail = log_path.read_text(encoding="utf-8", errors="replace")[-1200:]
            except OSError:
                detail = ""
            message = "Background worker exited before monitoring started."
            if detail.strip():
                message += f" See {log_path} for details.\n{detail.strip()}"
            raise RuntimeError(message)
        if active["status"] == "running":
            return active
        time.sleep(0.1)
    return state


def get_monitor_daemon_status(*, read_only: bool = False) -> dict[str, Any] | None:
    """Verify monitor state. Write access is needed only to clean up stale records."""
    if read_only:
        # State records are replaced atomically. An inspection needs no lock file.
        return _live_state_unlocked(clean_stale=False)
    with _control_lock():
        return _live_state_unlocked()


def _stop_worker(state: dict[str, Any], timeout: float = 8.0) -> None:
    import psutil

    try:
        process = psutil.Process(int(state["pid"]))
    except psutil.NoSuchProcess:
        return
    match = _process_matches(state)
    if match is False:
        return
    if match is None:
        raise RuntimeError(
            "Could not verify the ACE monitor process before stopping it."
        )
    try:
        process.wait(timeout=timeout)
        return
    except psutil.TimeoutExpired:
        pass
    match = _process_matches(state)
    if match is False:
        return
    if match is None:
        raise RuntimeError(
            "Could not re-verify the ACE monitor process before stopping it."
        )
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except psutil.TimeoutExpired:
        match = _process_matches(state)
        if match is False:
            return
        if match is None:
            raise RuntimeError(
                "Could not verify the ACE monitor process before force stop."
            )
        process.kill()
        try:
            process.wait(timeout=3.0)
        except psutil.TimeoutExpired as exc:
            raise RuntimeError("ACE monitor did not stop after termination.") from exc


def stop_monitor_daemon(*, only_desktop: bool = False) -> dict[str, Any] | None:
    """Request a graceful stop, escalating only for the verified worker."""
    with _control_lock():
        state = _live_state_unlocked()
        if state is None:
            return None
        if only_desktop and (
            not state.get("desktop_resource_only", False)
            or state["status"] == "stopping"
        ):
            raise MonitorStateConflict("Another operation owns the background worker.")
        state["status"] = "stopping"
        secure_write_json(_state_path(), state)

    _stop_worker(state)
    with _control_lock():
        current = _read_state_unlocked()
        if current and current.get("launch_token") == state["launch_token"]:
            _state_path().unlink(missing_ok=True)
    return state


def _read_owned_worker_state_unlocked(token: str) -> dict[str, Any] | None:
    """Read this worker's state using its PID and private launch token."""
    state = _read_state_unlocked()
    if state is None:
        print(
            "ACE monitor worker cannot read a valid launch-state record.",
            flush=True,
        )
        return None
    if state.get("pid") not in {os.getpid(), os.getppid()}:
        print(
            "ACE monitor worker launch-state PID is neither this process nor "
            "its launcher.",
            flush=True,
        )
        return None
    if state.get("launch_token") != token:
        print("ACE monitor worker launch token does not match its state.", flush=True)
        return None
    return state


def _worker_state(token: str) -> dict[str, Any] | None:
    with _control_lock():
        return _read_owned_worker_state_unlocked(token)


def _set_worker_running(token: str) -> dict[str, Any] | None:
    with _control_lock():
        state = _read_owned_worker_state_unlocked(token)
        if state is None or state["status"] == "stopping":
            return None
        import psutil

        state["pid"] = os.getpid()
        state["process_created"] = psutil.Process(os.getpid()).create_time()
        state["status"] = "running"
        secure_write_json(_state_path(), state)
        return state


def _clear_worker_state(token: str) -> None:
    with _control_lock():
        state = _read_state_unlocked()
        if (
            state is not None
            and state.get("pid") == os.getpid()
            and state.get("launch_token") == token
        ):
            _state_path().unlink(missing_ok=True)


def _build_local_reasoner(
    config: Any, monitor_store: Any = None
) -> tuple[Any | None, list[Any]]:
    """Build a reasoner with the shared local-only engine failover path."""
    if not getattr(config.proactive, "local_monitor_reasoning_enabled", False):
        return None, []
    if (getattr(config.intelligence, "provider", "") or "").strip().lower() != "local":
        print(
            "Proactive model analysis skipped: provider is not configured local.",
            flush=True,
        )
        return None, []

    resources: list[Any] = []
    try:
        import openjarvis.engine  # noqa: F401 — register local engine backends
        from openjarvis.cli.ask import _get_memory_backend
        from openjarvis.engine._discovery import get_engine
        from openjarvis.engine._locality import is_local_engine
        from openjarvis.memory import load_configured_facts
        from openjarvis.proactive.reasoning import LocalIncidentReasoner
        from openjarvis.security import setup_security

        preferred_engine = (
            getattr(config.intelligence, "preferred_engine", "")
            or config.engine.default
        )
        if not preferred_engine:
            print(
                "Proactive model analysis skipped: no local engine is configured.",
                flush=True,
            )
            return None, []

        local_config = copy.deepcopy(config)
        local_config.engine.default = preferred_engine
        resolved = get_engine(local_config)
        if resolved is None:
            print(
                "Proactive model analysis skipped: no local engine is available.",
                flush=True,
            )
            return None, []

        engine_key, engine = resolved
        resources.append(engine)
        if not is_local_engine(engine, engine_key) or not engine.health():
            print(
                "Proactive model analysis skipped: resolved engine is not healthy "
                "and local.",
                flush=True,
            )
            _close_resources(resources)
            return None, []

        try:
            models = engine.list_models()
        except Exception:
            models = []
        model = ""
        for candidate in (
            config.intelligence.default_model,
            config.intelligence.fallback_model,
        ):
            if candidate and (not models or candidate in models):
                try:
                    if engine.can_serve(candidate):
                        model = candidate
                        break
                except Exception:
                    continue
        if not model:
            print(
                "Proactive model analysis skipped: choose a local model with "
                "`jarvis model setup` first.",
                flush=True,
            )
            _close_resources(resources)
            return None, []

        security = setup_security(config, engine)
        safe_engine = security.engine
        if safe_engine is not engine:
            resources.remove(engine)
            resources.append(safe_engine)
        if security.audit_logger is not None:
            resources.append(security.audit_logger)
        memory_backend = None
        if config.memory.default_backend == "sqlite":
            memory_backend = _get_memory_backend(config)
        elif config.memory.default_backend:
            print(
                "Skipping proactive document recall because the configured "
                "memory backend is not verified local SQLite.",
                flush=True,
            )
        if memory_backend is not None:
            resources.append(memory_backend)
        reasoner = LocalIncidentReasoner(
            safe_engine,
            model,
            config,
            memory_backend,
            monitor_store=monitor_store,
            facts_provider=lambda: load_configured_facts(config),
        )
        return reasoner, resources
    except Exception as exc:
        _close_resources(resources)
        print(f"Proactive model analysis unavailable: {exc}", flush=True)
        return None, []


def _build_local_screen_monitor(
    config: Any,
    *,
    on_assessment: Callable[[str], None],
    on_error: Callable[[str], None],
) -> tuple[Any | None, list[Any]]:
    """Build screen analysis only through a verified local model route."""
    if (getattr(config.intelligence, "provider", "") or "").strip().lower() != "local":
        print(
            "Background screen monitoring requires provider = 'local'.",
            flush=True,
        )
        return None, []

    resources: list[Any] = []
    try:
        import openjarvis.engine  # noqa: F401 — register local engine backends
        from openjarvis.cli._local_setup_hint import local_model_setup_hint
        from openjarvis.engine._discovery import get_engine
        from openjarvis.engine._locality import (
            is_local_engine,
            is_local_engine_for_model,
        )
        from openjarvis.proactive.screen_monitor import LocalScreenMonitor
        from openjarvis.security import setup_security

        preferred_engine = (
            getattr(config.intelligence, "preferred_engine", "")
            or config.engine.default
        )
        if not preferred_engine:
            print(
                "Background screen monitoring needs a configured local engine.",
                flush=True,
            )
            return None, []

        local_config = copy.deepcopy(config)
        local_config.engine.default = preferred_engine
        resolved = get_engine(local_config)
        if resolved is None:
            print(
                "Background screen monitoring found no available local engine. "
                + local_model_setup_hint(config),
                flush=True,
            )
            return None, []

        engine_key, engine = resolved
        resources.append(engine)
        if not is_local_engine(engine, engine_key) or not engine.health():
            print(
                "Background screen monitoring refused an unverified local engine.",
                flush=True,
            )
            _close_resources(resources)
            return None, []

        try:
            models = list(engine.list_models())
        except Exception:
            models = []
        model = ""
        candidates = (
            getattr(config.intelligence, "default_model", ""),
            getattr(config.intelligence, "fallback_model", ""),
        )
        for candidate in dict.fromkeys(value for value in candidates if value):
            if models and candidate not in models:
                continue
            try:
                if engine.can_serve(candidate):
                    model = candidate
                    break
            except Exception:
                continue
        if not model or not is_local_engine_for_model(engine, model):
            print(
                "Background screen monitoring needs a selected, verified local "
                "vision model. Run `jarvis model setup` first.",
                flush=True,
            )
            _close_resources(resources)
            return None, []

        security = setup_security(config, engine)
        safe_engine = security.engine
        if safe_engine is not engine:
            resources.remove(engine)
            resources.append(safe_engine)
        if security.audit_logger is not None:
            resources.append(security.audit_logger)

        monitor = LocalScreenMonitor(
            safe_engine,
            model,
            interval_seconds=config.proactive.local_screen_monitor_interval_seconds,
            min_analysis_interval_seconds=(
                config.proactive.local_screen_monitor_min_analysis_interval_seconds
            ),
            on_assessment=on_assessment,
            on_error=on_error,
        )
        return monitor, resources
    except Exception as exc:
        _close_resources(resources)
        print(f"Background screen monitoring unavailable: {exc}", flush=True)
        return None, []


def _close_resources(resources: list[Any]) -> None:
    closed: set[int] = set()
    for resource in reversed(resources):
        if id(resource) in closed:
            continue
        closed.add(id(resource))
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


class _ScreenAssessmentLog:
    """Write bounded, terminal-safe text observations to the worker log."""

    def __init__(self, max_bytes: int = _MAX_SCREEN_ASSESSMENT_LOG_BYTES) -> None:
        self._max_bytes = max(0, int(max_bytes))
        self._written_bytes = 0
        self._capped = False

    def __call__(self, assessment: str) -> None:
        clean = "".join(
            char for char in assessment if char in "\n\t" or char.isprintable()
        )
        if not clean or self._capped:
            return
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        entry = (
            f"{timestamp} A.C.E. local screen observation "
            f"(suggestions only; no changes made):\n{clean}\n"
        )
        entry_bytes = len(entry.encode("utf-8"))
        if self._written_bytes + entry_bytes > self._max_bytes:
            self._capped = True
            print(
                "ACE screen assessment logging reached its 256 KiB "
                "per-run limit; later screen assessments will not be saved.",
                flush=True,
            )
            return
        print(entry, flush=True)
        self._written_bytes += entry_bytes


def _notify_monitor_notice(
    message: str, enabled: bool, *, desktop_resource_only: bool = False
) -> bool:
    """Deliver a local notice when enabled; otherwise leave it to the log."""
    if not enabled:
        return True
    from openjarvis.proactive.notifications import notify_monitor_event

    if desktop_resource_only:
        return notify_monitor_event(message, history_hint="System Monitoring")
    return notify_monitor_event(message)


class _LocalReasonerSupervisor:
    """Retry local incident-reasoner setup after runtime outages."""

    def __init__(
        self,
        config: Any,
        monitor_store: Any,
        *,
        retry_interval: float = _REASONER_RETRY_SECONDS,
    ) -> None:
        self._config = config
        self._monitor_store = monitor_store
        self._retry_interval = max(1.0, float(retry_interval))
        provider = (
            str((getattr(config.intelligence, "provider", "") or "")).strip().lower()
        )
        self._enabled = bool(
            getattr(config.proactive, "local_monitor_reasoning_enabled", False)
            and provider == "local"
        )
        self._reasoner: Any | None = None
        self._resources: list[Any] = []
        self._next_retry = 0.0
        self._lock = threading.Lock()

    @property
    def reasoner(self) -> Any | None:
        return self._reasoner

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def resources(self) -> list[Any]:
        return list(self._resources)

    def initialize(self) -> Any | None:
        """Attempt once during worker startup, even when retry is disabled."""
        with self._lock:
            return self._build_locked()

    def retry_if_due(self) -> Any | None:
        """Retry only when enabled, unavailable, and outside the cooldown."""
        with self._lock:
            if (
                not self._enabled
                or self._reasoner is not None
                or time.monotonic() < self._next_retry
            ):
                return None
            return self._build_locked()

    def _build_locked(self) -> Any | None:
        self._next_retry = time.monotonic() + self._retry_interval
        reasoner, resources = _build_local_reasoner(
            self._config,
            monitor_store=self._monitor_store,
        )
        if reasoner is None:
            _close_resources(resources)
            return None
        self._reasoner = reasoner
        self._resources.extend(resources)
        return self._reasoner


class _PendingIncidentGate:
    """Track queued incidents and rate-limit retries after failed analysis."""

    def __init__(self, retry_interval: float = _REASONER_RETRY_SECONDS) -> None:
        self._retry_interval = max(1.0, float(retry_interval))
        self._queued: set[int] = set()
        self._retry_after: dict[int, float] = {}
        self._lock = threading.Lock()

    def try_queue(
        self,
        incident_id: int,
        *,
        force: bool = False,
        now: float | None = None,
    ) -> bool:
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            if incident_id in self._queued:
                return False
            if not force and self._retry_after.get(incident_id, 0.0) > now:
                return False
            self._queued.add(incident_id)
            return True

    def release(
        self,
        incident_id: int,
        *,
        retry: bool = False,
        now: float | None = None,
    ) -> None:
        with self._lock:
            self._queued.discard(incident_id)
            if retry:
                current = time.monotonic() if now is None else float(now)
                self._retry_after[incident_id] = current + self._retry_interval
            else:
                self._retry_after.pop(incident_id, None)

    def retain_active(self, incident_ids: set[int]) -> None:
        """Forget retry state for resolved or already-consumed incidents."""
        with self._lock:
            for incident_id in set(self._retry_after) - incident_ids:
                self._retry_after.pop(incident_id, None)


def run_monitor_worker(token: str, screen_monitor_enabled: bool = False) -> int:
    """Run the detached polling loop until `--stop` requests shutdown."""
    if not token:
        print("ACE monitor worker rejected: launch token is missing.", flush=True)
        return 2
    state = _worker_state(token)
    if state is None:
        print(
            "ACE monitor worker rejected: no matching active launch state.",
            flush=True,
        )
        return 2
    if state["status"] == "stopping":
        print("ACE monitor worker cancelled before initialization.", flush=True)
        return 2
    if bool(state.get("screen_monitor_enabled", False)) != bool(screen_monitor_enabled):
        print(
            "ACE monitor worker rejected: screen mode differs from launch state.",
            flush=True,
        )
        return 2

    service = None
    screen_monitor = None
    screen_resources: list[Any] = []
    reasoner_supervisor: _LocalReasonerSupervisor | None = None
    reasoning_thread = None
    reasoning_stop = threading.Event()
    try:
        from openjarvis.core.config import load_config
        from openjarvis.proactive.local_monitor import (
            LocalMonitorStore,
            LocalProactiveMonitor,
        )
        from openjarvis.proactive.reasoning import measured_incident_fallback

        config = load_config()
        desktop_resource_only = state.get("desktop_resource_only", False)
        if desktop_resource_only:
            # The verified dashboard launch authorizes aggregate readings only.
            config.proactive.local_monitor_process_context_enabled = False
            config.proactive.local_monitor_reasoning_enabled = False
            config.proactive.local_monitor_notifications_enabled = state.get(
                "desktop_notifications_enabled", False
            )
        elif not config.proactive.local_monitor_enabled:
            print("ACE monitor is disabled in config.toml; worker exiting.", flush=True)
            return 2

        monitor_store = LocalMonitorStore()
        reasoner_supervisor = _LocalReasonerSupervisor(config, monitor_store)
        pending: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=20)
        pending_gate = _PendingIncidentGate()

        if screen_monitor_enabled:

            def report_screen_error(message: str) -> None:
                timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
                print(f"{timestamp} A.C.E. screen monitor: {message}", flush=True)

            screen_monitor, screen_resources = _build_local_screen_monitor(
                config,
                on_assessment=_ScreenAssessmentLog(),
                on_error=report_screen_error,
            )
            if screen_monitor is None:
                return 2

        def queue_pending_incidents(*, force: bool = False) -> None:
            if reasoner_supervisor is None or not reasoner_supervisor.enabled:
                return
            try:
                incidents = monitor_store.pending_context(limit=10)
            except Exception as exc:
                print(f"Could not read pending incidents: {exc}", flush=True)
                return
            pending_gate.retain_active({int(incident["id"]) for incident in incidents})
            for incident in incidents:
                incident_id = int(incident["id"])
                if not pending_gate.try_queue(incident_id, force=force):
                    continue
                try:
                    pending.put_nowait(incident)
                except queue.Full:
                    pending_gate.release(incident_id)
                    print(
                        "Proactive analysis queue is full; incident retained.",
                        flush=True,
                    )

        def write_notice(message: str) -> None:
            timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
            print(f"{timestamp} ACE monitor: {message}", flush=True)
            if not _notify_monitor_notice(
                message,
                config.proactive.local_monitor_notifications_enabled,
                desktop_resource_only=desktop_resource_only,
            ):
                print(
                    f"{timestamp} ACE monitor: no local desktop notification "
                    "backend; notice remains in the log.",
                    flush=True,
                )
            queue_pending_incidents()

        service = LocalProactiveMonitor(
            config.proactive,
            on_notice=write_notice,
            store=monitor_store,
            record_measured_assessments=desktop_resource_only,
        )
        if not service.start():
            print(
                "Another local ACE monitor is already active; worker exiting.",
                flush=True,
            )
            return 2
        if screen_monitor is not None and not screen_monitor.start():
            print("ACE screen monitor could not start.", flush=True)
            return 2
        state = _set_worker_running(token)
        if state is None:
            print(
                "ACE monitor worker stopped before its running state was committed.",
                flush=True,
            )
            return 2
        reasoner_supervisor.initialize()

        def reason_pending_incidents() -> None:
            while not reasoning_stop.is_set():
                try:
                    incident = pending.get(timeout=0.5)
                except queue.Empty:
                    if (
                        reasoner_supervisor is not None
                        and reasoner_supervisor.reasoner is None
                    ):
                        recovered = reasoner_supervisor.retry_if_due()
                        if recovered is not None:
                            print(
                                "ACE local incident analysis is available again.",
                                flush=True,
                            )
                            queue_pending_incidents(force=True)
                    continue
                incident_id = int(incident["id"])
                retry = True
                try:
                    reasoner = (
                        reasoner_supervisor.reasoner
                        if reasoner_supervisor is not None
                        else None
                    )
                    if reasoner is not None:
                        assessment = reasoner.analyze(incident)
                    else:
                        assessment = measured_incident_fallback(incident)
                        if assessment and monitor_store.save_assessment(
                            incident_id,
                            assessment,
                            fallback_once=True,
                        ):
                            timestamp = (
                                datetime.now()
                                .astimezone()
                                .isoformat(timespec="seconds")
                            )
                            print(
                                f"\n{timestamp} A.C.E. proactive rule-based "
                                f"fallback (incident {incident_id}; measured "
                                "facts and read-only checks; no changes made):\n"
                                f"{assessment}\n",
                                flush=True,
                            )
                    if assessment:
                        if reasoner is not None:
                            timestamp = (
                                datetime.now()
                                .astimezone()
                                .isoformat(timespec="seconds")
                            )
                            print(
                                f"\n{timestamp} A.C.E. proactive assessment "
                                f"(incident {incident_id}; suggestions only, "
                                f"no changes made):\n{assessment}\n",
                                flush=True,
                            )
                            retry = False
                except Exception as exc:
                    print(
                        f"Proactive analysis failed for incident {incident_id}: {exc}",
                        flush=True,
                    )
                finally:
                    pending.task_done()
                    pending_gate.release(incident_id, retry=retry)

        if reasoner_supervisor.enabled:
            reasoning_thread = threading.Thread(
                target=reason_pending_incidents,
                name="ace-monitor-reasoner",
                daemon=True,
            )
            reasoning_thread.start()
            queue_pending_incidents()
            if reasoner_supervisor.reasoner is None:
                print(
                    "Monitoring continues with measured alerts and read-only "
                    "rule-based fallbacks; ACE will retry local model analysis "
                    "periodically.",
                    flush=True,
                )
        elif config.proactive.local_monitor_reasoning_enabled:
            print(
                "Monitoring continues with measured alerts only; model analysis "
                "is unavailable or not local.",
                flush=True,
            )
        print(
            f"ACE monitor started (PID {os.getpid()}, "
            f"interval {state.get('interval_seconds')}s).",
            flush=True,
        )
        if screen_monitor is not None:
            print(
                "ACE background screen monitoring is active. Captures remain in "
                "memory and only local text assessments are written to this log.",
                flush=True,
            )
        while True:
            current = _worker_state(token)
            if current is None or current["status"] == "stopping":
                break
            if not service.is_running:
                print("ACE monitor sampling thread exited unexpectedly.", flush=True)
                return 1
            if screen_monitor is not None and not screen_monitor.is_running:
                print("ACE screen monitoring thread exited unexpectedly.", flush=True)
                return 1
            queue_pending_incidents()
            time.sleep(0.5)
        return 0
    except Exception as exc:
        print(f"ACE monitor worker failed: {exc}", flush=True)
        return 1
    finally:
        if screen_monitor is not None:
            screen_monitor.stop(timeout=5.0)
        if service is not None:
            service.stop()
        reasoning_stop.set()
        if reasoning_thread is not None:
            reasoning_thread.join(timeout=3.0)
        if (reasoning_thread is None or not reasoning_thread.is_alive()) and (
            reasoner_supervisor is not None
        ):
            _close_resources(reasoner_supervisor.resources)
        if screen_monitor is None or not screen_monitor.is_running:
            _close_resources(screen_resources)
        else:
            print(
                "ACE screen analysis did not stop before shutdown; its engine will "
                "be released when the worker exits.",
                flush=True,
            )
        _clear_worker_state(token)


__all__ = [
    "acquire_local_monitor_lock",
    "get_monitor_daemon_status",
    "release_local_monitor_lock",
    "run_monitor_worker",
    "start_monitor_daemon",
    "stop_monitor_daemon",
]
