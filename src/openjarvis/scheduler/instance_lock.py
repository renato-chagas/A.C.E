"""Exclusive per-database process lock for scheduled task execution."""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import BinaryIO

_registry_lock = threading.Lock()
_held_paths: set[str] = set()


class SchedulerInstanceLock:
    """Prevent multiple scheduler processes from executing the same tasks."""

    def __init__(self, database_path: str | Path) -> None:
        resolved_path = Path(database_path).expanduser().resolve()
        resolved = os.path.normcase(str(resolved_path))
        digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:24]
        self._path = resolved_path.with_name(f".openjarvis-scheduler-{digest}.lock")
        self._stop_path = resolved_path.with_name(
            f".openjarvis-scheduler-{digest}.stop"
        )
        self._registry_key = str(self._path)
        self._handle: BinaryIO | None = None

    @property
    def stop_request_path(self) -> Path:
        """Return the private file used to request graceful scheduler shutdown."""
        return self._stop_path

    def request_stop(self) -> None:
        """Signal a running scheduler to stop after its current task finishes."""
        from openjarvis.security.file_utils import secure_write_text

        secure_write_text(self._stop_path, "stop\n")

    def stop_requested(self) -> bool:
        return self._stop_path.is_file()

    def clear_stop_request(self) -> None:
        self._stop_path.unlink(missing_ok=True)

    def acquire(self) -> bool:
        """Acquire the database's exclusive process lock without waiting."""
        if self._handle is not None:
            return True
        with _registry_lock:
            if self._registry_key in _held_paths:
                return False
            _held_paths.add(self._registry_key)

        handle: BinaryIO | None = None
        try:
            from openjarvis.security.file_utils import secure_create

            try:
                secure_create(self._path)
            except FileExistsError:
                pass
            handle = self._path.open("r+b")
        except Exception:
            if handle is not None:
                handle.close()
            with _registry_lock:
                _held_paths.discard(self._registry_key)
            raise

        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            with _registry_lock:
                _held_paths.discard(self._registry_key)
            return False
        except Exception:
            handle.close()
            with _registry_lock:
                _held_paths.discard(self._registry_key)
            raise

        self._handle = handle
        return True

    def release(self) -> None:
        """Release the lock; the OS also releases it if the process exits."""
        handle = self._handle
        if handle is None:
            return
        self._handle = None
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
            with _registry_lock:
                _held_paths.discard(self._registry_key)


__all__ = ["SchedulerInstanceLock"]
