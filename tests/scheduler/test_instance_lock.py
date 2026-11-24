"""Tests for cross-process scheduler exclusivity."""

from __future__ import annotations

import multiprocessing

from openjarvis.scheduler.instance_lock import SchedulerInstanceLock


def _hold_lock(database_path: str, acquired, release) -> None:
    lock = SchedulerInstanceLock(database_path)
    if not lock.acquire():
        acquired.put(False)
        return
    acquired.put(True)
    release.wait(timeout=10)
    lock.release()


def test_scheduler_lock_is_exclusive_across_processes(tmp_path) -> None:
    database_path = str(tmp_path / "scheduler.db")
    acquired = multiprocessing.Queue()
    release = multiprocessing.Event()
    process = multiprocessing.Process(
        target=_hold_lock,
        args=(database_path, acquired, release),
    )
    process.start()
    local_lock = SchedulerInstanceLock(database_path)
    try:
        assert acquired.get(timeout=5) is True
        assert local_lock.acquire() is False
    finally:
        release.set()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        acquired.close()
        acquired.join_thread()

    assert process.exitcode == 0
    assert local_lock.acquire() is True
    local_lock.release()


def test_stop_request_is_shared_and_can_be_cleared(tmp_path) -> None:
    lock = SchedulerInstanceLock(tmp_path / "scheduler.db")

    lock.request_stop()
    assert lock.stop_requested()
    assert lock.stop_request_path.is_file()

    lock.clear_stop_request()
    assert not lock.stop_requested()
