"""Tests for TaskScheduler — scheduling logic, lifecycle, and execution."""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from openjarvis.scheduler.instance_lock import SchedulerInstanceLock
from openjarvis.scheduler.scheduler import ScheduledTask, TaskScheduler
from openjarvis.scheduler.store import SchedulerStore


@pytest.fixture()
def store(tmp_path):
    s = SchedulerStore(tmp_path / "scheduler_test.db")
    yield s
    s.close()


@pytest.fixture()
def scheduler(store):
    sched = TaskScheduler(store, poll_interval=1)
    yield sched
    sched.stop()


# -- ScheduledTask dataclass -------------------------------------------------


class TestScheduledTask:
    def test_round_trip(self):
        task = ScheduledTask(
            id="abc123",
            prompt="hello",
            schedule_type="interval",
            schedule_value="60",
            agent="orchestrator",
            tools="calculator,think",
            metadata={"key": "value"},
        )
        d = task.to_dict()
        restored = ScheduledTask.from_dict(d)
        assert restored.id == "abc123"
        assert restored.prompt == "hello"
        assert restored.schedule_type == "interval"
        assert restored.agent == "orchestrator"
        assert restored.tools == "calculator,think"
        assert restored.metadata == {"key": "value"}

    def test_defaults(self):
        task = ScheduledTask(
            id="x",
            prompt="p",
            schedule_type="once",
            schedule_value="2026-01-01T00:00:00",
        )
        assert task.context_mode == "isolated"
        assert task.status == "active"
        assert task.agent == "simple"
        assert task.tools == ""
        assert task.metadata == {}


# -- TaskScheduler create/list -----------------------------------------------


class TestCreateAndList:
    def test_create_task(self, scheduler):
        task = scheduler.create_task(
            prompt="hello world",
            schedule_type="interval",
            schedule_value="3600",
        )
        assert task.id
        assert task.prompt == "hello world"
        assert task.schedule_type == "interval"
        assert task.next_run is not None
        assert task.status == "active"

    def test_create_task_with_agent_and_tools(self, scheduler):
        task = scheduler.create_task(
            prompt="hello",
            schedule_type="once",
            schedule_value="2099-01-01T00:00:00+00:00",
            agent="orchestrator",
            tools="calculator,think",
        )
        assert task.agent == "orchestrator"
        assert task.tools == "calculator,think"

    def test_agent_created_task_stays_pending_until_user_approval(self, scheduler):
        task = scheduler.create_task(
            prompt="Review my inbox",
            schedule_type="interval",
            schedule_value="3600",
            requires_approval=True,
            agent="orchestrator",
            tools="email_archive",
        )

        assert task.status == "pending_approval"
        assert task.metadata["_openjarvis_explicit_tool_scope"] is True
        assert scheduler.list_tasks(status="pending_approval")[0].id == task.id
        assert scheduler._store.get_due_tasks("9999-12-31T23:59:59+00:00") == []
        with pytest.raises(RuntimeError, match="explicit approval"):
            scheduler.pause_task(task.id)
        with pytest.raises(RuntimeError, match="explicit approval"):
            scheduler.resume_task(task.id)

        scheduler.approve_task(task.id)
        approved = scheduler.list_tasks(status="active")[0]
        assert approved.id == task.id
        assert approved.tools == "email_archive"
        assert scheduler._store.get_due_tasks("9999-12-31T23:59:59+00:00")

    def test_agent_created_task_with_no_tools_has_an_explicit_empty_scope(
        self, scheduler
    ):
        task = scheduler.create_task(
            prompt="Summarize the project status",
            schedule_type="once",
            schedule_value="2099-01-01T00:00:00+00:00",
            requires_approval=True,
            agent="orchestrator",
        )
        system = MagicMock()
        system.ask.return_value = {"content": "summary"}
        scheduler.set_system(system)
        scheduler.approve_task(task.id)

        scheduler._execute_task(scheduler.list_tasks(status="active")[0])

        assert system.ask.call_args.kwargs["tools"] == []

    def test_agent_created_task_uses_only_the_approved_tool_list(self, scheduler):
        task = scheduler.create_task(
            prompt="Archive reviewed newsletters",
            schedule_type="once",
            schedule_value="2099-01-01T00:00:00+00:00",
            requires_approval=True,
            agent="orchestrator",
            tools="email_archive,think",
        )
        system = MagicMock()
        system.ask.return_value = {"content": "done"}
        scheduler.set_system(system)
        scheduler.approve_task(task.id)

        scheduler._execute_task(scheduler.list_tasks(status="active")[0])

        assert system.ask.call_args.kwargs["tools"] == ["email_archive", "think"]

    def test_list_tasks_empty(self, scheduler):
        assert scheduler.list_tasks() == []

    def test_list_tasks(self, scheduler):
        scheduler.create_task("a", "interval", "60")
        scheduler.create_task("b", "interval", "120")
        assert len(scheduler.list_tasks()) == 2

    def test_list_tasks_filter_status(self, scheduler):
        t1 = scheduler.create_task("a", "interval", "60")
        scheduler.create_task("b", "interval", "120")
        scheduler.pause_task(t1.id)
        active = scheduler.list_tasks(status="active")
        paused = scheduler.list_tasks(status="paused")
        assert len(active) == 1
        assert len(paused) == 1


# -- Pause / resume / cancel -------------------------------------------------


class TestPauseResumeCancel:
    def test_pause_task(self, scheduler):
        task = scheduler.create_task("test", "interval", "60")
        scheduler.pause_task(task.id)
        tasks = scheduler.list_tasks(status="paused")
        assert len(tasks) == 1
        assert tasks[0].status == "paused"

    def test_resume_task(self, scheduler):
        task = scheduler.create_task("test", "interval", "60")
        scheduler.pause_task(task.id)
        scheduler.resume_task(task.id)
        tasks = scheduler.list_tasks(status="active")
        assert len(tasks) == 1

    def test_cancel_task(self, scheduler):
        task = scheduler.create_task("test", "interval", "60")
        scheduler.cancel_task(task.id)
        tasks = scheduler.list_tasks(status="cancelled")
        assert len(tasks) == 1
        assert tasks[0].next_run is None

    def test_pause_nonexistent(self, scheduler):
        with pytest.raises(KeyError):
            scheduler.pause_task("nonexistent")

    def test_resume_nonexistent(self, scheduler):
        with pytest.raises(KeyError):
            scheduler.resume_task("nonexistent")

    def test_cancel_nonexistent(self, scheduler):
        with pytest.raises(KeyError):
            scheduler.cancel_task("nonexistent")


# -- _compute_next_run -------------------------------------------------------


class TestComputeNextRun:
    def test_interval(self, scheduler):
        task = ScheduledTask(
            id="t", prompt="p", schedule_type="interval", schedule_value="300"
        )
        next_run = scheduler._compute_next_run(task)
        assert next_run is not None
        # Should be roughly 300 seconds from now
        parsed = datetime.fromisoformat(next_run)
        diff = (parsed - datetime.now(timezone.utc)).total_seconds()
        assert 295 <= diff <= 310

    def test_once_not_yet_run(self, scheduler):
        target = "2099-06-15T12:00:00+00:00"
        task = ScheduledTask(
            id="t",
            prompt="p",
            schedule_type="once",
            schedule_value=target,
            last_run=None,
        )
        next_run = scheduler._compute_next_run(task)
        assert next_run == target

    def test_once_already_run(self, scheduler):
        task = ScheduledTask(
            id="t",
            prompt="p",
            schedule_type="once",
            schedule_value="2099-06-15T12:00:00+00:00",
            last_run="2099-06-15T12:01:00+00:00",
        )
        next_run = scheduler._compute_next_run(task)
        assert next_run is None

    def test_cron_fallback(self, scheduler):
        task = ScheduledTask(
            id="t",
            prompt="p",
            schedule_type="cron",
            schedule_value="30 2 * * *",
        )
        next_run = scheduler._compute_next_run(task)
        assert next_run is not None

    def test_cron_honors_metadata_timezone(self, scheduler, monkeypatch):
        fixed_now = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)

        class FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

        task = ScheduledTask(
            id="t",
            prompt="p",
            schedule_type="cron",
            schedule_value="0 5 * * *",
            metadata={"timezone": "America/Los_Angeles"},
        )
        monkeypatch.setattr("openjarvis.scheduler.scheduler.datetime", FixedDatetime)

        next_run = scheduler._compute_next_run(task)

        assert next_run == "2026-01-15T13:00:00+00:00"

    def test_cron_without_dependency_fails_loudly(self, scheduler, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def without_croniter(name, *args, **kwargs):
            if name == "croniter":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_croniter)
        with pytest.raises(RuntimeError, match="croniter is required"):
            scheduler._compute_next_cron(
                "0 5 * * *",
                datetime(2026, 1, 15, tzinfo=timezone.utc),
            )

    def test_unknown_type(self, scheduler):
        task = ScheduledTask(
            id="t", prompt="p", schedule_type="unknown", schedule_value="x"
        )
        assert scheduler._compute_next_run(task) is None


# -- _execute_task -----------------------------------------------------------


class TestExecuteTask:
    def test_execute_with_system(self, store):
        mock_system = MagicMock()
        mock_system.ask.return_value = "result text"
        sched = TaskScheduler(store, system=mock_system, poll_interval=1)

        task = sched.create_task("what is 2+2?", "once", "2026-01-01T00:00:00+00:00")
        sched._execute_task(task)

        mock_system.ask.assert_called_once()
        assert mock_system.ask.call_args.kwargs["tools"] is None
        call_args = mock_system.ask.call_args
        assert call_args[0][0] == "what is 2+2?"

        # Check that a run log was recorded
        logs = store.get_run_logs(task.id)
        assert len(logs) == 1
        assert logs[0]["success"] == 1
        assert logs[0]["result"] == "result text"

    def test_bind_system_after_construction_and_store_response_content(self, store):
        mock_system = MagicMock()
        mock_system.ask.return_value = {
            "content": "scheduled proactive report",
            "usage": {"output_tokens": 4},
        }
        sched = TaskScheduler(store, poll_interval=1)
        sched.set_system(mock_system)

        task = sched.create_task(
            "review pending work", "once", "2099-01-01T00:00:00+00:00"
        )
        sched._execute_task(task)

        mock_system.ask.assert_called_once()
        logs = store.get_run_logs(task.id)
        assert logs[0]["success"] == 1
        assert logs[0]["result"] == "scheduled proactive report"

    def test_execute_without_system(self, store):
        sched = TaskScheduler(store, poll_interval=1)
        task = sched.create_task("dry run", "once", "2026-01-01T00:00:00+00:00")
        sched._execute_task(task)

        logs = store.get_run_logs(task.id)
        assert len(logs) == 1
        assert logs[0]["success"] == 1
        assert "dry-run" in logs[0]["result"]

    def test_execute_with_error(self, store):
        mock_system = MagicMock()
        mock_system.ask.side_effect = RuntimeError("engine down")
        sched = TaskScheduler(store, system=mock_system, poll_interval=1)

        task = sched.create_task("fail", "once", "2026-01-01T00:00:00+00:00")
        sched._execute_task(task)

        logs = store.get_run_logs(task.id)
        assert len(logs) == 1
        assert logs[0]["success"] == 0
        assert "engine down" in logs[0]["error"]

    def test_execute_publishes_events(self, store):
        mock_bus = MagicMock()
        sched = TaskScheduler(store, bus=mock_bus, poll_interval=1)
        task = sched.create_task("test", "once", "2026-01-01T00:00:00+00:00")
        sched._execute_task(task)

        assert mock_bus.publish.call_count == 2
        start_call = mock_bus.publish.call_args_list[0]
        end_call = mock_bus.publish.call_args_list[1]
        assert start_call[0][0] == "scheduler_task_start"
        assert end_call[0][0] == "scheduler_task_end"

    def test_execute_once_task_completed_after_run(self, store):
        sched = TaskScheduler(store, poll_interval=1)
        task = sched.create_task("one-shot", "once", "2026-01-01T00:00:00+00:00")
        sched._execute_task(task)

        updated = store.get_task(task.id)
        assert updated["status"] == "completed"
        assert updated["next_run"] is None

    def test_execute_with_tools(self, store):
        mock_system = MagicMock()
        mock_system.ask.return_value = "4"
        sched = TaskScheduler(store, system=mock_system, poll_interval=1)

        task = sched.create_task(
            "what is 2+2?",
            "once",
            "2026-01-01T00:00:00+00:00",
            tools="calculator,think",
        )
        sched._execute_task(task)

        call_kwargs = mock_system.ask.call_args[1]
        assert call_kwargs["tools"] == ["calculator", "think"]


class TestDurableRunClaims:
    def test_restart_pauses_unknown_outcomes_then_resumes_only_future_runs(
        self, tmp_path
    ):
        database = tmp_path / "interrupted.db"
        system = MagicMock()
        system.ask.side_effect = KeyboardInterrupt()
        store = SchedulerStore(database)
        try:
            scheduler = TaskScheduler(store, system=system)
            task = scheduler.create_task("local sample", "interval", "60")
            with pytest.raises(KeyboardInterrupt):
                scheduler._execute_task(task)
            assert store.get_run_logs(task.id)[0]["status"] == "running"
            assert store.get_due_tasks("2099-01-01T00:00:00+00:00") == []
        finally:
            store.close()

        reopened = SchedulerStore(database)
        runner = TaskScheduler(reopened, system=system)
        try:
            runner.start()
            runner.stop()
            system.ask.assert_called_once()
            assert reopened.get_task(task.id)["status"] == "review_required"
            logs = reopened.get_run_logs(task.id)
            assert len(logs) == 1
            assert logs[0]["status"] == "interrupted"
            with pytest.raises(RuntimeError, match="Review the interrupted run"):
                runner.resume_task(task.id)
            runner.resume_task(task.id, reviewed_interruption=True)
            assert reopened.get_task(task.id)["status"] == "active"
            assert reopened.get_due_tasks(datetime.now(timezone.utc).isoformat()) == []
        finally:
            runner.stop()
            reopened.close()

    def test_log_commit_failure_keeps_occurrence_claimed(self, store):
        system = MagicMock()
        runner = TaskScheduler(store, system=system)
        task = runner.create_task("local sample", "once", "2020-01-01T00:00:00+00:00")
        store._conn.execute(
            "CREATE TRIGGER fail_schedule_update BEFORE UPDATE ON scheduled_tasks "
            "BEGIN SELECT RAISE(ABORT, 'simulated write error'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="simulated write error"):
            runner._execute_task(task)
        assert store.get_run_logs(task.id)[0]["status"] == "running"
        assert runner._execute_task(task) is None
        system.ask.assert_called_once()

    def test_one_time_run_paused_during_execution_does_not_repeat(self, store):
        system = MagicMock()
        scheduler = TaskScheduler(store, system=system)
        task = scheduler.create_task(
            "local sample", "once", "2020-01-01T00:00:00+00:00"
        )

        def ask(_prompt, **_kwargs):
            scheduler.pause_task(task.id)
            with pytest.raises(RuntimeError, match="current run"):
                scheduler.resume_task(task.id)
            return "finished"

        system.ask.side_effect = ask
        assert scheduler._execute_task(task) == (True, "finished")
        assert store.get_task(task.id)["status"] == "paused"
        scheduler.resume_task(task.id)
        assert store.get_task(task.id)["status"] == "completed"
        assert scheduler._execute_task(task) is None
        system.ask.assert_called_once()

    def test_manual_dispatch_shares_lease_and_consumes_the_occurrence(self, store):
        system = MagicMock()
        system.ask.return_value = "finished"
        scheduler = TaskScheduler(store, system=system)
        task = scheduler.create_task(
            "local sample", "once", "2099-01-01T00:00:00+00:00", tools="think"
        )
        lease = SchedulerInstanceLock(store.db_path)
        assert lease.acquire()
        try:
            with pytest.raises(RuntimeError, match="Another scheduler"):
                scheduler.run_task_now(task.id)
            system.ask.assert_not_called()
        finally:
            lease.release()
        assert scheduler.run_task_now(task.id) == (True, "finished")
        system.ask.assert_called_once_with(
            "local sample", agent="simple", tools=["think"]
        )
        assert store.get_task(task.id)["status"] == "completed"
        assert store.get_run_logs(task.id)[0]["status"] == "succeeded"

    def test_one_shot_with_unknown_outcome_requires_new_task(self, store):
        scheduler = TaskScheduler(store)
        task = scheduler.create_task(
            "local sample", "once", "2020-01-01T00:00:00+00:00"
        )
        assert store.start_run(task.id, task.next_run, task.schedule_value) is not None
        store.recover_interrupted_runs("2020-01-01T00:01:00+00:00")
        with pytest.raises(RuntimeError, match="create a new task"):
            scheduler.resume_task(task.id, reviewed_interruption=True)
        assert store.get_task(task.id)["status"] == "review_required"


# -- Start / stop lifecycle ---------------------------------------------------


class TestLifecycle:
    def test_start_stop(self, scheduler):
        scheduler.start()
        assert scheduler._thread is not None
        assert scheduler._thread.is_alive()
        scheduler.stop()
        assert not scheduler._thread

    def test_double_start(self, scheduler):
        scheduler.start()
        t1 = scheduler._thread
        scheduler.start()  # Should not create a second thread
        assert scheduler._thread is t1
        scheduler.stop()

    def test_only_one_persistent_scheduler_per_database(self, tmp_path):
        db_path = tmp_path / "shared_scheduler.db"
        first_store = SchedulerStore(db_path)
        second_store = SchedulerStore(db_path)
        first = TaskScheduler(first_store, poll_interval=1)
        second = TaskScheduler(second_store, poll_interval=1)
        try:
            first.start()
            with pytest.raises(RuntimeError, match="Another scheduler is already"):
                second.start()

            first.stop()
            second.start()
            second.stop()
        finally:
            first.stop()
            second.stop()
            first_store.close()
            second_store.close()

    def test_start_clears_a_stale_stop_request(self, scheduler):
        lock = SchedulerInstanceLock(scheduler._store.db_path)
        lock.request_stop()

        scheduler.start()

        assert not scheduler.external_stop_requested
        scheduler.stop()

    def test_external_stop_finishes_current_task_before_remaining_due_tasks(
        self, store
    ):
        entered_first = threading.Event()
        finish_first = threading.Event()
        calls: list[str] = []
        system = MagicMock()

        def ask(prompt: str, **_kwargs):
            calls.append(prompt)
            if prompt == "first task":
                entered_first.set()
                assert finish_first.wait(timeout=5)
            return prompt

        system.ask.side_effect = ask
        scheduler = TaskScheduler(store, system=system, poll_interval=1)
        scheduler.create_task("first task", "once", "2020-01-01T00:00:00+00:00")
        second = scheduler.create_task(
            "second task", "once", "2021-01-01T00:00:00+00:00"
        )
        scheduler.start()
        try:
            assert entered_first.wait(timeout=5)
            lock = SchedulerInstanceLock(store.db_path)
            lock.request_stop()
            finish_first.set()
            assert scheduler._thread is not None
            scheduler._thread.join(timeout=5)
            assert not scheduler._thread.is_alive()
            assert calls == ["first task"]
            assert store.get_run_logs(second.id) == []
        finally:
            finish_first.set()
            scheduler.stop()

    def test_poll_loop_finds_due_tasks(self, store):
        sched = TaskScheduler(store, poll_interval=1)
        # Create a task that is already due
        task = sched.create_task("immediate", "once", "2020-01-01T00:00:00+00:00")
        # Manually set next_run to the past
        d = store.get_task(task.id)
        d["next_run"] = "2020-01-01T00:00:00+00:00"
        store.update_task(d)

        sched.start()
        # Give the poll loop time to execute
        time.sleep(2.5)
        sched.stop()

        logs = store.get_run_logs(task.id)
        assert len(logs) >= 1
