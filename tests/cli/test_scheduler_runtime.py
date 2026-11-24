from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from click.testing import CliRunner

from openjarvis.cli import scheduler_cmd
from openjarvis.scheduler.scheduler import ScheduledTask


def test_scheduler_runtime_uses_live_system_by_default(monkeypatch) -> None:
    config = SimpleNamespace(scheduler=SimpleNamespace(enabled=False, poll_interval=60))
    system_scheduler = MagicMock()
    system = SimpleNamespace(scheduler=system_scheduler, close=MagicMock())
    captured = {}

    monkeypatch.setattr("openjarvis.core.config.load_config", lambda: config)

    class FakeBuilder:
        def __init__(self, received_config):
            captured["config"] = received_config

        def scheduler(self, enabled):
            captured["enabled"] = enabled
            return self

        def build(self):
            return system

    monkeypatch.setattr("openjarvis.system.builder.SystemBuilder", FakeBuilder)

    scheduler, built_system, store = scheduler_cmd._build_scheduler_runtime(
        15, dry_run=False
    )

    assert scheduler is system_scheduler
    assert built_system is system
    assert store is None
    assert captured["enabled"] is True
    assert config.scheduler.enabled is True
    assert config.scheduler.poll_interval == 15


def test_dry_run_runtime_does_not_build_a_system(monkeypatch) -> None:
    store = MagicMock()
    scheduler = MagicMock()
    monkeypatch.setattr(scheduler_cmd, "_get_store", lambda: store)
    monkeypatch.setattr(
        "openjarvis.scheduler.scheduler.TaskScheduler",
        lambda received_store, poll_interval: (
            scheduler if received_store is store and poll_interval == 30 else None
        ),
    )

    built_scheduler, system, built_store = scheduler_cmd._build_scheduler_runtime(
        30, dry_run=True
    )

    assert built_scheduler is scheduler
    assert system is None
    assert built_store is store


def test_scheduler_approve_displays_task_details_and_confirms(monkeypatch) -> None:
    task = ScheduledTask(
        id="task-123",
        prompt="Archive receipts from the inbox",
        schedule_type="cron",
        schedule_value="0 9 * * 1",
        status="pending_approval",
        next_run="2026-11-23T09:00:00+00:00",
        agent="orchestrator",
        tools="email_archive",
    )
    store = MagicMock()
    sched = MagicMock()
    sched.list_tasks.return_value = [task]
    monkeypatch.setattr(scheduler_cmd, "_get_store", lambda: store)
    monkeypatch.setattr(scheduler_cmd, "_get_scheduler", lambda received: sched)

    result = CliRunner().invoke(
        scheduler_cmd.scheduler, ["approve", task.id], input="y\n"
    )

    assert result.exit_code == 0
    assert task.prompt in result.output
    assert task.schedule_value in result.output
    assert task.tools in result.output
    sched.approve_task.assert_called_once_with(task.id)
    store.close.assert_called_once_with()


def test_scheduler_approve_decline_keeps_task_pending(monkeypatch) -> None:
    task = ScheduledTask(
        id="task-456",
        prompt="Send a message",
        schedule_type="once",
        schedule_value="2026-11-25T10:00:00+00:00",
        status="pending_approval",
    )
    store = MagicMock()
    sched = MagicMock()
    sched.list_tasks.return_value = [task]
    monkeypatch.setattr(scheduler_cmd, "_get_store", lambda: store)
    monkeypatch.setattr(scheduler_cmd, "_get_scheduler", lambda received: sched)

    result = CliRunner().invoke(
        scheduler_cmd.scheduler, ["approve", task.id], input="n\n"
    )

    assert result.exit_code == 0
    assert "remains pending" in result.output
    sched.approve_task.assert_not_called()


def test_scheduler_resume_requires_review_of_unknown_run(monkeypatch) -> None:
    store = MagicMock()
    store.get_task.return_value = {
        "status": "review_required",
        "schedule_type": "interval",
    }
    sched = MagicMock()
    monkeypatch.setattr(scheduler_cmd, "_get_store", lambda: store)
    monkeypatch.setattr(scheduler_cmd, "_get_scheduler", lambda received: sched)

    argv = ["resume", "task-interrupted"]
    declined = CliRunner().invoke(scheduler_cmd.scheduler, argv, input="n\n")
    assert declined.exit_code == 0
    assert "may have taken effect" in declined.output
    sched.resume_task.assert_not_called()

    confirmed = CliRunner().invoke(scheduler_cmd.scheduler, argv, input="y\n")
    assert confirmed.exit_code == 0
    sched.resume_task.assert_called_once_with(
        "task-interrupted", reviewed_interruption=True
    )
