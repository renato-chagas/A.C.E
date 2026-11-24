"""CLI tests for scheduler login startup management."""

from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from openjarvis.cli import scheduler_cmd
from openjarvis.core import config as core_config
from openjarvis.scheduler import startup


def test_install_startup_command_reports_opt_in_install(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        startup,
        "install_scheduler_startup",
        lambda: {"output": "ACE scheduler will start at the next login."},
    )

    result = CliRunner().invoke(scheduler_cmd.scheduler, ["install-startup"])

    assert result.exit_code == 0, result.output
    assert "start at the next login" in result.output
    assert "remove-startup" in result.output


def test_startup_status_command_reports_not_installed(monkeypatch) -> None:
    monkeypatch.setattr(
        startup,
        "get_scheduler_startup_status",
        lambda: {"installed": False, "state": "not installed"},
    )

    result = CliRunner().invoke(scheduler_cmd.scheduler, ["startup-status"])

    assert result.exit_code == 0, result.output
    assert "not installed" in result.output


def test_stop_command_reports_when_scheduler_is_not_running(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        core_config,
        "load_config",
        lambda: SimpleNamespace(scheduler=SimpleNamespace(db_path="")),
    )
    monkeypatch.setattr(core_config, "DEFAULT_CONFIG_DIR", tmp_path)

    result = CliRunner().invoke(scheduler_cmd.scheduler, ["stop"])

    assert result.exit_code == 0, result.output
    assert "No persistent scheduler is running" in result.output


def test_stop_command_signals_the_running_scheduler(tmp_path, monkeypatch) -> None:
    from openjarvis.scheduler.scheduler import TaskScheduler
    from openjarvis.scheduler.store import SchedulerStore

    database_path = tmp_path / "scheduler.db"
    store = SchedulerStore(database_path)
    running = TaskScheduler(store, poll_interval=1)
    running.start()
    monkeypatch.setattr(
        core_config,
        "load_config",
        lambda: SimpleNamespace(scheduler=SimpleNamespace(db_path=str(database_path))),
    )
    try:
        result = CliRunner().invoke(scheduler_cmd.scheduler, ["stop"])

        assert result.exit_code == 0, result.output
        assert "will finish its current task" in result.output
        assert running.external_stop_requested
    finally:
        running.stop()
        running.clear_external_stop_request()
        store.close()


def test_scheduler_start_exits_after_cooperative_stop(monkeypatch) -> None:
    class _Scheduler:
        external_stop_requested = True

        def start(self) -> None:
            pass

        def clear_external_stop_request(self) -> None:
            pass

    stopped: list[bool] = []
    scheduler = _Scheduler()
    monkeypatch.setattr(
        scheduler_cmd,
        "_build_scheduler_runtime",
        lambda _interval, dry_run: (scheduler, None, object()),
    )
    monkeypatch.setattr(
        scheduler_cmd,
        "_stop_scheduler_runtime",
        lambda _sched, _system, _store: stopped.append(True),
    )
    monkeypatch.setattr(scheduler_cmd.signal, "signal", lambda *_args: None)

    result = CliRunner().invoke(scheduler_cmd.scheduler, ["start"])

    assert result.exit_code == 0, result.output
    assert "Scheduler stopped" in result.output
    assert stopped == [True]
