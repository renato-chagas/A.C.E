from __future__ import annotations

from click.testing import CliRunner

from openjarvis.cli.monitor_cmd import monitor
from openjarvis.core.config import JarvisConfig
from openjarvis.proactive import local_monitor, monitor_daemon
from openjarvis.proactive.local_monitor import LocalMonitorStore


def test_history_command_shows_persisted_assessment(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(local_monitor, "get_config_dir", lambda: tmp_path)
    store = LocalMonitorStore()
    store.observe("cpu_high", "warning", "CPU usage reached 97%", {"cpu": 97})
    incident_id = int(store.recent(1)[0]["id"])
    store.save_assessment(incident_id, "Possible cause: sustained local workload.")

    result = CliRunner().invoke(monitor, ["--history"])

    assert result.exit_code == 0, result.output
    assert "A.C.E. local monitor history" in result.output
    assert "CPU usage reached 97%" in result.output
    assert "Possible cause: sustained local workload." in result.output
    assert "Episodic memory: not recorded" in result.output


def test_acknowledge_command_updates_history_and_filters_before_limit(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(local_monitor, "get_config_dir", lambda: tmp_path)
    store = LocalMonitorStore()
    store.observe("disk_low", "warning", "Review disk usage", {"free_percent": 4})
    store.observe("cpu_high", "warning", "Review CPU usage", {"cpu": 97})
    incident_id = store.recent(1)[0]["id"]
    runner = CliRunner()

    result = runner.invoke(monitor, ["--acknowledge", str(incident_id)])
    assert result.exit_code == 0, result.output
    assert f"Local incident #{incident_id} acknowledged." in result.output

    result = runner.invoke(monitor, ["--history", "--history-limit", "1"])
    assert result.exit_code == 0, result.output
    assert "warning / active" in result.output
    assert "Review: acknowledged at" in result.output

    result = runner.invoke(
        monitor, ["--history", "--unacknowledged", "--history-limit", "1"]
    )
    assert result.exit_code == 0, result.output
    assert "Review disk usage" in result.output
    assert "Review CPU usage" not in result.output
    assert "Review: awaiting acknowledgement" in result.output

    result = runner.invoke(monitor, ["--acknowledge", "12345"])
    assert result.exit_code == 1
    assert "Local incident #12345 not found" in result.output


def test_unacknowledged_filter_requires_history() -> None:
    result = CliRunner().invoke(monitor, ["--unacknowledged"])

    assert result.exit_code == 2
    assert "--unacknowledged requires --history" in result.output


def test_history_command_reports_empty_store(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(local_monitor, "get_config_dir", lambda: tmp_path)

    result = CliRunner().invoke(monitor, ["--history"])

    assert result.exit_code == 0, result.output
    assert "No local monitor history is available." in result.output


def test_screen_monitor_requires_background_mode() -> None:
    result = CliRunner().invoke(monitor, ["--monitor-screen"])

    assert result.exit_code == 2
    assert "--monitor-screen requires --background" in result.output


def test_background_screen_flag_is_passed_to_daemon(monkeypatch) -> None:
    config = JarvisConfig()
    config.proactive.local_monitor_enabled = True
    monkeypatch.setattr("openjarvis.core.config.load_config", lambda: config)
    monkeypatch.setattr(
        monitor_daemon,
        "start_monitor_daemon",
        lambda interval, *, screen_monitor_enabled: {
            "status": "running",
            "pid": 321,
            "log_file": "monitor.log",
            "screen_monitor_enabled": screen_monitor_enabled,
        },
    )

    result = CliRunner().invoke(
        monitor,
        ["--background", "--monitor-screen"],
    )

    assert result.exit_code == 0, result.output
    assert "Screen monitoring: enabled" in result.output
