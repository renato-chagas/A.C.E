"""Tests for the dashboard's bounded control of the local monitor worker."""

from __future__ import annotations

import json

import pytest

from openjarvis.proactive import monitor_daemon
from openjarvis.proactive.desktop_monitor import (
    DesktopMonitorConflict,
    DesktopMonitorController,
)


def _state(*, desktop: bool, phase: str) -> dict:
    return {
        "pid": 42,
        "process_created": 1700000000,
        "launch_token": "test-token",
        "status": phase,
        "started_at": "2026-01-02T03:04:05+00:00",
        "log_file": "monitor.log",
        "screen_monitor_enabled": not desktop,
        "desktop_resource_only": desktop,
    }


def test_dashboard_inspection_does_not_create_or_repair_worker_files(
    tmp_path, monkeypatch
) -> None:
    home = tmp_path / "headless-profile"
    monkeypatch.setenv("OPENJARVIS_HOME", str(home))
    controller = DesktopMonitorController(None)

    assert controller.status() == (False, False, False, False)
    assert not home.exists()

    home.mkdir()
    record = home / "proactive_monitor.json"
    record.write_text(
        json.dumps(_state(desktop=True, phase="running")), encoding="utf-8"
    )
    original = record.read_bytes()
    monkeypatch.setattr(monitor_daemon, "_process_matches", lambda _state: False)

    assert controller.status() == (False, False, False, False)
    assert record.read_bytes() == original
    assert not (home / "proactive_monitor.control.lock").exists()


def test_dashboard_preserves_a_worker_started_outside_the_app(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))
    monkeypatch.setattr(monitor_daemon, "_process_matches", lambda _state: True)
    record = tmp_path / "proactive_monitor.json"
    external = _state(desktop=False, phase="running")
    del external["desktop_resource_only"]  # Existing profiles use this older schema.
    record.write_text(json.dumps(external), encoding="utf-8")
    original = record.read_bytes()
    controller = DesktopMonitorController(None)

    assert controller.status() == (True, True, True, False)
    with pytest.raises(DesktopMonitorConflict):
        controller.set_enabled(True)
    with pytest.raises(DesktopMonitorConflict):
        controller.set_enabled(False)
    assert record.read_bytes() == original


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        ("starting", (True, False, False, True)),
        ("running", (True, True, False, True)),
        ("stopping", (False, True, False, False)),
    ],
)
def test_dashboard_dispatches_only_to_its_registered_worker(
    tmp_path, monkeypatch, phase, expected
):
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))
    monkeypatch.setattr(monitor_daemon, "_process_matches", lambda _state: True)
    record = tmp_path / "proactive_monitor.json"
    state = _state(desktop=True, phase=phase)
    state["desktop_notifications_enabled"] = True
    record.write_text(json.dumps(state), encoding="utf-8")
    controller = DesktopMonitorController(None)
    stopped = []
    monkeypatch.setattr(monitor_daemon, "_stop_worker", stopped.append)

    assert controller.status() == expected
    if phase == "stopping":
        with pytest.raises(DesktopMonitorConflict):
            controller.set_enabled(True, notifications_enabled=True)
        with pytest.raises(DesktopMonitorConflict):
            controller.set_enabled(False)
        assert stopped == []
    else:
        controller.set_enabled(True, notifications_enabled=True)
        with pytest.raises(
            DesktopMonitorConflict, match="before changing notifications"
        ):
            controller.set_enabled(True)
        assert json.loads(record.read_text(encoding="utf-8")) == state
        controller.set_enabled(False)
        assert len(stopped) == 1
        assert stopped[0]["launch_token"] == state["launch_token"]
        assert controller.status() == (False, False, False, False)


@pytest.mark.parametrize("flag", ["true", 0, None])
def test_dashboard_rejects_unverified_notification_state(tmp_path, monkeypatch, flag):
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))
    state = _state(desktop=True, phase="running")
    state["desktop_notifications_enabled"] = flag
    (tmp_path / "proactive_monitor.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    assert monitor_daemon._read_state_unlocked() is None


@pytest.mark.parametrize("choice", [None, False, True])
def test_dashboard_worker_uses_only_the_launch_notification_choice(
    tmp_path, monkeypatch, choice
):
    from openjarvis.core import config
    from openjarvis.proactive import local_monitor, notifications

    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))
    profile = config.JarvisConfig()
    profile.proactive.local_monitor_notifications_enabled = True
    profile.proactive.local_monitor_process_context_enabled = True
    profile.proactive.local_monitor_reasoning_enabled = True
    monkeypatch.setattr(config, "load_config", lambda: profile)

    state = _state(desktop=True, phase="starting")
    if choice is not None:
        state["desktop_notifications_enabled"] = choice
    states = iter((state, {**state, "status": "stopping"}))
    monkeypatch.setattr(monitor_daemon, "_worker_state", lambda _token: next(states))
    monkeypatch.setattr(monitor_daemon, "_set_worker_running", lambda _token: state)
    monkeypatch.setattr(monitor_daemon, "_clear_worker_state", lambda _token: None)
    delivered = []
    monkeypatch.setattr(
        notifications,
        "notify_monitor_event",
        lambda message, **kwargs: delivered.append((message, kwargs)) or True,
    )

    class Sampler:
        def __init__(self, proactive, *, on_notice, store, record_measured_assessments):
            assert not proactive.local_monitor_process_context_enabled
            assert not proactive.local_monitor_reasoning_enabled
            assert record_measured_assessments
            self.on_notice = on_notice

        def start(self):
            self.on_notice("CPU precisa de atenção.")
            return True

        def stop(self):
            pass

    monkeypatch.setattr(local_monitor, "LocalProactiveMonitor", Sampler)
    assert monitor_daemon.run_monitor_worker(state["launch_token"]) == 0
    expected = [("CPU precisa de atenção.", {"history_hint": "System Monitoring"})]
    assert delivered == (expected if choice else [])
