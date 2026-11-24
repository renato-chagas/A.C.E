from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openjarvis.proactive import local_monitor
from openjarvis.proactive.local_monitor import (
    LocalMonitorHistoryTool,
    LocalMonitorStore,
    LocalProactiveMonitor,
)
from openjarvis.proactive.reasoning import LocalIncidentReasoner


def test_read_assessment_incident_is_exact_read_only_and_includes_recovered(tmp_path):
    path = tmp_path / "nested" / "monitor.db"
    assert LocalMonitorStore.read_assessment_incident(1, path) is None
    assert not path.parent.exists()
    store = LocalMonitorStore(path)
    details = {"cpu_percent": 97, "process_load": {"top_cpu": [{"name": "fixture"}]}}
    assert store.observe("cpu_high", "warning", "Earlier CPU", details)
    earlier_id = store.recent(1)[0]["id"]
    assert store.resolve("cpu_high")
    assert store.observe("cpu_high", "warning", "Later CPU", {"cpu_percent": 92})
    original = path.read_bytes()
    recovered = LocalMonitorStore.read_assessment_incident(earlier_id, path)
    assert recovered["id"] == earlier_id
    assert recovered["status"] == "resolved"
    assert recovered["details"] == details
    assert recovered["episodes_last_30d"] == 2
    assert LocalMonitorStore.read_assessment_incident(999, path) is None
    assert path.read_bytes() == original


def test_save_panel_assessment_checks_snapshot_and_deduplicates_across_stores(tmp_path):
    path = tmp_path / "monitor.db"
    store = LocalMonitorStore(path)
    second_store = LocalMonitorStore(path)
    assert store.observe("disk_low", "warning", "Low disk", {"free_percent": 3})
    incident_id = store.recent(1)[0]["id"]
    snapshot = LocalMonitorStore.read_assessment_incident(incident_id, path)
    long_text = "x" * 1400
    with ThreadPoolExecutor(max_workers=2) as pool:
        same = list(
            pool.map(
                lambda writer: writer.save_panel_assessment(snapshot, long_text),
                [store, second_store],
            )
        )
    assert same[0] == same[1]
    assert same[0]["analysis"] == "x" * 1200
    incident = store.recent(1)[0]
    assert incident["assessments"] == [same[0]]
    assert incident["acknowledged_at"] is None
    assert store.pending_context()[0]["id"] == incident_id

    assert not store.observe("disk_low", "warning", "Low disk", {"free_percent": 2})
    assert store.save_panel_assessment(snapshot, "Old observations") is None
    newer = LocalMonitorStore.read_assessment_incident(incident_id, path)
    assert store.resolve("disk_low")
    assert store.save_panel_assessment(newer, "Old incident status") is None
    resolved = LocalMonitorStore.read_assessment_incident(incident_id, path)
    assert store.save_panel_assessment(resolved, "Historical reading")["analysis"] == (
        "Historical reading"
    )


@pytest.mark.parametrize("record_measured_assessments", [False, True])
def test_dashboard_monitor_assesses_new_measurements_and_escalations_only(
    tmp_path, monkeypatch, record_measured_assessments
):
    from openjarvis.core.config import ProactiveConfig

    store = LocalMonitorStore(tmp_path / "monitor.db")
    config = ProactiveConfig(local_monitor_cpu_percent=80)
    monitor = LocalProactiveMonitor(
        config, store=store, record_measured_assessments=record_measured_assessments
    )
    readings = iter((91.0, 92.0, 94.0, 99.0, 100.0))
    monkeypatch.setattr(monitor, "_sample", lambda: {"cpu_percent": next(readings)})

    for _ in range(5):
        monitor.poll_once()

    incident = store.recent(1)[0]
    assert incident["observations"] == 4
    assert incident["severity"] == "critical"
    assert incident["acknowledged_at"] is None
    assessments = [item["analysis"] for item in incident["assessments"]]
    if record_measured_assessments:
        assert len(assessments) == 2
        assert "CPU usage measured 99.0% against the configured 80.0%" in assessments[0]
        assert "CPU usage measured 92.0% against the configured 80.0%" in assessments[1]
        assert all("remains pending" not in assessment for assessment in assessments)
    else:
        assert assessments == []


def test_assessments_persist_with_incidents_and_are_bounded(tmp_path) -> None:
    path = tmp_path / "monitor.db"
    store = LocalMonitorStore(path)
    assert store.observe("cpu_high", "warning", "CPU high", {"cpu_percent": 97})
    incident_id = int(store.recent(1)[0]["id"])

    assert store.save_assessment(incident_id, "  Measured CPU is high.  ")
    assert store.save_assessment(incident_id, "x" * 5000)
    assert not store.save_assessment(9999, "No matching incident")

    reopened = LocalMonitorStore(path)
    history = reopened.recent(1)[0]
    assert [entry["analysis"] for entry in history["assessments"]] == [
        "x" * 1200,
        "Measured CPU is high.",
    ]
    assert reopened.recent_assessments(limit=1)[0]["analysis"] == "x" * 1200


def test_acknowledgement_is_durable_and_reopened_on_escalation(
    tmp_path, monkeypatch
) -> None:
    now = "2026-11-18T12:00:00+00:00"
    monkeypatch.setattr(local_monitor, "_utc_now", lambda: now)
    path = tmp_path / "monitor.db"
    store = LocalMonitorStore(path)
    store.observe("cpu_high", "warning", "CPU high", {"cpu_percent": 97})
    incident_id = store.recent(1)[0]["id"]
    assert store.save_assessment(incident_id, "Saved assessment.")
    assert store.acknowledge(incident_id)

    store = LocalMonitorStore(path)
    first_ack = now
    now = "2026-11-18T12:05:00+00:00"
    assert store.acknowledge(incident_id)
    assert not store.acknowledge(incident_id + 1)
    assert store.recent(1, unacknowledged_only=True) == []
    assert [incident["id"] for incident in store.pending_context()] == [incident_id]

    store.observe("cpu_high", "warning", "CPU still high", {"cpu_percent": 98})
    assert store.recent(1)[0]["acknowledged_at"] == first_ack

    assert not store.observe(
        "cpu_high", "critical", "CPU critical", {"cpu_percent": 99}
    )
    incident = store.recent(1, unacknowledged_only=True)[0]
    assert incident["id"] == incident_id
    assert incident["status"] == "active"
    assert incident["acknowledged_at"] is None
    assert incident["observations"] == 3
    assert [entry["analysis"] for entry in incident["assessments"]] == [
        "Saved assessment."
    ]

    assert store.resolve("cpu_high")
    assert store.acknowledge(incident_id)
    incident = LocalMonitorStore(path).recent(1)[0]
    assert incident["status"] == "resolved"
    assert incident["acknowledged_at"] == now


def test_acknowledgement_column_migrates_existing_history(tmp_path) -> None:
    path = tmp_path / "monitor.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE local_monitor_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                summary TEXT NOT NULL,
                details_json TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                resolved_at TEXT,
                observations INTEGER NOT NULL DEFAULT 1
            )"""
        )
        conn.execute(
            "INSERT INTO local_monitor_events "
            "(metric, severity, summary, details_json, first_seen, last_seen) "
            "VALUES ('disk_low', 'warning', 'Old incident', '{}', ?, ?)",
            (local_monitor._utc_now(), local_monitor._utc_now()),
        )

    store = LocalMonitorStore(path)
    old_incident = store.recent(1, unacknowledged_only=True)[0]
    assert old_incident["summary"] == "Old incident"
    assert old_incident["acknowledged_at"] is None
    assert store.acknowledge(old_incident["id"])
    assert LocalMonitorStore(path).recent(1)[0]["acknowledged_at"]


def test_rule_based_fallback_is_saved_only_once_per_incident(tmp_path) -> None:
    store = LocalMonitorStore(tmp_path / "monitor.db")
    store.observe("cpu_high", "warning", "CPU high", {"cpu_percent": 97})
    incident_id = int(store.recent(1)[0]["id"])
    prefix = "Rule-based local fallback (no language model; cause not inferred).\n"

    assert store.save_assessment(
        incident_id, prefix + "First measurement.", fallback_once=True
    )
    assert not store.save_assessment(
        incident_id,
        prefix + "A later measurement.",
        fallback_once=True,
    )
    assert len(store.recent(1)[0]["assessments"]) == 1
    assert [incident["id"] for incident in store.pending_context()] == [incident_id]


def test_failed_assessment_write_leaves_incident_pending_after_reopen(tmp_path):
    path = tmp_path / "monitor.db"
    store = LocalMonitorStore(path)
    store.observe("cpu_high", "warning", "CPU high", {"cpu_percent": 97})
    incident = store.pending_context()[0]
    engine = SimpleNamespace(
        is_local=True,
        generate=MagicMock(
            side_effect=[
                {"content": None},
                {"content": "This assessment cannot be saved."},
                {"content": "Durable assessment."},
            ]
        ),
    )
    config = SimpleNamespace(agent=SimpleNamespace(context_from_memory=False))
    reasoner = LocalIncidentReasoner(engine, "local-model", config, monitor_store=store)

    assert reasoner.analyze(incident) == ""
    assert len(store.pending_context()) == 1
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TRIGGER fail_assessment_write
            BEFORE INSERT ON local_monitor_assessments
            BEGIN
                SELECT RAISE(ABORT, 'assessment disk write failed');
            END
            """
        )
    with pytest.raises(sqlite3.IntegrityError, match="assessment disk write failed"):
        reasoner.analyze(incident)

    reopened = LocalMonitorStore(path)
    assert [item["id"] for item in reopened.pending_context()] == [incident["id"]]
    assert reopened.recent(1)[0]["assessments"] == []
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TRIGGER fail_assessment_write")
    reasoner = LocalIncidentReasoner(
        engine, "local-model", config, monitor_store=reopened
    )
    assert reasoner.analyze(incident) == "Durable assessment."

    reopened = LocalMonitorStore(path)
    assert reopened.pending_context() == []
    assert [item["analysis"] for item in reopened.recent(1)[0]["assessments"]] == [
        "Durable assessment."
    ]
    assert not reopened.complete_pending_assessment(
        incident["id"], "warning", "Duplicate assessment."
    )


def test_stale_assessment_does_not_consume_a_changed_or_resolved_incident(tmp_path):
    store = LocalMonitorStore(tmp_path / "monitor.db")
    store.observe("cpu_high", "warning", "CPU high", {"cpu_percent": 97})
    incident_id = store.pending_context()[0]["id"]
    store.observe("cpu_high", "critical", "CPU critical", {"cpu_percent": 99})

    assert not store.complete_pending_assessment(incident_id, "warning", "Stale.")
    assert store.pending_context()[0]["severity"] == "critical"
    assert store.recent(1)[0]["assessments"] == []

    assert store.resolve("cpu_high")
    assert not store.complete_pending_assessment(incident_id, "critical", "Stale.")
    assert store.recent(1)[0]["assessments"] == []


def test_expiring_incident_also_expires_its_assessments(tmp_path) -> None:
    path = tmp_path / "monitor.db"
    store = LocalMonitorStore(path)
    store.observe("disk_low", "warning", "Disk low", {"free_percent": 4})
    incident_id = int(store.recent(1)[0]["id"])
    store.save_assessment(incident_id, "Old local assessment.")
    old_date = "2000-01-01T00:00:00+00:00"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE local_monitor_events SET status='resolved', first_seen=? "
            "WHERE id=?",
            (old_date, incident_id),
        )

    reopened = LocalMonitorStore(path)

    assert reopened.recent(1) == []
    assert reopened.recent_assessments() == []


def test_history_tool_exposes_saved_assessments_to_later_chat(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(local_monitor, "get_config_dir", lambda: tmp_path)
    store = LocalMonitorStore()
    store.observe("cpu_high", "warning", "CPU high", {"cpu_percent": 97})
    incident_id = int(store.recent(1)[0]["id"])
    store.save_assessment(incident_id, "Earlier measured assessment.")

    result = LocalMonitorHistoryTool().execute(limit=5)

    assert result.success is True
    incidents = json.loads(result.content)
    assert incidents[0]["assessments"][0]["analysis"] == (
        "Earlier measured assessment."
    )


def test_incident_reasoner_uses_prior_assessments_as_untrusted_history(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(local_monitor, "_utc_now", lambda: "2026-11-18T12:00:00+00:00")
    store = LocalMonitorStore(tmp_path / "monitor.db")
    store.observe("cpu_high", "warning", "CPU high", {"cpu_percent": 97})
    first_incident = store.recent(1)[0]

    engine = MagicMock()
    engine.engine_id = "custom-local"
    engine.is_cloud = False
    engine.is_local = True
    engine.generate.side_effect = [
        {"content": "First local assessment."},
        {"content": "Second local assessment."},
    ]
    config = SimpleNamespace(agent=SimpleNamespace(context_from_memory=False))
    reasoner = LocalIncidentReasoner(
        engine,
        "model",
        config,
        monitor_store=store,
    )

    assert reasoner.analyze(first_incident) == "First local assessment."
    store.observe("memory_low", "warning", "Memory low", {"available_percent": 8})
    second_incident = store.recent(1)[0]
    assert reasoner.analyze(second_incident) == "Second local assessment."

    messages = engine.generate.call_args.args[0]
    assert "Avaliações anteriores" in messages[0].content
    assert "proposta de próximo passo reversível" in messages[0].content
    assert "confirmação explícitos do usuário" in messages[0].content
    assert "Nunca diga que uma ação foi executada" in messages[0].content
    assert "First local assessment." in messages[1].content
    assert '"prior_assessments"' in messages[1].content
    assert store.recent(1)[0]["assessments"][0]["analysis"] == (
        "Second local assessment."
    )
