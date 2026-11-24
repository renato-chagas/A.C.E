from __future__ import annotations

from types import SimpleNamespace

from openjarvis.core.config import ProactiveConfig
from openjarvis.memory import service
from openjarvis.memory.episodes import EpisodeStore
from openjarvis.proactive.local_monitor import (
    LocalMonitorStore,
    LocalProactiveMonitor,
)


class _CleanScanner:
    def scan(self, _text: str):
        return SimpleNamespace(is_clean=True)


def test_resolved_monitor_incident_is_consolidated_without_process_details(
    monkeypatch,
    tmp_path,
):
    monitor_store = LocalMonitorStore(tmp_path / "monitor.db")
    assert monitor_store.observe(
        "cpu_high",
        "warning",
        "CPU usage high: 97%",
        {
            "cpu_percent": 97,
            "limit_percent": 95,
            "process_load": [
                {"name": "private-process-name", "pid": 1234, "cpu_percent": 50}
            ],
        },
    )
    assert monitor_store.pending_memory_incidents() == []

    episodes_path = tmp_path / "episodes.db"
    full_config = SimpleNamespace(
        proactive=SimpleNamespace(local_monitor_memory_capture_enabled=True),
        memory=SimpleNamespace(
            episodes_enabled=True,
            episodes_path=str(episodes_path),
            episodes_max_count=100,
            episodes_retention_days=365,
            episodes_embedding_model="nomic-embed-text",
        ),
    )
    monkeypatch.setattr(service, "_build_scanner", lambda: _CleanScanner())
    monkeypatch.setattr(service, "_build_secret_scanner", lambda: _CleanScanner())
    monitor = LocalProactiveMonitor(
        ProactiveConfig(local_monitor_memory_capture_enabled=True),
        store=monitor_store,
    )

    monkeypatch.setattr(
        LocalProactiveMonitor,
        "_sample",
        staticmethod(lambda: {"cpu_percent": 50.0}),
    )
    monkeypatch.setattr(
        "openjarvis.core.config.load_config",
        lambda: full_config,
    )
    notices = monitor.poll_once()
    assert any("CPU" in notice for notice in notices)
    assert monitor.consolidate_pending_memory(full_config) == 0
    assert monitor_store.pending_memory_incidents() == []

    reopened_monitor_store = LocalMonitorStore(monitor_store.path)
    assert reopened_monitor_store.recent(1)[0]["memory_captured_at"]

    episodes = EpisodeStore(episodes_path, semantic_enabled=False)
    try:
        stored = episodes.list(limit=5)
        results = episodes.search("high CPU utilization", top_k=3)
    finally:
        episodes.close()
    assert len(stored) == 1
    assert stored[0].source == "local_monitor"
    assert "private-process-name" not in stored[0].user_text
    assert "private-process-name" not in stored[0].assistant_text
    assert results
    assert "cpu_percent=97" in results[0].content


def test_monitor_capture_requires_both_memory_flags(tmp_path):
    store = LocalMonitorStore(tmp_path / "monitor.db")
    store.observe("disk_low", "warning", "Disk low", {"free_percent": 4})
    store.resolve("disk_low")
    monitor = LocalProactiveMonitor(
        ProactiveConfig(local_monitor_memory_capture_enabled=True),
        store=store,
    )
    config = SimpleNamespace(
        proactive=SimpleNamespace(local_monitor_memory_capture_enabled=False),
        memory=SimpleNamespace(episodes_enabled=True),
    )

    assert monitor.consolidate_pending_memory(config) == 0
    assert len(store.pending_memory_incidents()) == 1
