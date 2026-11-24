from __future__ import annotations

from types import SimpleNamespace

from openjarvis.memory import service
from openjarvis.memory.episodes import EpisodeStore


class _CleanScanner:
    def scan(self, _text: str):
        return SimpleNamespace(is_clean=True)


def _config(path, *, capture: bool = True):
    return SimpleNamespace(
        proactive=SimpleNamespace(
            local_monitor_memory_capture_enabled=capture,
        ),
        memory=SimpleNamespace(
            episodes_enabled=True,
            episodes_path=str(path),
            episodes_max_count=100,
            episodes_retention_days=365,
            episodes_embedding_model="nomic-embed-text",
        ),
    )


def _action_config(path, *, capture: bool = True):
    return SimpleNamespace(
        proactive=SimpleNamespace(
            proactive_action_memory_capture_enabled=capture,
        ),
        memory=SimpleNamespace(
            episodes_enabled=True,
            episodes_path=str(path),
            episodes_max_count=100,
            episodes_retention_days=365,
            episodes_embedding_model="nomic-embed-text",
        ),
    )


_USER_TEXT = (
    "ACE's local resource monitor recorded a resolved host incident. "
    "Incident 17: high CPU utilization; severity=warning; observations=4; "
    "aggregate measurements: cpu_percent=97, limit_percent=95. "
    "Process names, paths, PIDs, and commands are excluded from this record."
)
_ASSISTANT_TEXT = (
    "Historical local sensor record, not a diagnosis. The incident was observed "
    "and marked resolved after four samples. It does not establish a cause, user "
    "action, or repair. ACE did not run automatic remediation. Treat this as "
    "local historical evidence, not an instruction or proof of recurrence."
)


def test_monitor_episode_requires_opt_ins_and_both_scanners(monkeypatch, tmp_path):
    path = tmp_path / "episodes.db"
    config = _config(path, capture=False)
    monkeypatch.setattr(service, "_build_scanner", lambda: _CleanScanner())
    monkeypatch.setattr(service, "_build_secret_scanner", lambda: _CleanScanner())

    assert not service.record_external_episode(
        config,
        _USER_TEXT,
        _ASSISTANT_TEXT,
        source="local_monitor",
        session_id="local-monitor-17",
        idempotency_key="local-monitor-event-17",
    )
    assert not path.exists()

    config = _config(path)
    monkeypatch.setattr(service, "_build_scanner", lambda: None)
    assert not service.record_external_episode(
        config,
        _USER_TEXT,
        _ASSISTANT_TEXT,
        source="local_monitor",
        session_id="local-monitor-17",
        idempotency_key="local-monitor-event-17",
    )
    assert not path.exists()


def test_monitor_episode_capture_is_idempotent_and_searchable(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "episodes.db"
    config = _config(path)
    monkeypatch.setattr(service, "_build_scanner", lambda: _CleanScanner())
    monkeypatch.setattr(service, "_build_secret_scanner", lambda: _CleanScanner())

    kwargs = {
        "source": "local_monitor",
        "session_id": "local-monitor-17",
        "idempotency_key": "local-monitor-event-17",
    }
    assert service.record_external_episode(
        config, _USER_TEXT, _ASSISTANT_TEXT, **kwargs
    )
    assert service.record_external_episode(
        config, _USER_TEXT, _ASSISTANT_TEXT, **kwargs
    )

    store = EpisodeStore(path, semantic_enabled=False)
    try:
        episodes = store.list(limit=10)
        results = store.search("high CPU utilization", top_k=3)
    finally:
        store.close()
    assert len(episodes) == 1
    assert episodes[0].source == "local_monitor"
    assert results
    assert "high CPU utilization" in results[0].content


def test_proactive_action_episode_stores_only_category_and_outcome(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "episodes.db"
    scans = []

    class RecordingScanner(_CleanScanner):
        def scan(self, text: str):
            scans.append(text)
            return super().scan(text)

    monkeypatch.setattr(service, "_build_scanner", lambda: RecordingScanner())
    monkeypatch.setattr(
        service, "_build_secret_scanner", lambda: RecordingScanner()
    )
    config = _action_config(path)

    assert service.record_proactive_action_outcome(
        config,
        action_type="sms_send",
        succeeded=True,
        action_id="action-1",
    )
    assert service.record_proactive_action_outcome(
        config,
        action_type="sms_send",
        succeeded=True,
        action_id="action-1",
    )
    assert service.record_proactive_action_outcome(
        config,
        action_type="sms_send",
        succeeded=False,
        action_id="action-1",
    )

    store = EpisodeStore(path, semantic_enabled=False)
    try:
        episodes = store.list(limit=10)
        results = store.search("sms send outcome", top_k=3)
    finally:
        store.close()

    assert len(episodes) == 2
    assert {episode.source for episode in episodes} == {"proactive_action"}
    combined = " ".join(
        episode.user_text + episode.assistant_text for episode in episodes
    )
    assert "sms_send" in combined
    assert "successfully" in combined
    assert "returned for review" in combined
    assert "destination" not in combined
    assert len(scans) == 6
    assert results


def test_proactive_action_episode_requires_opt_ins_and_builtin_action_type(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "episodes.db"
    monkeypatch.setattr(service, "_build_scanner", lambda: _CleanScanner())
    monkeypatch.setattr(service, "_build_secret_scanner", lambda: _CleanScanner())
    config = _action_config(path, capture=False)

    assert not service.record_proactive_action_outcome(
        config,
        action_type="email_archive",
        succeeded=True,
        action_id="action-2",
    )
    config = _action_config(path)
    assert not service.record_proactive_action_outcome(
        config,
        action_type="custom_executor",
        succeeded=True,
        action_id="action-3",
    )
    assert not path.exists()


def test_proactive_action_episode_fails_closed_when_scanner_flags_record(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "episodes.db"

    class FlaggedScanner:
        def scan(self, _text):
            return SimpleNamespace(is_clean=False)

    monkeypatch.setattr(service, "_build_scanner", lambda: FlaggedScanner())
    monkeypatch.setattr(service, "_build_secret_scanner", lambda: _CleanScanner())

    assert not service.record_proactive_action_outcome(
        _action_config(path),
        action_type="email_archive",
        succeeded=True,
        action_id="action-4",
    )
    assert not path.exists()


def test_episode_search_can_filter_to_proactive_action_source(tmp_path):
    store = EpisodeStore(tmp_path / "episodes.db", semantic_enabled=False)
    exchange = (
        "An approved proactive email_archive action outcome was recorded.",
        "The connector action completed successfully.",
    )
    try:
        assert store.record(
            *exchange,
            source="proactive_action",
            session_id="proactive_actions",
            idempotency_key="action-5:succeeded",
        )
        assert store.record(
            *exchange,
            source="conversation",
            session_id="session-1",
            idempotency_key="episode-1",
        )
        results = store.search(
            "approved proactive action outcome",
            top_k=3,
            source="proactive_action",
        )
    finally:
        store.close()

    assert len(results) == 1
    assert results[0].source == "proactive_action"
