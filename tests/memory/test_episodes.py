from __future__ import annotations

import time

import pytest

from openjarvis.memory.episodes import ArchiveClearIntent, EpisodeStore


def test_forgetting_episode_cascades_search_vectors_and_procedure_records(tmp_path):
    path = tmp_path / "episodes.db"
    store = EpisodeStore(path, semantic_enabled=False)
    episode_id = store.episode_id_for(
        source="chat",
        session_id="session-1",
        idempotency_key="quasar-turn",
    )
    assert store.record(
        "We chose the quasarwidget design for this local project because it is "
        "easy to recover and maintain.",
        "I will remember that this was the selected design.",
        source="chat",
        session_id="session-1",
        idempotency_key="quasar-turn",
    )
    now = time.time()
    with store._conn:
        store._conn.execute(
            "INSERT INTO episode_embeddings "
            "(episode_id, model_version, vector, updated_at) VALUES (?, ?, ?, ?)",
            (episode_id, "test-model", b"test-vector", now),
        )
        store._conn.execute(
            "INSERT INTO procedure_proposals "
            "(proposal_id, evidence_episode_id, session_id, skill_name, title, "
            "description, draft_markdown, success_signal, status, skill_path, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "proposal-1",
                episode_id,
                "session-1",
                "quasarwidget_recovery",
                "Recover quasarwidget state",
                "Restore the selected local design.",
                "# Recovery\n\n1. Restore local state.\n",
                "The project starts successfully.",
                "proposed",
                "",
                now,
                now,
            ),
        )
        store._conn.execute(
            "INSERT INTO procedure_outcomes "
            "(outcome_id, evidence_episode_id, session_id, task_title, outcome, "
            "signal, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "outcome-1",
                episode_id,
                "session-1",
                "Recover quasarwidget state",
                "failure",
                "The prior recovery steps failed.",
                now,
            ),
        )

    assert store.search("quasarwidget")
    assert store.delete(episode_id)
    assert store.search("quasarwidget") == []
    for table in (
        "episodes",
        "episode_embeddings",
        "procedure_proposals",
        "procedure_outcomes",
    ):
        assert store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    store.close()

    reopened = EpisodeStore(path, semantic_enabled=False)
    try:
        assert reopened.search("quasarwidget") == []
        assert reopened.get_by_id(episode_id) is None
        assert (
            reopened._conn.execute(
                "SELECT COUNT(*) FROM episode_embeddings"
            ).fetchone()[0]
            == 0
        )
    finally:
        reopened.close()


def test_archive_clear_intent_survives_reopen_and_keeps_individual_intents(tmp_path):
    path = tmp_path / "episodes.db"
    episode_id = "a" * 32
    pending_id = "b" * 32
    intent = ArchiveClearIntent(
        token="c" * 32,
        episode_ids=(episode_id,),
        graph_episode_ids=(episode_id,),
        conversation_sources=("conversation:test:old",),
    )
    store = EpisodeStore(path, semantic_enabled=False)
    store.record_deletion_intent(episode_id)
    store.record_archive_clear(intent)
    store.close()

    reopened = EpisodeStore(path, semantic_enabled=False)
    try:
        assert reopened.pending_archive_clear() == intent
        reopened.record_deletion_intent(pending_id)
        assert reopened.finish_archive_clear(intent) == 0
        assert reopened.pending_archive_clear() is None
        assert reopened.pending_deletion_intents() == [episode_id, pending_id]
        reopened.finish_deletion_intent(episode_id)
        assert reopened.pending_deletion_intents() == [pending_id]
        with pytest.raises(ValueError, match="32-character"):
            reopened.record_deletion_intent("invalid")
        with reopened._conn:
            reopened._conn.execute(
                "INSERT INTO episode_deletion_journal (scope_id, created_at) "
                "VALUES (?, ?)",
                ("invalid", time.time()),
            )
        with pytest.raises(ValueError, match="32-character"):
            reopened.pending_deletion_intents()
    finally:
        reopened.close()


def test_read_retained_respects_age_and_count_without_pruning(tmp_path):
    path = tmp_path / "episodes.db"
    store = EpisodeStore(path, retention_days=0)
    try:
        assert store.record("Old project decision.", "Old text.")
        older_id = store.list()[0].episode_id
        with store._conn:
            store._conn.execute(
                "UPDATE episodes SET created_at=? WHERE episode_id=?",
                (time.time() - 3 * 86_400, older_id),
            )
        assert store.record("New project decision.", "New text.")
        newer_id = store.list()[0].episode_id
        assert EpisodeStore.read_retained(older_id, path=path, retention_days=1) is None
        assert EpisodeStore.read_retained(older_id, path=path, max_episodes=1) is None
        assert EpisodeStore.read_retained(newer_id, path=path, max_episodes=1)
        assert store.get_by_id(older_id) is not None
    finally:
        store.close()
