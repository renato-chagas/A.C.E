from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from threading import Barrier

import pytest

from openjarvis.memory import episodes
from openjarvis.memory.episodes import ArchiveClearIntent, EpisodeStore


def _record(store, key: str, derivations: tuple[str, ...] = ()) -> str:
    source = "cli.ask"
    assert store.record(
        f"The {key} project uses a local audit trail.",
        "The audit trail can be recovered locally.",
        source=source,
        idempotency_key=key,
        derivations=derivations,
    )
    return EpisodeStore.episode_id_for(source=source, idempotency_key=key)


def test_derivations_are_opt_in_and_persist_across_reopen(tmp_path):
    path = tmp_path / "episodes.db"
    with closing(EpisodeStore(path)) as store:
        _record(store, "legacy")
        # An idempotent replay may not schedule work for an older episode.
        _record(store, "legacy", ("facts",))
        wanted = _record(store, "indigo", ("facts", "graph", "graph"))

    with closing(EpisodeStore(path)) as store:
        facts = store.claim_derivation(("facts",))
        graph = store.claim_derivation(("graph",))
        assert facts is not None and graph is not None
        assert facts.episode.episode_id == graph.episode.episode_id == wanted
        assert facts.kind == "facts" and graph.kind == "graph"
        assert facts.attempts == graph.attempts == 1
        assert store.claim_derivation(("facts", "graph")) is None
        published = []
        assert store.complete_derivation(facts, lambda: published.append("facts"))
        assert not store.complete_derivation(facts, lambda: pytest.fail("Duplicate"))
        assert store.complete_derivation(graph, lambda: published.append("graph"))
        assert published == ["facts", "graph"]
        assert store.claim_derivation(("facts", "graph")) is None


def test_episode_and_queue_rollback_together(tmp_path):
    path = tmp_path / "episodes.db"
    with closing(EpisodeStore(path)) as store, closing(sqlite3.connect(path)) as conn:
        conn.executescript(
            """
            CREATE TRIGGER reject_queue BEFORE INSERT ON episode_derivation_jobs BEGIN
                SELECT RAISE(ABORT, 'queue is unavailable');
            END;
            """
        )
        with pytest.raises(sqlite3.IntegrityError, match="queue is unavailable"):
            _record(store, "rollback", ("facts",))
        assert store.count() == 0
        assert store.claim_derivation(("facts",)) is None
        assert not store.search("rollback")

        conn.execute("DROP TRIGGER reject_queue")
        _record(store, "rollback", ("facts",))
        assert store.claim_derivation(("facts",)) is not None


def test_only_one_connection_can_claim_a_job(tmp_path):
    path = tmp_path / "episodes.db"
    with closing(EpisodeStore(path)) as first, closing(EpisodeStore(path)) as second:
        _record(first, "shared", ("facts",))
        gate = Barrier(2, timeout=5)

        def claim(store):
            gate.wait()
            return store.claim_derivation(("facts",))

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(claim, [first, second]))
        assert sum(result is not None for result in results) == 1


def test_expired_leases_retry_after_reopen_and_reject_the_old_worker(
    tmp_path,
    monkeypatch,
):
    now = [1_800_000_000.0]
    monkeypatch.setattr(episodes.time, "time", lambda: now[0])
    path = tmp_path / "episodes.db"

    with closing(EpisodeStore(path)) as store:
        _record(store, "lease", ("facts",))
        first = store.claim_derivation(("facts",))
        assert first is not None

    with closing(EpisodeStore(path)) as store:
        now[0] += 599
        assert store.claim_derivation(("facts",)) is None
        now[0] += 1
        assert not store.complete_derivation(first, lambda: pytest.fail("Expired"))
        assert not store.fail_derivation(first)
        second = store.claim_derivation(("facts",))
        assert second is not None and second.attempts == 2
        assert second.token != first.token
        assert not store.complete_derivation(first, lambda: pytest.fail("Stale"))
        assert not store.fail_derivation(first)
        assert store.complete_derivation(second, lambda: None)


def test_failed_jobs_obey_backoff_and_can_retry(tmp_path, monkeypatch):
    now = [1_800_000_000.0]
    monkeypatch.setattr(episodes.time, "time", lambda: now[0])
    with closing(EpisodeStore(tmp_path / "episodes.db")) as store:
        _record(store, "backoff", ("graph",))
        first = store.claim_derivation(("graph",))
        assert first is not None

        def abort_publication():
            raise RuntimeError("publication failed")

        with pytest.raises(RuntimeError, match="publication failed"):
            store.complete_derivation(first, abort_publication)
        assert store.fail_derivation(first)
        assert not store.fail_derivation(first)
        unclaimed = replace(first, token="")
        assert not store.complete_derivation(unclaimed, lambda: pytest.fail("No lease"))
        assert not store.fail_derivation(unclaimed)
        now[0] += 14
        assert store.claim_derivation(("graph",)) is None
        now[0] += 1
        second = store.claim_derivation(("graph",))
        assert second is not None and second.attempts == 2
        assert store.fail_derivation(second)

        now[0] += 29
        assert store.claim_derivation(("graph",)) is None
        now[0] += 1
        third = store.claim_derivation(("graph",))
        assert third is not None and third.attempts == 3
        assert store.complete_derivation(third, lambda: None)


def test_deletion_prevents_claims_and_inflight_publication(tmp_path):
    path = tmp_path / "episodes.db"
    with closing(EpisodeStore(path)) as worker, closing(EpisodeStore(path)) as deletes:
        target = _record(worker, "delete", ("facts", "graph"))
        claim = worker.claim_derivation(("facts",))
        assert claim is not None
        deletes.record_deletion_intent(target)
        assert worker.claim_derivation(("graph",)) is None
        assert not worker.complete_derivation(claim, lambda: pytest.fail("Deleting"))
        assert deletes.delete(target)
        deletes.finish_deletion_intent(target)
        assert worker.claim_derivation(("facts", "graph")) is None

        with closing(sqlite3.connect(path)) as probe:
            assert probe.execute(
                "SELECT COUNT(*) FROM episode_derivation_jobs"
            ).fetchone() == (0,)


def test_archive_clear_pauses_publication_and_removes_only_selected_jobs(tmp_path):
    path = tmp_path / "episodes.db"
    with closing(EpisodeStore(path)) as worker, closing(EpisodeStore(path)) as deletes:
        first = _record(worker, "old", ("facts",))
        claim = worker.claim_derivation(("facts",))
        assert claim is not None
        second = _record(worker, "new", ("facts",))
        intent = ArchiveClearIntent("a" * 32, (first,), (), ())
        deletes.record_archive_clear(intent)
        assert worker.claim_derivation(("facts",)) is None
        assert not worker.complete_derivation(claim, lambda: pytest.fail("Clearing"))
        assert deletes.finish_archive_clear(intent) == 1
        assert not worker.complete_derivation(claim, lambda: pytest.fail("Cleared"))
        remaining = worker.claim_derivation(("facts",))
        assert remaining is not None and remaining.episode.episode_id == second


def test_completion_serializes_against_deletion_writers(tmp_path):
    path = tmp_path / "episodes.db"
    with closing(EpisodeStore(path)) as store:
        target = _record(store, "race", ("facts",))
        claim = store.claim_derivation(("facts",))
        assert claim is not None

        def publish():
            with closing(sqlite3.connect(path, timeout=0)) as competing_writer:
                with pytest.raises(
                    sqlite3.OperationalError, match="database is locked"
                ):
                    competing_writer.execute(
                        "INSERT INTO episode_deletion_journal VALUES (?, 0)", (target,)
                    )

        assert store.complete_derivation(claim, publish)
        store.record_deletion_intent(target)
        assert store.pending_deletion_intents() == [target]


def test_retention_removes_queued_derivations(tmp_path, monkeypatch):
    now = [1_800_000_000.0]
    monkeypatch.setattr(episodes.time, "time", lambda: now[0])
    path = tmp_path / "episodes.db"
    with closing(EpisodeStore(path, retention_days=1)) as store:
        _record(store, "expired", ("facts",))
        now[0] += 86_401
        assert store.claim_derivation(("facts",)) is None
        assert store.count() == 0
        with closing(sqlite3.connect(path)) as probe:
            assert probe.execute(
                "SELECT COUNT(*) FROM episode_derivation_jobs"
            ).fetchone() == (0,)
