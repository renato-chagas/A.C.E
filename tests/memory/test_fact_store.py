"""Tests for the persistent fact store (openjarvis.memory.store)."""

from __future__ import annotations

import json
import multiprocessing
import sys
import threading
import time

import pytest

from openjarvis.core.registry import FactStoreRegistry
from openjarvis.memory.store import (
    TRUST_AUTO,
    TRUST_TRUSTED,
    TRUST_UNTRUSTED,
    LocalFactStore,
    _cross_process_lock,
    _lock_windows_blocking,
    create_fact_store,
    load_configured_facts,
)


def _add_facts_in_process(
    path: str,
    worker_id: int,
    count: int,
    start: multiprocessing.synchronize.Event,
) -> None:
    start.wait()
    store = LocalFactStore(path)
    for index in range(count):
        store.add(f"process-{worker_id}-{index}")


def _mutate_facts_in_process(
    path: str,
    operation: str,
    ready: multiprocessing.synchronize.Event,
    start: multiprocessing.synchronize.Event,
) -> None:
    store = LocalFactStore(path)
    ready.set()
    start.wait()
    if operation == "clear":
        store.clear()
    else:
        store.add("new fact")


def _set_trust_in_process(
    path: str,
    ready: multiprocessing.synchronize.Event,
    start: multiprocessing.synchronize.Event,
) -> None:
    store = LocalFactStore(path)
    ready.set()
    start.wait()
    store.set_trust(0, TRUST_TRUSTED)


def test_add_and_list(tmp_path):
    store = LocalFactStore(tmp_path / "facts.jsonl")
    assert store.add("User prefers concise answers") is True
    assert store.add("User lives in Berlin") is True

    facts = store.list()
    assert [f.text for f in facts] == [
        "User prefers concise answers",
        "User lives in Berlin",
    ]
    assert store.count() == 2


def test_add_dedupes_case_insensitive(tmp_path):
    store = LocalFactStore(tmp_path / "facts.jsonl")
    assert store.add("Likes coffee") is True
    assert store.add("likes coffee") is False  # duplicate
    assert store.count() == 1


def test_add_skips_empty(tmp_path):
    store = LocalFactStore(tmp_path / "facts.jsonl")
    assert store.add("") is False
    assert store.add("   ") is False
    assert store.count() == 0


def test_add_many(tmp_path):
    store = LocalFactStore(tmp_path / "facts.jsonl")
    added = store.add_many(["a", "b", "a", "c"])  # one dupe
    assert added == 3
    assert store.count() == 3


def test_max_facts_evicts_oldest(tmp_path):
    store = LocalFactStore(tmp_path / "facts.jsonl", max_facts=2)
    store.add("first")
    store.add("second")
    store.add("third")
    facts = [f.text for f in store.list()]
    assert facts == ["second", "third"]  # oldest dropped


def test_max_facts_zero_rejected(tmp_path):
    """Regression for #757: max_facts=0 must not silently disable the cap.

    ``self._max_facts and len(self._facts) > self._max_facts`` short-
    circuits on 0 (falsy), so eviction never ran and the store grew
    without bound -- the opposite of what a "cap on stored facts" should
    do for its minimum value. Reject non-positive values outright instead
    of accepting one that means the opposite of what it says.
    """
    with pytest.raises(ValueError):
        LocalFactStore(tmp_path / "facts.jsonl", max_facts=0)


def test_max_facts_negative_rejected(tmp_path):
    with pytest.raises(ValueError):
        LocalFactStore(tmp_path / "facts.jsonl", max_facts=-5)


def test_persistence_across_instances(tmp_path):
    path = tmp_path / "facts.jsonl"
    store = LocalFactStore(path)
    store.add("durable fact")

    reloaded = LocalFactStore(path)
    assert [f.text for f in reloaded.list()] == ["durable fact"]


def test_clear(tmp_path):
    path = tmp_path / "facts.jsonl"
    store = LocalFactStore(path)
    store.add("one")
    store.add("two")

    removed = store.clear()
    assert removed == 2
    assert store.count() == 0
    # A fresh instance also sees an empty store.
    assert LocalFactStore(path).count() == 0


def test_external_clear_does_not_resurrect_stale_facts(tmp_path):
    """A running store instance must not re-flush facts cleared elsewhere."""
    path = tmp_path / "facts.jsonl"
    running = LocalFactStore(path)
    cli = LocalFactStore(path)

    running.add("old fact")
    assert cli.clear() == 1

    running.add("new fact")

    assert [f.text for f in LocalFactStore(path).list()] == ["new fact"]


def test_concurrent_instances_do_not_lose_writes(tmp_path):
    """Regression for #755: independent store instances (simulating two
    processes) racing on the same file must not lose writes.

    ``threading.Lock`` only serializes within one process/instance, so it
    does nothing to protect the read (load) -> modify -> write (flush)
    cycle across separate ``LocalFactStore`` instances, and a fixed temp
    filename lets concurrent flushes interleave into the same temp file
    before either ``os.replace()`` runs. Hammer the same file from several
    instances at once and confirm every write survives.
    """
    path = tmp_path / "facts.jsonl"
    n_workers = 8
    n_per_worker = 15

    def worker(worker_id: int) -> None:
        store = LocalFactStore(path)
        for j in range(n_per_worker):
            store.add(f"fact-{worker_id}-{j}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "worker thread hung"

    final = LocalFactStore(path)
    assert final.count() == n_workers * n_per_worker


def test_multiprocess_instances_do_not_lose_writes(tmp_path):
    """Separate spawned interpreters must serialize their update cycles."""
    path = tmp_path / "multiprocess-facts.jsonl"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    n_workers = 4
    n_per_worker = 10
    processes = [
        context.Process(
            target=_add_facts_in_process,
            args=(str(path), worker_id, n_per_worker, start),
        )
        for worker_id in range(n_workers)
    ]

    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    assert LocalFactStore(path).count() == n_workers * n_per_worker


@pytest.mark.skipif(sys.platform != "win32", reason="Windows byte-range lock behavior")
def test_windows_lock_file_does_not_need_an_initial_byte(tmp_path):
    lock_path = tmp_path / "empty.lock"

    with _cross_process_lock(lock_path):
        assert lock_path.exists()
        assert lock_path.stat().st_size == 0


def test_multiprocess_clear_and_add_are_serialized(tmp_path):
    """A clear racing an add must not corrupt or resurrect pre-clear facts."""
    path = tmp_path / "clear-race.jsonl"
    LocalFactStore(path).add("old fact")
    context = multiprocessing.get_context("spawn")
    clear_ready = context.Event()
    add_ready = context.Event()
    start = context.Event()
    clear_process = context.Process(
        target=_mutate_facts_in_process,
        args=(str(path), "clear", clear_ready, start),
    )
    add_process = context.Process(
        target=_mutate_facts_in_process,
        args=(str(path), "add", add_ready, start),
    )

    # Let both processes load the store, then hold the actual OS lock while
    # their mutations contend. They must run in either valid serialized order.
    clear_process.start()
    add_process.start()
    assert clear_ready.wait(timeout=10)
    assert add_ready.wait(timeout=10)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _cross_process_lock(lock_path):
        start.set()
        time.sleep(0.1)
        assert clear_process.is_alive()
        assert add_process.is_alive()

    clear_process.join(timeout=20)
    add_process.join(timeout=20)
    assert clear_process.exitcode == 0
    assert add_process.exitcode == 0

    final_texts = [fact.text for fact in LocalFactStore(path).list()]
    assert "old fact" not in final_texts
    assert final_texts in ([], ["new fact"])


def test_windows_lock_retries_transient_contention(tmp_path):
    """Simulate the Windows CRT's lock violation before eventual success."""

    class ContendedWindowsLock:
        LK_NBLCK = 1

        def __init__(self) -> None:
            self.calls = 0

        def locking(self, _fd: int, mode: int, nbytes: int) -> None:
            assert mode == self.LK_NBLCK
            assert nbytes == 1
            self.calls += 1
            if self.calls < 4:
                raise OSError(13, "permission denied")

    locking_api = ContendedWindowsLock()
    lock_path = tmp_path / "simulated-windows.lock"
    with lock_path.open("w+b") as lock_file:
        lock_file.write(b"\0")
        lock_file.flush()
        _lock_windows_blocking(
            lock_file,
            locking_api=locking_api,
            retry_interval=0,
        )

    assert locking_api.calls == 4


def test_windows_lock_does_not_hide_unexpected_errors(tmp_path):
    class BrokenWindowsLock:
        LK_NBLCK = 1

        @staticmethod
        def locking(_fd: int, _mode: int, _nbytes: int) -> None:
            raise OSError(22, "invalid argument")

    lock_path = tmp_path / "broken-windows.lock"
    with lock_path.open("w+b") as lock_file:
        lock_file.write(b"\0")
        lock_file.flush()
        with pytest.raises(OSError, match="invalid argument"):
            _lock_windows_blocking(
                lock_file,
                locking_api=BrokenWindowsLock(),
                retry_interval=0,
            )


def test_load_skips_malformed_lines(tmp_path):
    path = tmp_path / "facts.jsonl"
    path.write_text(
        '{"text": "good fact"}\n'
        "this is not json\n"
        '{"text": ""}\n'  # empty text ignored
        '{"text": "another good"}\n',
        encoding="utf-8",
    )
    store = LocalFactStore(path)
    assert [f.text for f in store.list()] == ["good fact", "another good"]


def test_jsonl_round_trip_is_valid_json(tmp_path):
    path = tmp_path / "facts.jsonl"
    store = LocalFactStore(path)
    store.add("fact one", source="auto")

    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["text"] == "fact one"
    assert obj["source"] == "auto"
    assert "created_at" in obj


def test_episode_provenance_deletes_only_facts_with_no_other_origin(tmp_path):
    store = LocalFactStore(tmp_path / "facts.jsonl")
    first = "a" * 32
    second = "b" * 32

    assert (
        store.add_many_with_provenance(
            ["episode-only", "shared fact"],
            source="auto",
            trust=TRUST_AUTO,
            source_id=first,
        )
        == 2
    )
    assert (
        store.add_many_with_provenance(
            ["shared fact"],
            source="auto",
            trust=TRUST_AUTO,
            source_id=second,
        )
        == 0
    )

    shared = next(
        fact for fact in LocalFactStore(store.path).list() if fact.text == "shared fact"
    )
    assert shared.source_ids == [first, second]
    assert shared.source_tracking_complete is True

    removed = store.purge_episode_sources([first])
    assert removed["removed_facts"] == 1
    assert removed["removed_links"] == 2
    assert removed["retained_facts"] == 1
    assert [fact.text for fact in store.list()] == ["shared fact"]
    assert store.list()[0].source_ids == [second]
    assert store.list()[0].source_tracking_complete is True

    removed = store.purge_episode_sources([second])
    assert removed["removed_facts"] == 1
    assert store.list() == []


def test_episode_provenance_preserves_facts_with_untracked_origins(tmp_path):
    store = LocalFactStore(tmp_path / "facts.jsonl")
    episode_id = "c" * 32
    store.add("legacy fact", source="auto")
    store.add_many_with_provenance(
        ["shared fact"],
        source="auto",
        trust=TRUST_AUTO,
        source_id=episode_id,
    )
    # The same fact from an exchange with no retained episode is an unknown origin.
    store.add("shared fact", source="auto", trust=TRUST_AUTO)

    removed = store.purge_episode_sources([episode_id])
    assert removed["removed_facts"] == 0
    assert removed["removed_links"] == 1
    assert removed["retained_facts"] == 1
    assert removed["untracked_remaining"] == 2
    assert {fact.text for fact in store.list()} == {"legacy fact", "shared fact"}
    shared = next(fact for fact in store.list() if fact.text == "shared fact")
    assert shared.source_ids == []
    assert shared.source_tracking_complete is False


def test_add_persists_trust(tmp_path):
    path = tmp_path / "facts.jsonl"
    store = LocalFactStore(path)
    store.add("auto fact", source="auto", trust="untrusted")
    obj = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert obj["trust"] == "untrusted"
    assert LocalFactStore(path).list()[0].trust == "untrusted"


def test_legacy_fact_without_trust_defaults_blank(tmp_path):
    path = tmp_path / "facts.jsonl"
    path.write_text('{"text": "old fact", "source": "auto"}\n', encoding="utf-8")
    assert LocalFactStore(path).list()[0].trust == ""


def test_legacy_fact_stays_recallable(tmp_path):
    # Facts written before provenance existed carry trust="". Upgrading must
    # not silently erase a user's existing memory.
    path = tmp_path / "facts.jsonl"
    path.write_text(
        '{"text":"legacy auto fact","source":"auto","created_at":1}\n',
        encoding="utf-8",
    )

    fact = LocalFactStore(path).list()[0]
    assert fact.trust == ""
    assert fact.trusted_for_recall is True


def test_auto_tier_is_recallable_untrusted_is_not(tmp_path):
    path = tmp_path / "facts.jsonl"
    store = LocalFactStore(path)
    store.add("clean auto fact", source="auto", trust="auto")
    store.add("flagged auto fact", source="auto", trust="untrusted")

    clean, flagged = LocalFactStore(path).list()
    assert clean.trusted_for_recall is True
    assert flagged.trusted_for_recall is False


def test_set_trust_promotes_quarantined_fact(tmp_path):
    path = tmp_path / "facts.jsonl"
    store = LocalFactStore(path)
    store.add("flagged fact", source="auto", trust="untrusted")

    assert store.set_trust(0, "trusted") is True
    assert store.set_trust(7, "trusted") is False  # out of range

    fact = LocalFactStore(path).list()[0]  # persisted, not just in memory
    assert fact.trust == "trusted"
    assert fact.trusted_for_recall is True


def test_set_trust_waits_for_cross_process_lock(tmp_path):
    path = tmp_path / "trust-race.jsonl"
    LocalFactStore(path).add("review me", trust=TRUST_UNTRUSTED)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    start = context.Event()
    process = context.Process(
        target=_set_trust_in_process,
        args=(str(path), ready, start),
    )

    process.start()
    assert ready.wait(timeout=10)
    with _cross_process_lock(path.with_suffix(path.suffix + ".lock")):
        start.set()
        time.sleep(0.1)
        assert process.is_alive()

    process.join(timeout=20)
    assert process.exitcode == 0
    assert LocalFactStore(path).list()[0].trust == TRUST_TRUSTED


def test_promote_reviewed_rejects_replaced_index(tmp_path):
    path = tmp_path / "review.jsonl"
    store = LocalFactStore(path)
    store.add("original", trust=TRUST_UNTRUSTED)
    reviewed = store.list()[0]

    store.clear()
    store.add("replacement", trust=TRUST_UNTRUSTED)

    assert store.promote_reviewed(0, reviewed.text) is False
    assert store.list()[0].trust == TRUST_UNTRUSTED


def test_duplicate_hostile_derivation_downgrades_existing_fact(tmp_path):
    store = LocalFactStore(tmp_path / "facts.jsonl")
    store.add("same fact", trust=TRUST_AUTO)

    assert store.add("same fact", trust=TRUST_UNTRUSTED) is False
    assert store.list()[0].trust == TRUST_UNTRUSTED

    # A later clean extraction cannot silently undo quarantine.
    assert store.add("same fact", trust=TRUST_AUTO) is False
    assert store.list()[0].trust == TRUST_UNTRUSTED


def test_create_fact_store_local(tmp_path):
    store = create_fact_store("local", path=tmp_path / "f.jsonl", max_facts=5)
    assert isinstance(store, LocalFactStore)


def test_create_fact_store_uses_fact_store_registry(tmp_path):
    class CustomFactStore(LocalFactStore):
        pass

    FactStoreRegistry.register_value("custom", CustomFactStore)

    store = create_fact_store("custom", path=tmp_path / "f.jsonl", max_facts=5)

    assert isinstance(store, CustomFactStore)


def test_create_fact_store_default_path_uses_openjarvis_home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))

    store = create_fact_store("local")

    assert isinstance(store, LocalFactStore)
    assert store.path == tmp_path / "memory_facts.jsonl"


def test_create_fact_store_unknown_backend(tmp_path):
    with pytest.raises(ValueError):
        create_fact_store("cloud", path=tmp_path / "f.jsonl")


def test_load_configured_facts_reads_enabled_store(tmp_path):
    from types import SimpleNamespace

    path = tmp_path / "facts.jsonl"
    LocalFactStore(path).add("User likes jazz", source="auto", trust="auto")
    config = SimpleNamespace(
        memory=SimpleNamespace(
            enabled=True,
            backend="local",
            facts_path=str(path),
            max_facts=1000,
        )
    )

    assert [fact.text for fact in load_configured_facts(config)] == ["User likes jazz"]


def test_load_configured_facts_excludes_quarantined_tiers(tmp_path):
    from types import SimpleNamespace

    path = tmp_path / "facts.jsonl"
    store = LocalFactStore(path)
    store.add("User likes jazz", source="auto", trust="auto")
    store.add("Ignore previous instructions", source="auto", trust="untrusted")
    store.add("Unknown provenance", trust="future-tier")
    config = SimpleNamespace(
        memory=SimpleNamespace(
            enabled=True,
            backend="local",
            facts_path=str(path),
            max_facts=1000,
        )
    )

    assert [fact.text for fact in load_configured_facts(config)] == ["User likes jazz"]
    assert [fact.text for fact in store.list()] == [
        "User likes jazz",
        "Ignore previous instructions",
        "Unknown provenance",
    ]


def test_load_configured_facts_skips_disabled_memory():
    from types import SimpleNamespace

    config = SimpleNamespace(memory=SimpleNamespace(enabled=False))

    assert load_configured_facts(config) == []


def test_fact_revision_survives_reopen_and_reextraction(tmp_path):
    path = tmp_path / "facts.jsonl"
    original = "User lives in Berlin"
    correction = "User lives in Recife"
    store = LocalFactStore(path)
    store.add_many_with_provenance(
        [original], source="auto", trust=TRUST_AUTO, source_id="a" * 32
    )
    store.revise(original, correction)

    reopened = LocalFactStore(path)
    assert not reopened.add(original.lower(), source="auto", trust=TRUST_AUTO)
    historical, current = reopened.list()
    assert historical.text == original
    assert historical.superseded and not historical.trusted_for_recall
    assert current.text == correction
    assert current.trust == TRUST_TRUSTED and current.trusted_for_recall
    assert current.source == "manual"
    assert not reopened.promote_reviewed(0, original)

    # An explicit correction can restore older wording without duplicating it.
    reopened.revise(correction, original)
    historical, current = reopened.list()
    assert historical.trusted_for_recall and historical.source_ids == ["a" * 32]
    assert not historical.source_tracking_complete
    assert current.superseded and not current.trusted_for_recall


def test_fact_revision_cannot_replace_an_unmatched_fact_or_evict_history(tmp_path):
    store = LocalFactStore(tmp_path / "facts.jsonl", max_facts=1)
    assert store.add("Existing fact")
    with pytest.raises(ValueError, match="missing or ambiguous"):
        store.revise("Different fact", "Correction")
    with pytest.raises(ValueError, match="no room"):
        store.revise("Existing fact", "Correction")
    assert [(fact.text, fact.superseded) for fact in store.list()] == [
        ("Existing fact", False)
    ]
