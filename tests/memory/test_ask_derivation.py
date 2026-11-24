"""Exercise one-shot local consolidation through the memory service."""

from __future__ import annotations

import json
import time
from contextlib import closing
from importlib import import_module
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from openjarvis.cli import cli
from openjarvis.core.config import JarvisConfig
from openjarvis.memory import episodes as episode_module
from openjarvis.memory.episodes import EpisodeStore
from openjarvis.memory.service import (
    build_memory_service,
    drain_ask_derivations,
    record_ask_episode,
)
from openjarvis.memory.store import LocalFactStore
from openjarvis.tools.storage.knowledge_graph import KnowledgeGraphMemory

_USER = "The Indigo project uses SQLite to keep a local audit log."
_ANSWER = "I noted where the project keeps the audit log."
_FACT = "The Indigo project keeps its audit log in SQLite."
_CLAIMS = {
    "entities": [
        {"name": "Indigo", "type": "project", "quote": "Indigo project"},
        {"name": "SQLite", "type": "software", "quote": "uses SQLite"},
    ],
    "relations": [
        {
            "subject": "Indigo",
            "predicate": "uses",
            "object": "SQLite",
            "quote": "Indigo project uses SQLite",
        }
    ],
}


class LocalExtractionEngine:
    def __init__(self):
        self.calls = []
        self.before_facts = lambda: None
        self.reject_facts_once = False

    def is_local_for_model(self, model):
        return model == "offline"

    def generate(self, messages, *, model, **kwargs):
        assert model == "offline"
        graph = "entities and relations" in messages[-1].content
        self.calls.append("graph" if graph else "facts")
        if graph:
            assert _ANSWER not in messages[-1].content
            return {"content": json.dumps(_CLAIMS)}
        self.before_facts()
        if self.reject_facts_once:
            self.reject_facts_once = False
            return {"content": "The model failed to provide JSON."}
        return {"content": json.dumps([_FACT])}


@pytest.fixture
def local_profile(tmp_path):
    config = JarvisConfig()
    config.memory.enabled = True
    config.memory.episodes_enabled = True
    config.memory.graph_capture_enabled = True
    config.memory.facts_path = str(tmp_path / "facts.jsonl")
    config.memory.episodes_path = str(tmp_path / "episodes.db")
    config.memory.graph_db_path = str(tmp_path / "graph.db")
    return config


@pytest.mark.parametrize("source", ["cli.ask", "sdk.ask"])
def test_derives_retained_ask_into_facts_and_graph_with_provenance(
    local_profile, source
):
    mem = local_profile.memory
    assert record_ask_episode(local_profile, _USER, _ANSWER, source=source)
    assert not LocalFactStore(mem.facts_path).list()
    with closing(EpisodeStore(mem.episodes_path)) as archive:
        (episode,) = archive.list()

    engine = LocalExtractionEngine()
    assert drain_ask_derivations(local_profile, engine, "offline", limit=10) == (2, 0)
    assert sorted(engine.calls) == ["facts", "graph"]
    (fact,) = LocalFactStore(mem.facts_path).list()
    assert fact.text == _FACT
    assert fact.source_ids == [episode.episode_id]
    with closing(KnowledgeGraphMemory(mem.graph_db_path)) as graph:
        (relation,) = graph.query_pattern(relation_type="uses").relations
        assert relation.properties["episode_ids"] == [episode.episode_id]
        assert (
            relation.evidence[0]["source"]
            == f"conversation:{source}:{episode.episode_id}"
        )
        assert relation.evidence[0]["quote"] == "Indigo project uses SQLite"

    assert drain_ask_derivations(local_profile, engine, "offline", limit=10) == (0, 0)
    assert sorted(engine.calls) == ["facts", "graph"]


def test_failure_retries_only_the_failed_kind(local_profile, monkeypatch):
    clock = [time.time()]
    monkeypatch.setattr(episode_module, "time", SimpleNamespace(time=lambda: clock[0]))
    assert record_ask_episode(local_profile, _USER, _ANSWER)
    engine = LocalExtractionEngine()
    engine.reject_facts_once = True
    assert drain_ask_derivations(local_profile, engine, "offline", limit=10) == (1, 1)
    assert drain_ask_derivations(local_profile, engine, "offline", limit=10) == (0, 0)
    clock[0] += 30
    assert drain_ask_derivations(local_profile, engine, "offline", limit=10) == (1, 0)
    assert sorted(engine.calls) == ["facts", "facts", "graph"]


def test_configured_remote_extraction_model_never_receives_episode(local_profile):
    assert record_ask_episode(local_profile, _USER, _ANSWER)
    engine = LocalExtractionEngine()
    local_profile.memory.extraction_model = "remote"
    assert drain_ask_derivations(local_profile, engine, "offline", limit=10) == (0, 0)
    assert engine.calls == []
    local_profile.memory.extraction_model = "offline"
    assert drain_ask_derivations(local_profile, engine, "remote", limit=10) == (2, 0)


def test_forget_during_inference_prevents_publication(local_profile):
    mem = local_profile.memory
    mem.graph_capture_enabled = False
    assert record_ask_episode(local_profile, _USER, _ANSWER)
    engine = LocalExtractionEngine()
    with closing(EpisodeStore(mem.episodes_path)) as archive:
        (episode,) = archive.list()
        engine.before_facts = lambda: archive.record_deletion_intent(episode.episode_id)
        assert drain_ask_derivations(local_profile, engine, "offline", limit=1) == (
            0,
            1,
        )
        assert archive.delete(episode.episode_id)
        archive.finish_deletion_intent(episode.episode_id)

    assert not LocalFactStore(mem.facts_path).list()
    assert drain_ask_derivations(local_profile, engine, "offline", limit=10) == (0, 0)
    assert engine.calls == ["facts"]


def test_background_worker_continues_one_shot_jobs(local_profile):
    mem = local_profile.memory
    mem.graph_capture_enabled = False
    assert record_ask_episode(local_profile, _USER, _ANSWER)
    engine = LocalExtractionEngine()
    worker = build_memory_service(local_profile, engine, "offline")
    assert worker is not None and worker._episode_store is not None
    worker.start()
    try:
        end = time.monotonic() + 5
        while not worker.list_facts() and time.monotonic() < end:
            time.sleep(0.02)
        assert [fact.text for fact in worker.list_facts()] == [_FACT]
    finally:
        worker.stop()
        worker._episode_store.close()
    assert engine.calls == ["facts"]


def test_manual_cli_drains_the_same_persistent_queue(local_profile, monkeypatch):
    assert record_ask_episode(local_profile, _USER, _ANSWER)
    cli_memory = import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(cli_memory, "load_config", lambda: local_profile)
    engine = LocalExtractionEngine()
    monkeypatch.setattr(
        "openjarvis.engine._discovery.get_configured_engine",
        lambda *_args, **_kwargs: ("ollama", engine),
    )

    run = CliRunner().invoke(
        cli, ["memory", "episodes", "derive", "--model", "offline", "--limit", "2"]
    )
    assert run.exit_code == 0, (run.output, run.exception)
    assert "Completed 2 task(s). Deferred 0 task(s)." in run.output
    assert [
        fact.text for fact in LocalFactStore(local_profile.memory.facts_path).list()
    ] == [_FACT]
    assert sorted(engine.calls) == ["facts", "graph"]
