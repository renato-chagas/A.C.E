"""SDK episode capture uses the shared local, scanner-gated archive."""

from __future__ import annotations

from contextlib import closing
from importlib import import_module
from unittest.mock import Mock

import pytest

from openjarvis.agents.orchestrator import OrchestratorAgent
from openjarvis.agents.simple import SimpleAgent
from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import AgentRegistry
from openjarvis.engine._local_failover import LocalFailoverEngine
from openjarvis.engine._stubs import StreamChunk
from openjarvis.memory.episodes import EpisodeStore
from openjarvis.memory.service import record_ask_episode
from openjarvis.sdk import Jarvis

_QUESTION = "Our decision was to put the Indigo project handouts in the blue binder."
_ANSWER = "The Indigo handouts are in the blue binder by the shared desk."


@pytest.fixture
def sdk_env(tmp_path, monkeypatch):
    path = tmp_path / "episodes.db"
    config = JarvisConfig()
    config.memory.enabled = True
    config.memory.episodes_enabled = True
    config.memory.episodes_path = str(path)
    config.agent.context_from_memory = False
    config.telemetry.enabled = False
    config.telemetry.gpu_metrics = False
    engine = Mock(
        engine_id="ollama",
        _host="http://127.0.0.1:11434",
        is_local_for_model=None,
        is_cloud=False,
    )
    engine.generate.return_value = {
        "content": _ANSWER,
        "usage": {},
        "finish_reason": "stop",
    }
    monkeypatch.setattr("openjarvis.sdk.get_engine", lambda *_: ("ollama", engine))
    for agent in (SimpleAgent, OrchestratorAgent):
        if not AgentRegistry.contains(agent.agent_id):
            AgentRegistry.register_value(agent.agent_id, agent)
    return path, config, engine


def assert_saved_episode(path):
    with closing(EpisodeStore(path)) as episodes:
        (episode,) = episodes.list()
        (hit,) = episodes.search("Indigo")
    assert hit.metadata["episode_id"] == episode.episode_id
    assert (episode.user_text, episode.assistant_text, episode.source) == (
        _QUESTION,
        _ANSWER,
        "sdk.ask",
    )


@pytest.mark.parametrize("agent", [None, "simple", "orchestrator"])
def test_completed_local_sdk_ask_is_searchable(sdk_env, agent):
    path, config, engine = sdk_env
    with Jarvis(config=config, model="offline") as jarvis:
        assert jarvis.ask(_QUESTION, agent=agent) == _ANSWER
    assert_saved_episode(path)
    engine.generate.assert_called_once()


def test_sdk_recalls_terminal_memory_from_the_same_profile(sdk_env, monkeypatch):
    _, config, engine = sdk_env
    assert record_ask_episode(config, _QUESTION, _ANSWER)
    config.agent.context_from_memory = True
    cli_ask = import_module("openjarvis.cli.ask")
    monkeypatch.setattr(cli_ask, "_get_memory_backend", lambda _: None)
    monkeypatch.setattr(cli_ask, "_get_memory_facts", lambda _: [])
    with Jarvis(config=config, model="offline") as jarvis:
        jarvis.ask("Where are the Indigo handouts?")

    messages = engine.generate.call_args.args[0]
    assert any(_ANSWER in message.content for message in messages)


@pytest.mark.parametrize(
    "reason",
    ["remote", "context_disabled", "episodes_off", "length", "tools", "secret"],
)
def test_sdk_ask_skips_private_or_incomplete_replies(sdk_env, reason):
    path, config, engine = sdk_env
    query = _QUESTION
    if reason == "remote":
        engine._host = "http://remote.invalid:11434"
    elif reason == "episodes_off":
        config.memory.episodes_enabled = False
    elif reason == "length":
        engine.generate.return_value["finish_reason"] = "length"
    elif reason == "tools":
        engine.generate.return_value["tool_calls"] = [{"name": "read_file"}]
    elif reason == "secret":
        query += " api_key = 'sk-test-AceEpisodeFixture0123456789'"

    with Jarvis(config=config, model="offline") as jarvis:
        result = jarvis.ask_full(query, context=reason != "context_disabled")
    assert result["content"] == _ANSWER
    assert not path.exists()


def test_incomplete_agent_reply_is_not_captured(sdk_env):
    path, config, engine = sdk_env
    engine.generate.return_value["finish_reason"] = "length"
    with Jarvis(config=config, model="offline") as jarvis:
        assert jarvis.ask(_QUESTION, agent="simple") == _ANSWER
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    ["completed", "no_terminal", "length", "synthetic_stop", "tools", "closed"],
)
async def test_sdk_stream_saves_only_verified_text_completion(
    sdk_env, state, monkeypatch
):
    path, config, engine = sdk_env
    closed = False

    async def stream(*_args, **_kwargs):
        nonlocal closed
        try:
            yield StreamChunk(content=_ANSWER)
            if state == "tools":
                yield StreamChunk(tool_calls=[{"name": "read_file"}])
            if state != "no_terminal":
                yield StreamChunk(
                    finish_reason="length" if state == "length" else "stop",
                    finish_reason_reported=state != "synthetic_stop",
                )
        finally:
            closed = True

    engine.stream_full = stream
    config.security.enabled = True
    config.security.secret_scanner = True
    route = LocalFailoverEngine(engine, [], primary_key="ollama")
    monkeypatch.setattr("openjarvis.sdk.get_engine", lambda *_: ("ollama", route))
    with Jarvis(config=config, model="offline") as jarvis:
        result = jarvis.ask_full_stream(_QUESTION)
        if state == "closed":
            assert await anext(result) == {"token": _ANSWER, "index": 0}
            await result.aclose()
        else:
            parts = [part async for part in result]
            assert parts[-1]["done"]
            assert parts[-1]["content"] == _ANSWER

    assert closed
    engine.generate.assert_not_called()
    if state == "completed":
        assert_saved_episode(path)
    else:
        assert not path.exists()
