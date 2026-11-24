"""Regression coverage for local one-shot episode capture."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from openjarvis.agents._stubs import AgentResult
from openjarvis.agents.orchestrator import OrchestratorAgent
from openjarvis.cli import cli
from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import ToolResult
from openjarvis.memory import service
from openjarvis.memory.episodes import EpisodeStore

_ASK = importlib.import_module("openjarvis.cli.ask")
_QUESTION = "Remember that the Indigo project uses a local SQLite audit log."
_ANSWER = "The project uses SQLite to retain the audit history locally."


class _Engine:
    engine_id = "mock"
    is_local = True
    finish_reason = "stop"

    def generate(self, messages, *, model, **kwargs):
        return {
            "content": _ANSWER,
            "model": model,
            "usage": {},
            "finish_reason": self.finish_reason,
        }


@pytest.fixture
def ask_env(monkeypatch, tmp_path):
    path = tmp_path / "episodes.db"
    config = JarvisConfig()
    config.agent.default_agent = "orchestrator"
    config.agent.context_from_memory = False
    config.telemetry.enabled = False
    config.traces.enabled = False
    config.memory.enabled = True
    config.memory.episodes_enabled = True
    config.memory.episodes_path = str(path)
    config.memory.episodes_semantic_enabled = True
    config.intelligence.default_model = "test-model"
    engine = _Engine()
    if not AgentRegistry.contains("orchestrator"):
        AgentRegistry.register_value("orchestrator", OrchestratorAgent)
    monkeypatch.setattr(_ASK, "load_config", lambda: config)
    monkeypatch.setattr(_ASK, "get_engine", lambda *args, **kw: ("mock", engine))
    monkeypatch.setattr(_ASK, "discover_engines", lambda _config: [("mock", engine)])
    monkeypatch.setattr(
        _ASK, "discover_models", lambda _engines: {"mock": ["test-model"]}
    )
    scanner = SimpleNamespace(scan=lambda _text: SimpleNamespace(is_clean=True))
    monkeypatch.setattr(service, "_build_scanner", lambda: scanner)
    monkeypatch.setattr(service, "_build_secret_scanner", lambda: scanner)
    return path, config, engine


@pytest.mark.parametrize("arguments", [[], ["--agent", "", "--json"]])
def test_plain_local_ask_stores_searchable_episode(ask_env, arguments):
    path, _, _ = ask_env
    result = CliRunner().invoke(cli, ["ask", *arguments, _QUESTION])
    assert result.exit_code == 0, result.exception
    assert _ANSWER in result.output

    episodes = EpisodeStore(path, semantic_enabled=False)
    try:
        (hit,) = episodes.search("Indigo")
        (episode,) = episodes.list()
        assert hit.metadata["episode_id"] == episode.episode_id
        assert (episode.user_text, episode.assistant_text, episode.source) == (
            _QUESTION,
            _ANSWER,
            "cli.ask",
        )
    finally:
        episodes.close()


@pytest.mark.parametrize(
    "excluded",
    [
        "no_context",
        "remote",
        "image",
        "screen",
        "screen_text",
        "tools",
        "incomplete_direct",
        "incomplete_agent",
        "memory_disabled",
        "episodes_disabled",
        "scanner_missing",
        "secret",
    ],
)
def test_ask_does_not_capture_ineligible_exchanges(
    ask_env,
    monkeypatch,
    tmp_path,
    excluded,
):
    path, config, engine = ask_env
    arguments = []
    if excluded == "no_context":
        arguments.append("--no-context")
        config.agent.context_from_memory = True
        monkeypatch.setattr(
            _ASK, "_get_memory_backend", lambda _cfg: pytest.fail("Memory was read")
        )
    elif excluded == "remote":
        engine.is_local = False
    elif excluded == "image":
        image = tmp_path / "sample.png"
        image.write_bytes(b"image")
        arguments.extend(["--image", str(image)])
    elif excluded in {"screen", "screen_text"}:
        arguments.extend(["--agent", "", "--" + excluded.replace("_", "-")])
        monkeypatch.setattr(
            "openjarvis.cli._screen.capture_screen_bytes", lambda: b"image"
        )
        monkeypatch.setattr(
            "openjarvis.tools.storage.ocr.screen_ocr_reader",
            lambda: lambda _image: "screen text",
        )
    elif excluded == "tools":
        monkeypatch.setattr(
            _ASK,
            "_run_agent",
            lambda *args, **kw: AgentResult(
                content=_ANSWER,
                tool_results=[ToolResult(tool_name="file_read", content="fixture")],
            ),
        )
    elif excluded.startswith("incomplete"):
        if excluded == "incomplete_direct":
            arguments.extend(["--agent", ""])
        engine.finish_reason = "length"
    elif excluded == "memory_disabled":
        config.memory.enabled = False
    elif excluded == "episodes_disabled":
        config.memory.episodes_enabled = False
    elif excluded == "scanner_missing":
        monkeypatch.setattr(service, "_build_scanner", lambda: None)
    elif excluded == "secret":
        scanner = SimpleNamespace(scan=lambda _text: SimpleNamespace(is_clean=False))
        monkeypatch.setattr(service, "_build_secret_scanner", lambda: scanner)

    result = CliRunner().invoke(cli, ["ask", *arguments, _QUESTION])
    assert result.exit_code == 0, result.exception
    assert _ANSWER in result.output
    assert not path.exists()
