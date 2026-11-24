"""End-to-end tests for ``jarvis ask``."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from openjarvis.cli import cli
from openjarvis.core.config import JarvisConfig

# Import the actual module (not the Click command attribute)
_ask_mod = importlib.import_module("openjarvis.cli.ask")


def _mock_engine_response():
    """Return a mock engine that generates a fixed response."""
    return {
        "content": "The answer is 4.",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "model": "test-model",
        "finish_reason": "stop",
    }


def _patch_ask(monkeypatch, tmp_path, *, engine_result=None, no_engine=False):
    """Set up common mocks for ask tests."""
    # Re-register SimpleAgent after the autouse `_clean_registries` conftest
    # fixture clears it. ``JarvisConfig().agent.default_agent`` defaults to
    # ``"simple"``, so ``jarvis ask "..."`` (no --agent) routes through it.
    from openjarvis.agents.simple import SimpleAgent
    from openjarvis.core.registry import AgentRegistry

    if not AgentRegistry.contains("simple"):
        AgentRegistry.register_value("simple", SimpleAgent)

    cfg = JarvisConfig()
    cfg.telemetry.db_path = str(tmp_path / "telemetry.db")
    cfg.traces.enabled = False

    monkeypatch.setattr(_ask_mod, "load_config", lambda: cfg)

    if no_engine:
        monkeypatch.setattr(_ask_mod, "get_engine", lambda *a, **kw: None)
    else:
        fake_engine = mock.MagicMock()
        fake_engine.engine_id = "mock"
        fake_engine.health.return_value = True
        fake_engine.generate.return_value = engine_result or _mock_engine_response()
        fake_engine.list_models.return_value = ["test-model"]
        monkeypatch.setattr(
            _ask_mod,
            "get_engine",
            lambda *a, **kw: ("mock", fake_engine),
        )
        monkeypatch.setattr(
            _ask_mod,
            "discover_engines",
            lambda c: [("mock", fake_engine)],
        )
        monkeypatch.setattr(
            _ask_mod,
            "discover_models",
            lambda e: {"mock": ["test-model"]},
        )


class TestAskCommand:
    def test_basic_response(self, monkeypatch, tmp_path: Path) -> None:
        _patch_ask(monkeypatch, tmp_path)
        result = CliRunner().invoke(cli, ["ask", "What is 2+2?"])
        assert result.exit_code == 0
        assert "The answer is 4" in result.output

    def test_no_engine_error(self, monkeypatch, tmp_path: Path) -> None:
        _patch_ask(monkeypatch, tmp_path, no_engine=True)
        result = CliRunner().invoke(cli, ["ask", "Hello"])
        assert result.exit_code != 0

    def test_local_ollama_without_models_shows_configured_pull_commands(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        from openjarvis.agents.simple import SimpleAgent
        from openjarvis.core.registry import AgentRegistry

        if not AgentRegistry.contains("simple"):
            AgentRegistry.register_value("simple", SimpleAgent)

        engine = mock.MagicMock()
        engine.engine_id = "ollama"
        engine.health.return_value = True
        engine.list_models.return_value = []
        engine.can_serve.return_value = False
        config = JarvisConfig()
        config.telemetry.db_path = str(tmp_path / "telemetry.db")
        config.traces.enabled = False
        config.intelligence.provider = "local"
        config.intelligence.default_model = "qwen3.5:4b"
        config.intelligence.fallback_model = "qwen3.5:2b"
        config.engine.auto_start_local = True

        monkeypatch.setattr(_ask_mod, "load_config", lambda: config)
        monkeypatch.setattr(
            _ask_mod, "get_engine", lambda *args, **kwargs: ("ollama", engine)
        )
        monkeypatch.setattr(
            _ask_mod, "discover_engines", lambda _config: [("ollama", engine)]
        )
        monkeypatch.setattr(
            _ask_mod, "discover_models", lambda _engines: {"ollama": []}
        )

        result = CliRunner().invoke(cli, ["ask", "Hello"])

        assert result.exit_code == 1
        output = " ".join(result.output.split())
        assert "ollama pull qwen3.5:4b" in output, (
            f"output={result.output!r}; exception={result.exception!r}"
        )
        assert "ollama pull qwen3.5:2b" in output
        engine.generate.assert_not_called()

    def test_model_override(self, monkeypatch, tmp_path: Path) -> None:
        _patch_ask(monkeypatch, tmp_path)
        result = CliRunner().invoke(cli, ["ask", "-m", "custom-model", "Hello"])
        assert result.exit_code == 0

    def test_json_output(self, monkeypatch, tmp_path: Path) -> None:
        _patch_ask(monkeypatch, tmp_path)
        result = CliRunner().invoke(cli, ["ask", "--json", "Hello"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "content" in data

    def test_telemetry_recorded(self, monkeypatch, tmp_path: Path) -> None:
        _patch_ask(monkeypatch, tmp_path)
        CliRunner().invoke(cli, ["ask", "Hello"])
        db_path = tmp_path / "telemetry.db"
        assert db_path.exists()
