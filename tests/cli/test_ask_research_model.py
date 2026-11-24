"""Tests for the ``jarvis ask --research`` planner-model resolution."""

from __future__ import annotations

import importlib

import click
import pytest

from openjarvis.agents.research_loop import DEFAULT_PLANNER_MODEL
from openjarvis.cli.ask import _build_research_engine, _resolve_research_model
from openjarvis.core.config import JarvisConfig
from openjarvis.engine.ollama import OllamaEngine


def test_explicit_model_flag_wins() -> None:
    config = JarvisConfig()
    config.deep_research.model = "dr-model"
    config.intelligence.default_model = "cfg-model"
    assert _resolve_research_model("cli-model", config) == "cli-model"


def test_deep_research_section_beats_default_model() -> None:
    config = JarvisConfig()
    config.deep_research.model = "dr-model"
    config.intelligence.default_model = "cfg-model"
    assert _resolve_research_model(None, config) == "dr-model"


def test_config_default_model_used_when_no_override() -> None:
    """The bug: a configured default model must not be ignored for gemma4."""
    config = JarvisConfig()
    config.intelligence.default_model = "qwen3.5:2b"
    assert _resolve_research_model(None, config) == "qwen3.5:2b"


def test_legacy_fallback_when_nothing_configured() -> None:
    config = JarvisConfig()
    config.intelligence.default_model = ""
    assert _resolve_research_model(None, config) == DEFAULT_PLANNER_MODEL
    assert _resolve_research_model(None, None) == DEFAULT_PLANNER_MODEL


def test_local_profile_requires_a_selected_research_model() -> None:
    config = JarvisConfig()
    config.intelligence.provider = "local"
    config.intelligence.default_model = ""
    config.intelligence.fallback_model = ""

    assert _resolve_research_model(None, config) == ""
    assert _resolve_research_model("chosen-model", config) == "chosen-model"


def test_local_profile_accepts_explicit_research_fallback() -> None:
    config = JarvisConfig()
    config.intelligence.provider = "local"
    config.intelligence.default_model = ""
    config.intelligence.fallback_model = "user-configured-model"

    assert _resolve_research_model(None, config) == "user-configured-model"


def test_local_research_rejects_remote_active_engine() -> None:
    config = JarvisConfig()
    config.intelligence.provider = "local"
    engine = OllamaEngine(host="https://remote.example:11434")

    try:
        with pytest.raises(click.ClickException, match="local-only mode"):
            _build_research_engine(
                config,
                active_engine=engine,
                active_engine_key="ollama",
                model_name="chosen-model",
            )
    finally:
        engine.close()


def test_local_research_disables_environment_proxies() -> None:
    config = JarvisConfig()
    config.intelligence.provider = "local"
    config.engine.ollama_host = "http://127.0.0.1:11434"

    engine = OllamaEngine(host="http://127.0.0.1:11434", trust_env=False)
    resolved, local_only = _build_research_engine(
        config,
        active_engine=engine,
        active_engine_key="ollama",
        model_name="chosen-model",
    )
    try:
        assert local_only is True
        assert resolved is engine
        assert resolved._trust_env is False
        assert resolved._client._trust_env is False
    finally:
        resolved.close()


def test_explicit_research_engine_overrides_active_route(monkeypatch) -> None:
    config = JarvisConfig()
    config.deep_research.engine = "lmstudio"

    class SelectedEngine:
        engine_id = "lmstudio"

        def can_serve(self, model: str) -> bool:
            return model == "selected-model"

        def close(self) -> None:
            pass

    active_engine = object()
    selected_engine = SelectedEngine()
    calls: list[tuple[str | None, str | None]] = []

    def get_engine(_config, *, engine_key=None, model=None):
        calls.append((engine_key, model))
        return "lmstudio", selected_engine

    ask_module = importlib.import_module("openjarvis.cli.ask")
    monkeypatch.setattr(ask_module, "get_engine", get_engine)

    engine, local_only = _build_research_engine(
        config,
        active_engine=active_engine,
        active_engine_key="ollama",
        model_name="selected-model",
    )

    assert engine is selected_engine
    assert local_only is False
    assert calls == [("lmstudio", "selected-model")]
