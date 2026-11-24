"""Tests for local-only inference setup hints."""

import sys

from openjarvis.cli._local_setup_hint import local_model_setup_hint
from openjarvis.core.config import JarvisConfig


def test_ollama_hint_uses_configured_models_and_autostart_policy() -> None:
    config = JarvisConfig()
    config.intelligence.provider = "local"
    config.intelligence.default_model = "qwen3.5:4b"
    config.intelligence.fallback_model = "qwen3.5:2b"
    config.engine.auto_start_local = True

    hint = local_model_setup_hint(config)

    assert "start loopback Ollama automatically" in hint
    assert "ollama pull qwen3.5:4b" in hint
    assert "ollama pull qwen3.5:2b" in hint
    assert "does not download models" in hint
    assert "jarvis model setup" in hint


def test_windows_hint_exposes_explicit_ollama_runtime_install(monkeypatch) -> None:
    config = JarvisConfig()
    config.intelligence.provider = "local"
    config.engine.default = "ollama"
    monkeypatch.setattr(sys, "platform", "win32")

    assert "jarvis model setup --install-runtime" in local_model_setup_hint(config)


def test_explicit_model_overrides_server_and_default_models() -> None:
    config = JarvisConfig()
    config.intelligence.provider = "local"
    config.intelligence.default_model = "qwen3.5:4b"
    config.intelligence.fallback_model = "qwen3.5:2b"
    config.server.model = "server-model"

    hint = local_model_setup_hint(config, requested_model="cli-model")

    assert "ollama pull cli-model" in hint
    assert "ollama pull server-model" not in hint
    assert "ollama pull qwen3.5:4b" not in hint
    assert "ollama pull qwen3.5:2b" in hint


def test_chat_setup_hint_uses_chat_default_not_server_model() -> None:
    config = JarvisConfig()
    config.intelligence.provider = "local"
    config.intelligence.default_model = "qwen3.5:4b"
    config.server.model = "server-model"

    hint = local_model_setup_hint(config)

    assert "ollama pull qwen3.5:4b" in hint
    assert "ollama pull server-model" not in hint


def test_nonlocal_or_other_local_backend_does_not_suggest_ollama_pull() -> None:
    config = JarvisConfig()
    config.intelligence.provider = "cloud"
    assert local_model_setup_hint(config) == ""

    config.intelligence.provider = "local"
    config.engine.default = "vllm"
    hint = local_model_setup_hint(config)
    assert "jarvis doctor" in hint
    assert "ollama pull" not in hint


def test_unsafe_model_identifier_is_never_rendered_as_a_pull_command() -> None:
    config = JarvisConfig()
    config.intelligence.provider = "local"
    config.intelligence.default_model = "qwen3.5:4b; Remove-Item *"

    hint = local_model_setup_hint(config)

    assert "Remove-Item" not in hint
    assert "jarvis model setup" in hint
