from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from openjarvis.cli import cli, memory_cmd
from openjarvis.core.config import JarvisConfig
from openjarvis.engine import _discovery
from openjarvis.memory import service


@pytest.mark.parametrize(
    ("model_arg", "extraction_model", "saved_model", "expected_model"),
    [
        ("chosen", "configured", "saved", "chosen"),
        (None, "configured", "saved", "configured"),
        (None, "", "saved", "saved"),
    ],
)
def test_derive_episodes_resolves_selected_model_through_a_local_profile(
    monkeypatch, model_arg, extraction_model, saved_model, expected_model
) -> None:
    config = JarvisConfig()
    config.intelligence.provider = "cloud"
    config.intelligence.default_model = saved_model
    config.intelligence.fallback_model = "unselected"
    config.engine.auto_start_local = True
    config.memory.extraction_model = extraction_model
    monkeypatch.setattr(memory_cmd, "load_config", lambda: config)
    engine = SimpleNamespace(is_local=True, close=Mock())
    resolver = Mock(return_value=("test-local", engine))
    drain = Mock(return_value=(3, 1))
    monkeypatch.setattr(_discovery, "get_configured_engine", resolver)
    monkeypatch.setattr(service, "drain_ask_derivations", drain)
    args = ["memory", "episodes", "derive", "--limit", "4", "--engine", "test-local"]
    if model_arg:
        args += ["--model", model_arg]

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "Completed 3 task(s). Deferred 1 task(s)."
    local_config = resolver.call_args.args[0]
    resolver.assert_called_once_with(
        local_config, engine_key="test-local", model=expected_model
    )
    assert local_config.intelligence.provider == "local"
    assert local_config.intelligence.fallback_model == ""
    assert local_config.engine.auto_start_local is False
    assert local_config.memory.extraction_model == expected_model
    assert config.intelligence.provider == "cloud"
    assert config.intelligence.fallback_model == "unselected"
    assert config.engine.auto_start_local is True
    assert config.memory.extraction_model == extraction_model
    drain.assert_called_once_with(local_config, engine, expected_model, limit=4)
    engine.close.assert_called_once_with()


def test_derive_episodes_requires_selected_model_before_discovery(monkeypatch) -> None:
    config = JarvisConfig()
    config.intelligence.fallback_model = "only-a-fallback"
    monkeypatch.setattr(memory_cmd, "load_config", lambda: config)
    resolver = Mock()
    drain = Mock()
    monkeypatch.setattr(_discovery, "get_configured_engine", resolver)
    monkeypatch.setattr(service, "drain_ask_derivations", drain)

    result = CliRunner().invoke(cli, ["memory", "episodes", "derive"])

    assert result.exit_code == 1
    assert "Choose a local model" in result.output
    resolver.assert_not_called()
    drain.assert_not_called()


def test_derive_episodes_checks_the_specific_model_route(monkeypatch) -> None:
    monkeypatch.setattr(memory_cmd, "load_config", JarvisConfig)
    engine = SimpleNamespace(
        is_local=True, is_local_for_model=Mock(return_value=False), close=Mock()
    )
    monkeypatch.setattr(
        _discovery, "get_configured_engine", Mock(return_value=("mixed", engine))
    )
    drain = Mock()
    monkeypatch.setattr(service, "drain_ask_derivations", drain)

    result = CliRunner().invoke(
        cli, ["memory", "episodes", "derive", "--model", "routed"]
    )

    assert result.exit_code == 1
    assert "requires a verified local route" in result.output
    engine.is_local_for_model.assert_called_once_with("routed")
    drain.assert_not_called()
    engine.close.assert_called_once_with()


def test_derive_episodes_does_not_print_private_failure_details(monkeypatch) -> None:
    monkeypatch.setattr(memory_cmd, "load_config", JarvisConfig)
    engine = SimpleNamespace(is_local=True, close=Mock())
    monkeypatch.setattr(
        _discovery, "get_configured_engine", Mock(return_value=("test-local", engine))
    )
    drain = Mock(side_effect=RuntimeError("private episode content"))
    monkeypatch.setattr(service, "drain_ask_derivations", drain)

    result = CliRunner().invoke(
        cli, ["memory", "episodes", "derive", "--model", "chosen"]
    )

    assert result.exit_code == 1
    assert "Could not process queued memory derivations." in result.output
    assert "private episode content" not in result.output
    assert drain.call_args.kwargs == {"limit": 5}
    engine.close.assert_called_once_with()
