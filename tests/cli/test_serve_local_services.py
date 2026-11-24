from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from openjarvis.cli import _should_skip_update_check
from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import EngineRegistry
from openjarvis.server import daemon

serve_module = importlib.import_module("openjarvis.cli.serve")


def test_local_services_skips_inference_runtime_and_uses_loopback(monkeypatch) -> None:
    config = JarvisConfig()
    config.server.host = "0.0.0.0"
    config.server.port = 8123
    monkeypatch.setattr(serve_module, "load_config", lambda: config)
    monkeypatch.setattr(serve_module, "_load_server_api_key", lambda: "")
    discovery = Mock(side_effect=AssertionError("unexpected inference discovery"))
    monkeypatch.setattr(serve_module, "get_configured_engine", discovery)
    monkeypatch.setattr(serve_module, "discover_engines", discovery)
    monkeypatch.setattr(serve_module, "inject_credentials", discovery)
    run = Mock()
    monkeypatch.setattr(daemon, "run_server", run)

    result = CliRunner().invoke(serve_module.serve, ["--local-services-only"])

    assert result.exit_code == 0, result.output
    assert run.call_args.args[0].state.config is config
    assert run.call_args.kwargs == {
        "host": "127.0.0.1",
        "port": 8123,
        "log_level": "info",
    }
    discovery.assert_not_called()


@pytest.mark.parametrize(
    "extra", [("--host", "0.0.0.0"), ("--host", "example.com"), ("--model", "m")]
)
def test_local_services_rejects_other_bind_addresses_and_inference_args(extra) -> None:
    result = CliRunner().invoke(serve_module.serve, ["--local-services-only", *extra])
    assert result.exit_code == 2
    assert "--local-services-only" in result.output


def test_only_local_services_serve_skips_the_cli_update_check() -> None:
    ctx = SimpleNamespace(invoked_subcommand="serve")
    assert _should_skip_update_check(ctx, ["serve", "--local-services-only"])
    assert _should_skip_update_check(ctx, ["serve", "--local-inference-only"])
    assert _should_skip_update_check(
        ctx, ["serve", "--local-engine-host=http://localhost"]
    )
    assert not _should_skip_update_check(ctx, ["serve", "--port", "8123"])


def test_local_inference_only_blocks_remote_ollama_before_health(monkeypatch) -> None:
    config = JarvisConfig()
    config.intelligence.provider = "openai"
    config.engine.ollama_host = "http://remote.invalid:11434"
    config.engine.auto_start_local = False
    config.telemetry.enabled = False
    health = Mock(side_effect=AssertionError("remote endpoint was probed"))
    close = Mock()

    class RemoteOllama:
        def __init__(self, host: str, **kwargs) -> None:
            self._host = host
            self.health = health
            self.close = close

    EngineRegistry.register_value("ollama", RemoteOllama)
    monkeypatch.setattr(serve_module, "load_config", lambda: config)
    monkeypatch.setattr(serve_module, "inject_credentials", Mock())
    run = Mock()
    monkeypatch.setattr(daemon, "run_server", run)

    result = CliRunner().invoke(
        serve_module.serve, ["--local-inference-only", "--engine", "ollama"]
    )

    assert result.exit_code == 1
    assert "No inference engine available" in result.output
    health.assert_not_called()
    close.assert_called_once_with()
    run.assert_not_called()
    assert config.intelligence.provider == "openai"


@pytest.mark.parametrize(
    ("engine_key", "host", "expected"),
    [
        ("ollama", "http://127.0.0.1:11434/", "http://127.0.0.1:11434"),
        ("lmstudio", "https://[::1]:1234/local", "https://[::1]:1234/local"),
    ],
)
def test_local_engine_host_reaches_the_selected_http_engine_without_editing_config(
    monkeypatch, engine_key, host, expected
) -> None:
    config = JarvisConfig()
    config.intelligence.provider = "openai"
    config.telemetry.enabled = False
    config.engine.auto_start_local = False
    saved_host = "http://remote.invalid:9000"
    settings = getattr(config.engine, engine_key)
    settings.host = saved_host
    make_engine = Mock()

    class TestHttpEngine:
        def __init__(self, host: str, **kwargs) -> None:
            self._host = host
            make_engine(host)

        def health(self) -> bool:
            return False

        def close(self) -> None:
            pass

    EngineRegistry.register_value(engine_key, TestHttpEngine)
    monkeypatch.setattr(serve_module, "load_config", lambda: config)
    monkeypatch.setattr(serve_module, "inject_credentials", Mock())

    result = CliRunner().invoke(
        serve_module.serve,
        ["--local-inference-only", "--engine", engine_key, "--local-engine-host", host],
    )

    assert result.exit_code == 1, result.output
    assert "No inference engine available" in result.output
    make_engine.assert_called_once_with(expected)
    assert config.intelligence.provider == "openai"
    assert settings.host == saved_host


@pytest.mark.parametrize(
    "arguments",
    [
        ["--engine", "ollama"],
        ["--local-inference-only"],
        ["--local-inference-only", "--engine", "gemma_cpp"],
        ["--local-inference-only", "--engine", "litellm"],
        ["--local-inference-only", "--engine", "nim"],
        ["--local-services-only", "--local-inference-only", "--engine", "ollama"],
    ],
)
def test_local_engine_host_requires_a_configurable_engine_and_local_inference(
    arguments, monkeypatch
) -> None:
    config_loader = Mock()
    monkeypatch.setattr(serve_module, "load_config", config_loader)
    result = CliRunner().invoke(
        serve_module.serve,
        [*arguments, "--local-engine-host", "http://localhost:11434"],
    )
    assert result.exit_code == 2, result.output
    config_loader.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "https://external.invalid:11434",
        "http://127.0.0.1.external.invalid",
        "http://user:secret@127.0.0.1:11434",
        "http://@localhost:11434",
        "http://localhost?",
        "http://localhost#fragment",
        "http://localhost:65536",
        "http://localhost:0",
        "http://localhost:",
        "http://[::1]invalid",
        "localhost:11434",
        "file://localhost/share",
        "http://local\\host:11434",
        "http://local\thost:11434",
    ],
)
def test_local_engine_host_rejects_non_loopback_or_ambiguous_urls(
    url, monkeypatch
) -> None:
    config_loader = Mock()
    monkeypatch.setattr(serve_module, "load_config", config_loader)
    result = CliRunner().invoke(
        serve_module.serve,
        ["--local-inference-only", "--engine", "ollama", "--local-engine-host", url],
    )
    assert result.exit_code == 2, result.output
    assert "--local-engine-host" in result.output
    assert "secret" not in result.output
    config_loader.assert_not_called()
