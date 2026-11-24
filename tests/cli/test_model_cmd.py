"""Tests for ``jarvis model`` subcommands."""

from __future__ import annotations

import importlib
from unittest import mock

from click.testing import CliRunner

from openjarvis.cli import cli
from openjarvis.core.config import JarvisConfig

# Import the actual module (not the Click group attribute)
_model_mod = importlib.import_module("openjarvis.cli.model")


def _mock_engine():
    """Create a mock engine with list_models and health."""
    engine = mock.MagicMock()
    engine.engine_id = "mock"
    engine.health.return_value = True
    engine.list_models.return_value = ["model-a", "model-b"]
    return engine


class TestModelList:
    def test_list_from_mock_engine(self, monkeypatch) -> None:
        cfg = JarvisConfig()
        monkeypatch.setattr(_model_mod, "load_config", lambda: cfg)
        fake = _mock_engine()
        monkeypatch.setattr(
            _model_mod,
            "discover_engines",
            lambda c: [("mock", fake)],
        )
        monkeypatch.setattr(
            _model_mod,
            "discover_models",
            lambda e: {"mock": ["model-a", "model-b"]},
        )
        result = CliRunner().invoke(cli, ["model", "list"])
        assert result.exit_code == 0
        assert "model-a" in result.output

    def test_no_engines_message(self, monkeypatch) -> None:
        cfg = JarvisConfig()
        monkeypatch.setattr(_model_mod, "load_config", lambda: cfg)
        monkeypatch.setattr(_model_mod, "discover_engines", lambda c: [])
        result = CliRunner().invoke(cli, ["model", "list"])
        assert result.exit_code == 0
        assert "No inference engines" in result.output


class TestModelInfo:
    def test_info_known_model(self, monkeypatch) -> None:
        cfg = JarvisConfig()
        monkeypatch.setattr(_model_mod, "load_config", lambda: cfg)
        monkeypatch.setattr(_model_mod, "discover_engines", lambda c: [])
        monkeypatch.setattr(_model_mod, "discover_models", lambda e: {})
        result = CliRunner().invoke(cli, ["model", "info", "qwen3:8b"])
        assert result.exit_code == 0
        assert "Qwen3 8B" in result.output

    def test_unknown_model_not_found(self, monkeypatch) -> None:
        cfg = JarvisConfig()
        monkeypatch.setattr(_model_mod, "load_config", lambda: cfg)
        monkeypatch.setattr(_model_mod, "discover_engines", lambda c: [])
        monkeypatch.setattr(_model_mod, "discover_models", lambda e: {})
        result = CliRunner().invoke(cli, ["model", "info", "nonexistent-model"])
        assert result.exit_code != 0


class TestModelSetup:
    def _prepare(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[engine]\ndefault = "ollama"\nauto_start_local = true\n'
            '[engine.ollama]\nhost = "http://localhost:11434"\n'
            '[intelligence]\nprovider = "local"\ndefault_model = ""\n'
            'fallback_model = ""\n',
            encoding="utf-8",
        )
        config = JarvisConfig()
        config.intelligence.provider = "local"
        config.engine.auto_start_local = True
        monkeypatch.setenv("OPENJARVIS_CONFIG", str(config_path))
        monkeypatch.setattr(_model_mod, "load_config", lambda: config)
        return config_path

    def test_interactive_setup_lets_user_choose_and_save_without_download(
        self, tmp_path, monkeypatch
    ):
        config_path = self._prepare(tmp_path, monkeypatch)
        pull = mock.Mock()
        monkeypatch.setattr(_model_mod, "ollama_pull", pull)

        result = CliRunner().invoke(
            cli,
            ["model", "setup", "--no-install"],
            input="1\n",
        )

        assert result.exit_code == 0, result.output
        assert "Local model suggestions for Ollama" in result.output
        assert 'default_model = "qwen3:0.6b"' in config_path.read_text(encoding="utf-8")
        pull.assert_not_called()

    def test_setup_downloads_selected_model_and_then_sets_it_default(
        self, tmp_path, monkeypatch
    ):
        config_path = self._prepare(tmp_path, monkeypatch)
        monkeypatch.setattr(
            _model_mod, "_ensure_local_ollama", lambda *_args, **_kw: True
        )
        pull = mock.Mock(return_value=True)
        monkeypatch.setattr(_model_mod, "ollama_pull", pull)

        result = CliRunner().invoke(
            cli,
            ["model", "setup", "--model", "qwen3.5:4b", "--install"],
        )

        assert result.exit_code == 0, result.output
        pull.assert_called_once_with("http://localhost:11434", "qwen3.5:4b", mock.ANY)
        assert 'default_model = "qwen3.5:4b"' in config_path.read_text(encoding="utf-8")

    def test_setup_installs_missing_runtime_only_when_selected(
        self, tmp_path, monkeypatch
    ):
        config_path = self._prepare(tmp_path, monkeypatch)
        monkeypatch.setattr(
            _model_mod,
            "_ensure_local_ollama",
            mock.Mock(side_effect=[False, True]),
        )
        install_runtime = mock.Mock(return_value=True)
        monkeypatch.setattr(_model_mod, "_install_ollama_runtime", install_runtime)
        monkeypatch.setattr(
            _model_mod,
            "find_local_ollama_executable",
            mock.Mock(side_effect=[None, r"C:\Users\test\Ollama\ollama.exe"]),
        )
        pull = mock.Mock(return_value=True)
        monkeypatch.setattr(_model_mod, "ollama_pull", pull)

        result = CliRunner().invoke(
            cli,
            [
                "model",
                "setup",
                "--model",
                "qwen3.5:4b",
                "--install",
                "--install-runtime",
            ],
        )

        assert result.exit_code == 0, result.output
        install_runtime.assert_called_once()
        pull.assert_called_once_with("http://localhost:11434", "qwen3.5:4b", mock.ANY)
        assert 'default_model = "qwen3.5:4b"' in config_path.read_text(encoding="utf-8")

    def test_setup_does_not_install_runtime_when_explicitly_disabled(
        self, tmp_path, monkeypatch
    ):
        config_path = self._prepare(tmp_path, monkeypatch)
        monkeypatch.setattr(_model_mod, "find_local_ollama_executable", lambda: None)
        monkeypatch.setattr(
            _model_mod, "_ensure_local_ollama", lambda *_args, **_kwargs: False
        )
        install_runtime = mock.Mock()
        monkeypatch.setattr(_model_mod, "_install_ollama_runtime", install_runtime)

        result = CliRunner().invoke(
            cli,
            [
                "model",
                "setup",
                "--model",
                "qwen3.5:4b",
                "--install",
                "--no-install-runtime",
            ],
        )

        assert result.exit_code != 0
        install_runtime.assert_not_called()
        assert 'default_model = ""' in config_path.read_text(encoding="utf-8")

    def test_install_ollama_runtime_uses_winget_without_shell(self, monkeypatch):
        monkeypatch.setattr(_model_mod.sys, "platform", "win32")
        monkeypatch.setattr(
            _model_mod.shutil,
            "which",
            lambda name: (
                r"C:\Windows\System32\winget.exe" if name == "winget" else None
            ),
        )
        monkeypatch.setattr(
            _model_mod,
            "find_local_ollama_executable",
            lambda: r"C:\Users\test\Ollama\ollama.exe",
        )
        run = mock.Mock(return_value=mock.Mock(returncode=0))
        monkeypatch.setattr(_model_mod.subprocess, "run", run)

        assert _model_mod._install_ollama_runtime(_model_mod.Console()) is True

        command = run.call_args.args[0]
        assert command == [
            r"C:\Windows\System32\winget.exe",
            "install",
            "--exact",
            "--id",
            "Ollama.Ollama",
            "--source",
            "winget",
        ]
        assert "shell" not in run.call_args.kwargs
        assert run.call_args.kwargs["timeout"] == 900

    def test_failed_download_does_not_change_model_config(self, tmp_path, monkeypatch):
        config_path = self._prepare(tmp_path, monkeypatch)
        monkeypatch.setattr(
            _model_mod, "_ensure_local_ollama", lambda *_args, **_kw: True
        )
        monkeypatch.setattr(_model_mod, "ollama_pull", lambda *_args: False)

        result = CliRunner().invoke(
            cli,
            ["model", "setup", "--model", "qwen3.5:4b", "--install"],
        )

        assert result.exit_code != 0
        assert 'default_model = ""' in config_path.read_text(encoding="utf-8")

    def test_local_ollama_setup_refuses_non_loopback_hosts(self):
        assert not _model_mod._ensure_local_ollama(
            "http://192.168.1.10:11434",
            auto_start=True,
        )
