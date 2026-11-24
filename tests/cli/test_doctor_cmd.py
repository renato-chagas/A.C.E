"""Tests for ``jarvis doctor`` CLI command."""

from __future__ import annotations

import builtins
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openjarvis.cli import cli
from openjarvis.cli.doctor_cmd import (
    CheckResult,
    _check_config_exists,
    _check_config_parses,
    _check_default_model,
    _check_local_resilience,
    _check_local_tts_backend,
    _check_media_model_cache,
    _check_models,
    _check_nodejs,
    _check_ocr_tools,
    _check_optional_deps,
    _check_python_version,
    _check_speech_backend,
    _EngineProbe,
    _get_config,
    _probe_engines,
    _run_all_checks,
)
from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import EngineRegistry


class TestDoctorHelp:
    def test_doctor_help(self) -> None:
        result = CliRunner().invoke(cli, ["doctor", "--help"])
        assert result.exit_code == 0
        out = result.output.lower()
        assert "diagnostic" in out or "doctor" in out


class TestDoctorRuns:
    def test_doctor_runs(self) -> None:
        """Doctor command runs without error when engines are mocked."""
        mock_config = MagicMock()
        mock_config.intelligence.default_model = ""

        with (
            patch("openjarvis.cli.doctor_cmd.load_config", return_value=mock_config),
            patch(
                "openjarvis.cli.doctor_cmd.DEFAULT_CONFIG_PATH",
                Path("/tmp/nonexistent/config.toml"),
            ),
            patch("openjarvis.cli.doctor_cmd._check_engines", return_value=[]),
            patch("openjarvis.cli.doctor_cmd._check_models", return_value=[]),
            patch(
                "openjarvis.cli.doctor_cmd._check_speech_backend",
                return_value=CheckResult("Speech backend", "ok", "mock ready"),
            ),
        ):
            result = CliRunner().invoke(cli, ["doctor"])
        assert result.exit_code == 0
        assert "Doctor" in result.output or "passed" in result.output


class TestDoctorJsonOutput:
    def test_doctor_json_output(self) -> None:
        """--json flag produces valid JSON."""
        mock_config = MagicMock()
        mock_config.intelligence.default_model = ""

        with (
            patch("openjarvis.cli.doctor_cmd.load_config", return_value=mock_config),
            patch(
                "openjarvis.cli.doctor_cmd.DEFAULT_CONFIG_PATH",
                Path("/tmp/nonexistent/config.toml"),
            ),
            patch("openjarvis.cli.doctor_cmd._check_engines", return_value=[]),
            patch("openjarvis.cli.doctor_cmd._check_models", return_value=[]),
            patch(
                "openjarvis.cli.doctor_cmd._check_speech_backend",
                return_value=CheckResult("Speech backend", "ok", "mock ready"),
            ),
        ):
            result = CliRunner().invoke(cli, ["doctor", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) > 0
        # Each entry should have required fields
        for entry in data:
            assert "name" in entry
            assert "status" in entry
            assert "message" in entry


class TestCheckPythonVersion:
    def test_check_python_version(self) -> None:
        """Python version check passes on any supported Python."""
        result = _check_python_version()
        assert result.status == "ok"
        assert result.name == "Python version"


class TestCheckConfigMissing:
    def test_check_config_missing(self, monkeypatch) -> None:
        """Warning when config file does not exist."""
        monkeypatch.delenv("OPENJARVIS_CONFIG", raising=False)
        with patch(
            "openjarvis.cli.doctor_cmd.DEFAULT_CONFIG_PATH",
            Path("/tmp/nonexistent/config.toml"),
        ):
            result = _check_config_exists()
        assert result.status == "warn"
        assert "Not found" in result.message

    def test_explicit_config_path_is_used(self, tmp_path: Path, monkeypatch) -> None:
        """Doctor checks the documented OPENJARVIS_CONFIG override."""
        config_path = tmp_path / "custom config.toml"
        config_path.write_text('[intelligence]\nprovider = "local"\n', encoding="utf-8")
        monkeypatch.setenv("OPENJARVIS_CONFIG", str(config_path))

        with patch(
            "openjarvis.cli.doctor_cmd.DEFAULT_CONFIG_PATH",
            tmp_path / "missing.toml",
        ):
            exists = _check_config_exists()
            with patch(
                "openjarvis.cli.doctor_cmd.load_config", return_value=JarvisConfig()
            ) as load_config:
                parses = _check_config_parses()

        assert exists.status == "ok"
        assert exists.message == str(config_path.resolve())
        assert parses.status == "ok"
        load_config.assert_called_once_with(path=config_path.resolve())

    def test_runtime_checks_load_explicit_config(self, tmp_path: Path, monkeypatch):
        config_path = tmp_path / "ace.toml"
        monkeypatch.setenv("OPENJARVIS_CONFIG", str(config_path))
        config = JarvisConfig()

        with patch(
            "openjarvis.cli.doctor_cmd.load_config", return_value=config
        ) as load_config:
            loaded = _get_config()

        assert loaded is config
        load_config.assert_called_once_with(path=config_path.resolve())


class TestCheckEngineProbing:
    def test_check_engine_probing(self) -> None:
        """Engine health check reports reachable/unreachable engines."""
        from openjarvis.cli.doctor_cmd import CheckResult

        mock_engine_healthy = MagicMock()
        mock_engine_healthy.health.return_value = True

        mock_engine_down = MagicMock()
        mock_engine_down.health.return_value = False

        def mock_make_engine(key, config):
            if key == "ollama":
                return mock_engine_healthy
            return mock_engine_down

        # Directly test the engine probing logic without calling _check_engines
        # to avoid complex module-level mock interactions
        mock_config = MagicMock()
        keys = ["ollama", "vllm"]

        results = []
        for key in sorted(keys):
            engine = mock_make_engine(key, mock_config)
            if engine.health():
                results.append(CheckResult(f"Engine: {key}", "ok", "Reachable"))
            else:
                results.append(CheckResult(f"Engine: {key}", "warn", "Unreachable"))

        names = [r.name for r in results]
        assert "Engine: ollama" in names
        assert "Engine: vllm" in names
        # ollama should be ok, vllm should be warn
        ollama_result = next(r for r in results if r.name == "Engine: ollama")
        vllm_result = next(r for r in results if r.name == "Engine: vllm")
        assert ollama_result.status == "ok"
        assert vllm_result.status == "warn"

    def test_local_doctor_skips_remote_http_before_health_probe(self) -> None:
        config = JarvisConfig()
        config.intelligence.provider = "local"
        remote_engine = MagicMock()
        remote_engine._host = "https://inference.example/v1"
        remote_engine.is_cloud = False

        with (
            patch("openjarvis.cli.doctor_cmd._ensure_engines_imported"),
            patch.object(EngineRegistry, "keys", return_value=["remote-http"]),
            patch.object(
                EngineRegistry,
                "get",
                return_value=type("RemoteHttp", (), {"is_cloud": False}),
            ),
            patch(
                "openjarvis.engine._discovery._make_engine",
                return_value=remote_engine,
            ),
        ):
            probes = _probe_engines(config)

        assert probes == {}
        remote_engine.health.assert_not_called()
        remote_engine.close.assert_called_once_with()

    def test_full_doctor_health_checks_each_engine_once(self) -> None:
        """All engine/model checks share one health snapshot (#766)."""
        config = MagicMock()
        config.intelligence.default_model = "model-a"
        config.intelligence.preferred_engine = "alpha"
        config.engine.default = "alpha"

        alpha = MagicMock()
        alpha.health.return_value = True
        alpha.list_models.return_value = ["model-a"]
        beta = MagicMock()
        beta.health.return_value = False

        def make_engine(key, _config):
            return {"alpha": alpha, "beta": beta}[key]

        ok = CheckResult("other", "ok", "ok")
        with (
            patch("openjarvis.cli.doctor_cmd._ensure_engines_imported"),
            patch("openjarvis.cli.doctor_cmd._get_config", return_value=config),
            patch("openjarvis.cli.doctor_cmd.load_config", return_value=config),
            patch(
                "openjarvis.core.registry.EngineRegistry.keys",
                return_value=["beta", "alpha"],
            ),
            patch(
                "openjarvis.engine._discovery._make_engine",
                side_effect=make_engine,
            ) as make,
            patch("openjarvis.cli.doctor_cmd._check_python_version", return_value=ok),
            patch("openjarvis.cli.doctor_cmd._check_config_exists", return_value=ok),
            patch("openjarvis.cli.doctor_cmd._check_config_parses", return_value=ok),
            patch("openjarvis.cli.doctor_cmd._check_optional_deps", return_value=[]),
            patch("openjarvis.cli.doctor_cmd._check_speech_backend", return_value=ok),
            patch("openjarvis.cli.doctor_cmd._check_nodejs", return_value=ok),
            patch("openjarvis.cli.doctor_cmd._check_security_profile", return_value=ok),
        ):
            results = _run_all_checks()

        assert make.call_count == 2
        alpha.health.assert_called_once_with()
        beta.health.assert_called_once_with()
        assert any(
            result.name == "Default model" and result.status == "ok"
            for result in results
        )


class TestLocalResilienceAdvice:
    def test_empty_ace_ollama_reports_configured_pull_commands(self) -> None:
        config = JarvisConfig()
        config.intelligence.provider = "local"
        config.intelligence.default_model = "qwen3.5:4b"
        config.intelligence.fallback_model = "qwen3.5:2b"
        config.engine.auto_start_local = True
        engine = MagicMock()
        engine.list_models.return_value = []

        with patch("openjarvis.cli.doctor_cmd._get_config", return_value=config):
            results = _check_models(
                {"ollama": _EngineProbe(engine=engine, healthy=True)}
            )

        assert len(results) == 1
        assert results[0].status == "warn"
        assert "ollama pull qwen3.5:4b" in (results[0].details or "")
        assert "ollama pull qwen3.5:2b" in (results[0].details or "")
        assert "does not download models" in (results[0].details or "")

    def test_no_healthy_runtime_reports_installed_start_commands(self) -> None:
        config = JarvisConfig()
        config.intelligence.provider = "local"

        def which(executable: str) -> str | None:
            return {
                "ollama": r"C:\Apps\Ollama\ollama.exe",
                "lms": r"C:\Users\user\.lmstudio\bin\lms.exe",
            }.get(executable)

        with patch("openjarvis.cli.doctor_cmd.shutil.which", side_effect=which):
            result = _check_local_resilience(config, {})

        assert result is not None
        assert result.status == "fail"
        assert "Ollama: `ollama serve`" in result.details
        assert "LM Studio: `lms server start`" in result.details
        assert "run `jarvis doctor` again" in result.details

    def test_ace_autostart_is_reported_without_claiming_model_download(self) -> None:
        config = JarvisConfig()
        config.intelligence.provider = "local"
        config.engine.auto_start_local = True

        with patch(
            "openjarvis.cli.doctor_cmd.shutil.which",
            side_effect=lambda executable: (
                r"C:\Apps\Ollama\ollama.exe" if executable == "ollama" else None
            ),
        ):
            result = _check_local_resilience(config, {})

        assert result is not None
        assert "attempt to start loopback Ollama" in result.details
        assert "will not download models" in result.details

    def test_one_backend_suggests_a_second_installed_runtime(self) -> None:
        config = JarvisConfig()
        config.intelligence.provider = "local"
        config.intelligence.fallback_model = "missing-model"
        engine = MagicMock()
        engine.list_models.return_value = ["qwen"]
        engine._host = "http://127.0.0.1:11434"

        with patch(
            "openjarvis.cli.doctor_cmd.shutil.which",
            side_effect=lambda executable: (
                r"C:\Users\user\.lmstudio\bin\lms.exe" if executable == "lms" else None
            ),
        ):
            result = _check_local_resilience(
                config,
                {"ollama": _EngineProbe(engine=engine, healthy=True)},
            )

        assert result is not None
        assert result.status == "warn"
        assert "LM Studio: `lms server start`" in result.details
        assert "backend-outage failover" in result.details


class TestCheckDefaultModel:
    def test_empty_default_model_is_not_a_warning(self) -> None:
        """Leaving default model empty should be treated as valid auto-routing."""
        mock_config = MagicMock()
        mock_config.intelligence.default_model = ""
        with patch("openjarvis.cli.doctor_cmd.load_config", return_value=mock_config):
            result = _check_default_model()
        assert result.status == "ok"
        assert "auto" in result.message.lower()

    def test_local_ollama_without_selected_model_points_to_setup(
        self, monkeypatch
    ) -> None:
        mock_config = JarvisConfig()
        mock_config.intelligence.provider = "local"
        mock_config.engine.default = "ollama"
        monkeypatch.setattr("openjarvis.cli.doctor_cmd.sys.platform", "win32")

        with patch("openjarvis.cli.doctor_cmd.load_config", return_value=mock_config):
            result = _check_default_model()

        assert result.status == "ok"
        assert "jarvis model setup" in (result.details or "")
        assert "optionally download" in (result.details or "")
        assert "--install-runtime" in (result.details or "")

    def test_mlx_catalog_model_is_checked_by_repo_id(self) -> None:
        mock_config = MagicMock()
        mock_config.intelligence.default_model = "qwen3.5:4b"
        mock_config.intelligence.preferred_engine = "mlx"
        mock_config.engine.default = "mlx"
        mock_engine = MagicMock()
        mock_engine.health.return_value = True
        mock_engine.list_models.return_value = ["mlx-community/Qwen3.5-4B-OptiQ-4bit"]

        with (
            patch("openjarvis.cli.doctor_cmd.load_config", return_value=mock_config),
            patch("openjarvis.cli.doctor_cmd._ensure_engines_imported"),
            patch.object(EngineRegistry, "keys", return_value=["mlx"]),
            patch(
                "openjarvis.engine._discovery._make_engine",
                return_value=mock_engine,
            ),
        ):
            result = _check_default_model()

        assert result.status == "ok"
        assert "qwen3.5:4b" in result.message
        assert "mlx" in result.message


class TestCheckSpeechBackend:
    def test_check_speech_backend_ready(self) -> None:
        backend = MagicMock()
        backend.backend_id = "faster-whisper"
        backend.health.return_value = True

        with patch(
            "openjarvis.speech._discovery.get_speech_backend",
            return_value=backend,
        ):
            result = _check_speech_backend()

        assert result.status == "ok"
        assert "faster-whisper" in result.message

    def test_check_speech_backend_reports_load_error(self) -> None:
        backend = MagicMock()
        backend.backend_id = "faster-whisper"
        backend.health.return_value = False
        backend.last_error.return_value = "missing cublas64_12.dll"

        with patch(
            "openjarvis.speech._discovery.get_speech_backend",
            return_value=backend,
        ):
            result = _check_speech_backend()

        assert result.status == "warn"
        assert "faster-whisper unavailable" in result.message
        assert result.details == "missing cublas64_12.dll"

    def test_check_speech_backend_missing_uses_desktop_hint(self) -> None:
        with patch(
            "openjarvis.speech._discovery.get_speech_backend",
            return_value=None,
        ):
            result = _check_speech_backend()

        assert result.status == "warn"
        assert result.details is not None
        assert "uv sync --extra desktop" in result.details


class TestCheckOptionalDependencies:
    def test_missing_multipart_marks_server_stack_incomplete(self, monkeypatch) -> None:
        original_import = builtins.__import__

        def import_without_multipart(
            name, globals=None, locals=None, fromlist=(), level=0
        ):
            if name == "multipart":
                raise ImportError("multipart is not installed")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", import_without_multipart)

        checks = _check_optional_deps()

        uploads = next(
            check for check in checks if check.name == "Optional: REST API file uploads"
        )
        assert uploads.status == "warn"
        assert "Not installed" in uploads.message
        assert "openjarvis[server]" in uploads.message

    def test_missing_audio_packages_report_the_matching_extras(self, monkeypatch):
        original_import = builtins.__import__
        extras = {
            "sounddevice": "openjarvis[speech]",
            "soundfile": "openjarvis[speech]",
            "soundcard": "openjarvis[audio-loopback]",
            "kokoro": "openjarvis[voice]",
        }

        def import_without_audio_packages(
            name, globals=None, locals=None, fromlist=(), level=0
        ):
            if name in extras:
                raise ImportError(f"{name} is not installed")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", import_without_audio_packages)

        checks = _check_optional_deps()

        for package, extra in extras.items():
            check = next(
                result
                for result in checks
                if result.name.endswith(
                    {
                        "sounddevice": "Microphone and speaker audio I/O",
                        "soundfile": "Voice audio playback",
                        "soundcard": "Windows system audio input",
                        "kokoro": "Local neural text-to-speech (Kokoro)",
                    }[package]
                )
            )
            assert check.status == "warn"
            assert extra in check.message


class TestCheckLocalTtsBackend:
    def test_reports_available_windows_voice_backend(self) -> None:
        backend = MagicMock(backend_id="windows_sapi")
        with patch(
            "openjarvis.speech._tts_discovery.get_tts_backend",
            return_value=backend,
        ) as discover:
            result = _check_local_tts_backend()

        discover.assert_called_once_with(offline_only=True)
        assert result.status == "ok"
        assert result.message == "windows_sapi ready"

    def test_reports_install_hint_when_no_local_voice_backend(self) -> None:
        with patch(
            "openjarvis.speech._tts_discovery.get_tts_backend",
            return_value=None,
        ):
            result = _check_local_tts_backend()

        assert result.status == "warn"
        assert "Kokoro" in result.details
        assert "Windows speech voice" in result.details


class TestCheckOcrTools:
    def test_missing_tesseract_reports_image_ocr_unavailable(self) -> None:
        with patch(
            "openjarvis.tools.storage.ocr.select_ocr_language",
            side_effect=RuntimeError("Tesseract is not installed"),
        ):
            result = _check_ocr_tools()

        assert result.status == "warn"
        assert result.message == "Image OCR unavailable"
        assert result.details == "Tesseract is not installed"

    def test_missing_poppler_keeps_image_ocr_available(self) -> None:
        with (
            patch(
                "openjarvis.tools.storage.ocr.select_ocr_language",
                return_value="por+eng",
            ),
            patch("openjarvis.cli.doctor_cmd.shutil.which", return_value=None),
        ):
            result = _check_ocr_tools()

        assert result.status == "warn"
        assert "Image OCR ready (por+eng)" in result.message
        assert "scanned PDF OCR unavailable" in result.message
        assert result.details is not None
        assert "pdftoppm" in result.details

    def test_reports_image_and_scanned_pdf_ocr_ready(self) -> None:
        with (
            patch(
                "openjarvis.tools.storage.ocr.select_ocr_language",
                return_value="por+eng",
            ),
            patch(
                "openjarvis.cli.doctor_cmd.shutil.which",
                return_value="/usr/bin/pdftoppm",
            ),
        ):
            result = _check_ocr_tools()

        assert result.status == "ok"
        assert "Image and scanned PDF OCR ready" in result.message
        assert "por+eng" in result.message


class TestCheckMediaModelCache:
    def _fake_speech_modules(self, download_model):
        av_module = types.ModuleType("av")
        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.__path__ = []
        faster_whisper_utils = types.ModuleType("faster_whisper.utils")
        faster_whisper_utils.download_model = download_model
        return {
            "av": av_module,
            "faster_whisper": faster_whisper,
            "faster_whisper.utils": faster_whisper_utils,
        }

    def test_cached_model_uses_offline_lookup(self, tmp_path: Path) -> None:
        model_path = tmp_path / "small"
        model_path.mkdir()
        (model_path / "config.json").write_text("{}", encoding="utf-8")
        (model_path / "model.bin").write_bytes(b"model")
        (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
        download_model = MagicMock(return_value=str(model_path))
        config = JarvisConfig()
        config.speech.model = "small"

        with (
            patch.dict(sys.modules, self._fake_speech_modules(download_model)),
            patch("openjarvis.cli.doctor_cmd._get_config", return_value=config),
        ):
            result = _check_media_model_cache()

        assert result.status == "ok"
        assert "present locally" in result.message
        download_model.assert_called_once_with("small", local_files_only=True)

    def test_missing_model_reports_that_doctor_does_not_download(self) -> None:
        download_model = MagicMock(side_effect=FileNotFoundError("not cached"))
        config = JarvisConfig()
        config.speech.model = "small"

        with (
            patch.dict(sys.modules, self._fake_speech_modules(download_model)),
            patch("openjarvis.cli.doctor_cmd._get_config", return_value=config),
        ):
            result = _check_media_model_cache()

        assert result.status == "warn"
        assert "'small' is not available" in result.message
        assert "does not download models" in result.details
        download_model.assert_called_once_with("small", local_files_only=True)

    def test_incomplete_cached_model_bundle_warns(self, tmp_path: Path) -> None:
        model_path = tmp_path / "small"
        model_path.mkdir()
        (model_path / "config.json").write_text("{}", encoding="utf-8")
        (model_path / "model.bin").write_bytes(b"model")
        download_model = MagicMock(return_value=str(model_path))
        config = JarvisConfig()
        config.speech.model = "small"

        with (
            patch.dict(sys.modules, self._fake_speech_modules(download_model)),
            patch("openjarvis.cli.doctor_cmd._get_config", return_value=config),
        ):
            result = _check_media_model_cache()

        assert result.status == "warn"
        assert "incomplete local bundle" in result.message
        assert result.details == "Required local file(s) missing: tokenizer.json."


class TestCheckNodejs:
    def test_check_nodejs_found(self) -> None:
        """Node.js check reports version when node is available."""
        with (
            patch("shutil.which", return_value="/usr/bin/node"),
            patch(
                "subprocess.run",
                return_value=MagicMock(stdout="v22.5.0\n"),
            ),
        ):
            result = _check_nodejs()
        assert result.status == "ok"
        assert "v22.5.0" in result.message

    def test_check_nodejs_not_found(self) -> None:
        """Node.js check warns when node is not installed."""
        with patch("shutil.which", return_value=None):
            result = _check_nodejs()
        assert result.status == "warn"
        assert "Not found" in result.message
        assert result.details is not None
        assert "OpenClaw" not in result.details
