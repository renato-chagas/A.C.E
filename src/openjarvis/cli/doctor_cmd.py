"""``jarvis doctor`` — run diagnostic checks on the OpenJarvis installation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.table import Table

from openjarvis.core.config import DEFAULT_CONFIG_PATH, load_config


@dataclass
class CheckResult:
    """Result of a single diagnostic check."""

    name: str
    status: str  # "ok", "warn", "fail"
    message: str
    details: Optional[str] = None


@dataclass
class _EngineProbe:
    """One engine instance and the result of its single health probe."""

    engine: Any | None
    healthy: bool
    error: str | None = None


# -- Individual checks -------------------------------------------------------


def _check_python_version() -> CheckResult:
    """Check that Python version is >= 3.10."""
    ver = sys.version_info
    version_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    if (ver.major, ver.minor) >= (3, 10):
        return CheckResult("Python version", "ok", version_str)
    return CheckResult("Python version", "fail", f"{version_str} (requires >= 3.10)")


def _configured_config_path() -> Path:
    """Resolve the config file selected for this doctor run."""
    custom_path = os.environ.get("OPENJARVIS_CONFIG")
    if custom_path:
        return Path(custom_path).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


def _check_config_exists() -> CheckResult:
    """Check that the active config file exists."""
    config_path = _configured_config_path()
    if config_path.exists():
        return CheckResult("Config file", "ok", str(config_path))
    details = (
        "Check the OPENJARVIS_CONFIG path."
        if os.environ.get("OPENJARVIS_CONFIG")
        else "Run `jarvis init` to generate a config file."
    )
    return CheckResult(
        "Config file",
        "warn",
        f"Not found at {config_path}",
        details=details,
    )


def _check_config_parses() -> CheckResult:
    """Check that the active config file parses successfully."""
    config_path = _configured_config_path()
    if not config_path.exists():
        return CheckResult("Config parsing", "warn", "Skipped (no config file)")
    try:
        load_config(path=config_path)
        return CheckResult("Config parsing", "ok", "Config loaded successfully")
    except Exception as exc:
        return CheckResult("Config parsing", "fail", f"Parse error: {exc}")


def _ensure_engines_imported() -> None:
    """Import engine modules to trigger registration decorators."""
    try:
        import openjarvis.engine  # noqa: F401
    except Exception:
        pass


def _get_config() -> Any:
    """Load config or return a default if parsing fails."""
    try:
        return load_config(path=_configured_config_path())
    except Exception:
        from openjarvis.core.config import JarvisConfig

        return JarvisConfig()


def _probe_engines(config: Any | None = None) -> Dict[str, _EngineProbe]:
    """Instantiate and health-check every engine once, concurrently."""
    _ensure_engines_imported()

    from openjarvis.core.registry import EngineRegistry
    from openjarvis.engine import _discovery
    from openjarvis.engine._locality import is_local_engine

    resolved_config = config or _get_config()
    keys = sorted(EngineRegistry.keys())
    local_only = (
        resolved_config.intelligence.provider or ""
    ).strip().lower() == "local"
    if local_only:
        # A local-only profile must not send even health probes to cloud APIs.
        keys = [
            key
            for key in keys
            if not bool(getattr(EngineRegistry.get(key), "is_cloud", False))
        ]

    def probe(key: str) -> tuple[str, _EngineProbe] | None:
        engine = None
        try:
            engine = _discovery._make_engine(key, resolved_config)
            if local_only and not is_local_engine(engine, key):
                # The configured host is known only after construction. Skip
                # remote OpenAI-compatible endpoints before health() can
                # issue a request, and keep them out of local diagnostics.
                try:
                    engine.close()
                except Exception:
                    pass
                return None
            return key, _EngineProbe(engine=engine, healthy=bool(engine.health()))
        except Exception as exc:
            if engine is not None:
                try:
                    engine.close()
                except Exception:
                    pass
            return key, _EngineProbe(engine=None, healthy=False, error=str(exc))

    if not keys:
        return {}
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        probed = [item for item in pool.map(probe, keys) if item is not None]
        return dict(probed)


def _check_engines(
    probes: Optional[Dict[str, _EngineProbe]] = None,
) -> List[CheckResult]:
    """Probe each registered engine for health."""
    results: List[CheckResult] = []

    resolved_probes = probes if probes is not None else _probe_engines()

    for key, probe in resolved_probes.items():
        if probe.healthy:
            results.append(CheckResult(f"Engine: {key}", "ok", "Reachable"))
        elif probe.error:
            results.append(
                CheckResult(f"Engine: {key}", "warn", f"Unreachable ({probe.error})")
            )
        else:
            results.append(CheckResult(f"Engine: {key}", "warn", "Unreachable"))

    if not results:
        results.append(CheckResult("Engines", "warn", "No engines registered"))

    return results


def _check_models(
    probes: Optional[Dict[str, _EngineProbe]] = None,
) -> List[CheckResult]:
    """List models from healthy engines."""
    results: List[CheckResult] = []

    resolved_probes = probes if probes is not None else _probe_engines()
    config = _get_config()
    local_only = (config.intelligence.provider or "").strip().lower() == "local"

    for key, probe in resolved_probes.items():
        try:
            if probe.healthy and probe.engine is not None:
                models = probe.engine.list_models()
                if models:
                    model_list = ", ".join(models[:5])
                    suffix = f" (+{len(models) - 5} more)" if len(models) > 5 else ""
                    results.append(
                        CheckResult(
                            f"Models: {key}",
                            "ok",
                            f"{model_list}{suffix}",
                        )
                    )
                else:
                    details = "Pull a model (e.g. `ollama pull qwen3.5:2b`)."
                    if local_only and key == "ollama":
                        from openjarvis.cli._local_setup_hint import (
                            local_model_setup_hint,
                        )

                        details = local_model_setup_hint(config)
                    results.append(
                        CheckResult(
                            f"Models: {key}",
                            "warn",
                            "No models available",
                            details=details,
                        )
                    )
        except Exception:
            continue

    return results


def _check_default_model(
    probes: Optional[Dict[str, _EngineProbe]] = None,
) -> CheckResult:
    """Check whether the configured default model is available."""
    try:
        config = _get_config()
    except Exception:
        return CheckResult("Default model", "warn", "Skipped (config unavailable)")

    default_model = config.intelligence.default_model
    if not default_model:
        details = "Router will select a model dynamically."
        provider = (
            str(getattr(config.intelligence, "provider", "") or "").strip().lower()
        )
        engine = (
            str(
                getattr(config.intelligence, "preferred_engine", "")
                or getattr(config.engine, "default", "")
                or ""
            )
            .strip()
            .lower()
        )
        if provider == "local" and engine == "ollama":
            details += (
                " Run \x60jarvis model setup\x60 to choose a local Ollama model "
                "and optionally download it."
            )
            if sys.platform == "win32":
                details += (
                    " Add \x60--install-runtime\x60 to install a missing Ollama "
                    "runtime with WinGet."
                )
        return CheckResult(
            "Default model",
            "ok",
            "Not configured (auto-routing enabled)",
            details=details,
        )

    from openjarvis.intelligence.model_catalog import resolve_model_id_for_engine

    preferred = config.intelligence.preferred_engine or config.engine.default
    resolved_probes = probes if probes is not None else _probe_engines(config)
    check_order = []
    if preferred:
        check_order.append(preferred)
    check_order += [k for k in resolved_probes if k != preferred]

    for key in check_order:
        try:
            probe = resolved_probes.get(key)
            if probe is not None and probe.healthy and probe.engine is not None:
                models = probe.engine.list_models()
                resolved_model = resolve_model_id_for_engine(default_model, key)
                if resolved_model in models:
                    return CheckResult(
                        "Default model",
                        "ok",
                        f"{default_model} (on {key})",
                    )
        except Exception:
            continue

    return CheckResult(
        "Default model",
        "warn",
        f"{default_model} not found on any engine",
    )


def _check_local_resilience(
    config: Any,
    probes: Dict[str, _EngineProbe],
) -> CheckResult | None:
    """Explain whether local mode has model and backend redundancy."""
    if (config.intelligence.provider or "").strip().lower() != "local":
        return None

    runtime_commands = (
        ("ollama", "Ollama", "ollama", "ollama serve"),
        ("lmstudio", "LM Studio", "lms", "lms server start"),
        ("vllm", "vLLM", "vllm", "vllm serve <model> --host 127.0.0.1"),
        (
            "llamacpp",
            "llama.cpp",
            "llama-server",
            "llama-server --model <path> --host 127.0.0.1",
        ),
    )
    installed_runtimes = [
        (key, name, command)
        for key, name, executable, command in runtime_commands
        if shutil.which(executable)
    ]

    healthy: list[tuple[str, Any, list[str]]] = []
    for key, probe in probes.items():
        if not probe.healthy or probe.engine is None:
            continue
        try:
            models = list(probe.engine.list_models())
        except Exception:
            models = []
        healthy.append((key, probe.engine, models))

    if not healthy:
        details = (
            "Start a configured local runtime. Cloud backends are disabled "
            "while provider = 'local'."
        )
        if installed_runtimes:
            commands = "; ".join(
                f"{name}: `{command}`" for _key, name, command in installed_runtimes
            )
            details += (
                f" Detected runtime CLI(s): {commands}. Start a service, load a "
                "compatible model, then run `jarvis doctor` again."
            )
            selected_engine = (
                (config.intelligence.preferred_engine or config.engine.default or "")
                .strip()
                .lower()
            )
            if (
                config.engine.auto_start_local
                and selected_engine == "ollama"
                and any(key == "ollama" for key, _name, _command in installed_runtimes)
            ):
                details += (
                    " This profile will also attempt to start loopback Ollama when "
                    "chat or the server resolves its model; it will not download "
                    "models, and remote Ollama hosts are refused."
                )
        return CheckResult(
            "Local resilience",
            "fail",
            "No healthy local inference backend",
            details=details,
        )

    fallback_model = (config.intelligence.fallback_model or "").strip()
    fallback_present = any(
        fallback_model in models for _key, _engine, models in healthy
    )

    # Two registry names may point to the same HTTP service. Count distinct
    # endpoints so the report does not mistake aliases for redundancy.
    endpoints: dict[str, str] = {}
    for key, engine, models in healthy:
        if not models:
            continue
        host = getattr(engine, "_host", None)
        identity = str(host).rstrip("/").casefold() if host else key
        endpoints.setdefault(identity, key)

    if len(endpoints) > 1:
        names = ", ".join(endpoints.values())
        detail = f"Healthy backends: {names}. ACE can retry on a second local backend."
        if fallback_model and fallback_present:
            detail += f" Configured fallback model {fallback_model!r} is available."
        return CheckResult(
            "Local resilience",
            "ok",
            f"{len(endpoints)} local inference backends available",
            details=detail,
        )

    key, _engine, models = healthy[0]
    healthy_keys = {item[0] for item in healthy}
    additional_runtimes = [
        (name, command)
        for runtime_key, name, command in installed_runtimes
        if runtime_key not in healthy_keys
    ]
    extra_runtime_hint = ""
    if additional_runtimes:
        name, command = additional_runtimes[0]
        extra_runtime_hint = (
            f" Another local runtime CLI is installed ({name}: `{command}`); "
            "start it with a compatible model for backend-outage failover."
        )
    if fallback_model and fallback_present:
        return CheckResult(
            "Local resilience",
            "warn",
            f"Primary and fallback models share one backend ({key})",
            details=(
                f"{fallback_model!r} is available, which helps if the primary "
                "model is missing. It cannot cover an outage of this runtime; "
                "start a second local backend for runtime failover."
                f"{extra_runtime_hint}"
            ),
        )

    return CheckResult(
        "Local resilience",
        "warn",
        f"Only one local backend is available ({key})",
        details=(
            "No separate local runtime is available for failover. Configure a "
            "fallback model and start a second local backend."
            f"{extra_runtime_hint}"
        ),
    )


def _check_optional_deps() -> List[CheckResult]:
    """Check availability of optional dependency packages."""
    results: List[CheckResult] = []
    optional_packages = [
        ("fastapi", "openjarvis[server]", "REST API framework"),
        ("uvicorn", "openjarvis[server]", "REST API runner"),
        ("multipart", "openjarvis[server]", "REST API file uploads"),
        ("faster_whisper", "openjarvis[speech]", "Local media transcription"),
        ("av", "openjarvis[speech]", "Local audio/video decoding"),
        ("sounddevice", "openjarvis[speech]", "Microphone and speaker audio I/O"),
        ("soundfile", "openjarvis[speech]", "Voice audio playback"),
        ("soundcard", "openjarvis[audio-loopback]", "Windows system audio input"),
        ("kokoro", "openjarvis[voice]", "Local neural text-to-speech (Kokoro)"),
        ("torch", "pip install torch", "SFT/GRPO training"),
        ("pynvml", "openjarvis[gpu-metrics]", "NVIDIA energy monitoring"),
        ("amdsmi", "openjarvis[energy-amd]", "AMD energy monitoring"),
        ("colbert", "openjarvis[memory-colbert]", "ColBERT memory backend"),
        (
            "zeus_apple_silicon",
            "openjarvis[energy-apple]",
            "Apple Silicon energy monitoring",
        ),
        ("apple_fm_sdk", "openjarvis[afm]", "Apple Foundation Models (AFM 3)"),
    ]
    for pkg, install_hint, description in optional_packages:
        try:
            __import__(pkg)
            results.append(CheckResult(f"Optional: {description}", "ok", "Installed"))
        except Exception:
            results.append(
                CheckResult(
                    f"Optional: {description}",
                    "warn",
                    f"Not installed ({install_hint})",
                )
            )
    return results


def _check_ocr_tools() -> CheckResult:
    """Check local OCR executables and whether Tesseract has language data."""
    from openjarvis.tools.storage.ocr import select_ocr_language

    try:
        language = select_ocr_language()
    except Exception as exc:
        return CheckResult(
            "Local OCR",
            "warn",
            "Image OCR unavailable",
            details=str(exc)
            or "Install Tesseract with OCR language data and add it to PATH.",
        )

    if not shutil.which("pdftoppm"):
        return CheckResult(
            "Local OCR",
            "warn",
            f"Image OCR ready ({language}); scanned PDF OCR unavailable",
            details=(
                "Install Poppler and add its `pdftoppm` executable to PATH "
                "to OCR scanned PDF pages."
            ),
        )

    return CheckResult(
        "Local OCR",
        "ok",
        f"Image and scanned PDF OCR ready (Tesseract language: {language})",
    )


def _check_media_model_cache() -> CheckResult:
    """Check for the configured Faster-Whisper model without downloading it."""
    try:
        import av  # noqa: F401
        from faster_whisper.utils import download_model
    except Exception as exc:
        return CheckResult(
            "Media model cache",
            "warn",
            "Cannot load Faster-Whisper or PyAV",
            details=(
                str(exc)
                or "Install media transcription dependencies with "
                "`uv sync --extra speech`."
            ),
        )

    config = _get_config()
    model = (getattr(config.speech, "model", "base") or "base").strip()
    model_path = Path(model).expanduser()
    try:
        if not model_path.is_dir():
            # This library call honors Hugging Face's local cache settings;
            # local_files_only prevents doctor from making a network request.
            model_path = Path(download_model(model, local_files_only=True))
    except Exception:
        return CheckResult(
            "Media model cache",
            "warn",
            f"Configured model {model!r} is not available in the local cache",
            details=(
                "Prepare this Faster-Whisper model in the local cache before "
                "indexing media. `jarvis doctor` does not download models."
            ),
        )

    from openjarvis.speech.faster_whisper import missing_cached_model_files

    if missing := missing_cached_model_files(model_path):
        return CheckResult(
            "Media model cache",
            "warn",
            f"Configured model {model!r} has an incomplete local bundle",
            details=f"Required local file(s) missing: {', '.join(missing)}.",
        )

    return CheckResult(
        "Media model cache",
        "ok",
        f"Configured model {model!r} is present locally",
        details=(
            "This confirms cached files only; it does not load the model or "
            "verify device compatibility."
        ),
    )


def _check_speech_backend() -> CheckResult:
    """Check whether the configured speech backend can load."""
    try:
        from openjarvis.speech._discovery import get_speech_backend

        config = _get_config()
        backend = get_speech_backend(config)
        if backend is None:
            if config.speech.offline_only:
                return CheckResult(
                    "Speech backend",
                    "warn",
                    "Local speech recognition is unavailable",
                    details=(
                        "Check the desktop dependencies, the Media model cache "
                        "check above, and the configured speech device."
                    ),
                )
            return CheckResult(
                "Speech backend",
                "warn",
                "Not configured",
                details="Install desktop dependencies with `uv sync --extra desktop`.",
            )

        if backend.health():
            return CheckResult(
                "Speech backend",
                "ok",
                f"{backend.backend_id} ready",
            )

        details = None
        last_error = getattr(backend, "last_error", None)
        if callable(last_error):
            details = last_error()
        return CheckResult(
            "Speech backend",
            "warn",
            f"{backend.backend_id} unavailable",
            details=details
            or "Install desktop dependencies with `uv sync --extra desktop`.",
        )
    except Exception as exc:
        return CheckResult(
            "Speech backend",
            "warn",
            f"Could not check: {exc}",
        )


def _check_local_tts_backend() -> CheckResult:
    """Check whether a local text-to-speech backend is ready."""
    try:
        from openjarvis.speech._tts_discovery import get_tts_backend

        backend = get_tts_backend(offline_only=True)
    except Exception as exc:
        return CheckResult(
            "Local text-to-speech",
            "warn",
            "Could not check local TTS",
            details=str(exc),
        )
    if backend is not None:
        return CheckResult(
            "Local text-to-speech",
            "ok",
            f"{backend.backend_id} ready",
        )
    return CheckResult(
        "Local text-to-speech",
        "warn",
        "No local TTS backend available",
        details=(
            "Install Kokoro with `uv sync --extra voice`, or install a Windows "
            "speech voice for the built-in SAPI backend."
        ),
    )


def _check_security_profile() -> CheckResult:
    """Check if a security profile is configured."""
    try:
        from openjarvis.core.config import load_config

        config = load_config()
        if config.security.profile:
            return CheckResult(
                name="Security profile",
                status="ok",
                message=f"Profile '{config.security.profile}' active",
            )
        return CheckResult(
            name="Security profile",
            status="warn",
            message="No security profile set",
            details="Recommended: add security.profile = 'personal' to config.toml",
        )
    except Exception as exc:
        return CheckResult(
            name="Security profile",
            status="fail",
            message=f"Could not check: {exc}",
        )


def _check_nodejs() -> CheckResult:
    """Check Node.js version for Node-backed integrations."""
    node_path = shutil.which("node")
    if not node_path:
        return CheckResult(
            "Node.js",
            "warn",
            "Not found",
            details=(
                "Node.js 22+ is required for ClaudeCodeAgent and the "
                "WhatsApp Baileys channel bridge."
            ),
        )
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version_str = result.stdout.strip()
        # Parse "v22.1.0" -> (22, 1, 0)
        parts = version_str.lstrip("v").split(".")
        major = int(parts[0])
        if major >= 22:
            return CheckResult("Node.js", "ok", version_str)
        return CheckResult(
            "Node.js",
            "warn",
            f"{version_str} (requires >= v22)",
            details=(
                "Upgrade Node.js for ClaudeCodeAgent and WhatsApp Baileys support."
            ),
        )
    except Exception as exc:
        return CheckResult("Node.js", "warn", f"Error checking version: {exc}")


# -- Main command -------------------------------------------------------------

_STATUS_ICONS = {
    "ok": "[green]\u2713[/green]",
    "warn": "[yellow]![/yellow]",
    "fail": "[red]\u2717[/red]",
}


def _run_all_checks() -> List[CheckResult]:
    """Run all diagnostic checks and return results."""
    checks: List[CheckResult] = []
    checks.append(_check_python_version())
    checks.append(_check_config_exists())
    checks.append(_check_config_parses())
    config = _get_config()
    engine_probes = _probe_engines(config)
    checks.extend(_check_engines(engine_probes))
    checks.extend(_check_models(engine_probes))
    checks.append(_check_default_model(engine_probes))
    local_resilience = _check_local_resilience(config, engine_probes)
    if local_resilience is not None:
        checks.append(local_resilience)
    checks.extend(_check_optional_deps())
    checks.append(_check_ocr_tools())
    checks.append(_check_media_model_cache())
    checks.append(_check_speech_backend())
    checks.append(_check_local_tts_backend())
    checks.append(_check_nodejs())
    checks.append(_check_security_profile())
    return checks


def _results_to_dicts(checks: List[CheckResult]) -> List[Dict[str, Any]]:
    """Convert CheckResult list to JSON-serializable dicts."""
    return [asdict(c) for c in checks]


@click.command()
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON.")
def doctor(as_json: bool) -> None:
    """Run diagnostic checks on your OpenJarvis installation."""
    checks = _run_all_checks()

    if as_json:
        click.echo(json.dumps(_results_to_dicts(checks), indent=2))
        return

    console = Console()
    console.print()
    console.print("[bold]OpenJarvis Doctor[/bold]")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Status", width=3, justify="center")
    table.add_column("Check")
    table.add_column("Result")

    for check in checks:
        icon = _STATUS_ICONS.get(check.status, "?")
        message = check.message
        if check.details:
            message += f"\n  [dim]{check.details}[/dim]"
        table.add_row(icon, check.name, message)

    console.print(table)

    ok_count = sum(1 for c in checks if c.status == "ok")
    warn_count = sum(1 for c in checks if c.status == "warn")
    fail_count = sum(1 for c in checks if c.status == "fail")
    console.print()
    console.print(f"  {ok_count} passed, {warn_count} warnings, {fail_count} failures")
    console.print()

    # Background tasks section
    from openjarvis.cli._bg_state import get_status
    from openjarvis.core.paths import get_config_dir

    scripts_dir = get_config_dir() / ".scripts"
    console.print("[bold]Background tasks[/bold]")
    bg = get_status()
    bg_failed = False

    if bg.rust_extension == "ready":
        console.print("  [green]✓[/green] Rust extension: ready")
    elif bg.rust_extension == "failed":
        console.print(f"  [red]✗[/red] Rust extension: failed — {bg.rust_error[:80]}")
        console.print(
            f"    retry: {scripts_dir}/install-rust.sh && "
            f"{scripts_dir}/build-extension.sh"
        )
        bg_failed = True
    else:
        console.print(
            "  [yellow]…[/yellow] Rust extension: building (run in background)"
        )

    if not bg.models:
        console.print("  [dim]no model downloads tracked[/dim]")
    for model_id, state in bg.models.items():
        if state == "ready":
            console.print(f"  [green]✓[/green] {model_id}: ready")
        elif state == "failed":
            console.print(f"  [red]✗[/red] {model_id}: failed")
            console.print(f"    retry: {scripts_dir}/pull-model.sh {model_id}")
            bg_failed = True
        else:
            console.print(f"  [yellow]…[/yellow] {model_id}: downloading")

    if bg_failed:
        raise click.exceptions.Exit(code=1)
