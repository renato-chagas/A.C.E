"""Explicitly configured, loopback-only local inference runtime startup."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from openjarvis.engine._locality import is_local_engine

logger = logging.getLogger(__name__)

_START_LOCK = threading.Lock()
_STARTED_PROCESSES: dict[str, subprocess.Popen[bytes]] = {}


def find_local_ollama_executable() -> str | None:
    """Find Ollama on PATH or in its standard Windows install locations."""
    executable = shutil.which("ollama")
    if executable:
        return executable
    if sys.platform != "win32":
        return None

    local_app_data = os.environ.get("LOCALAPPDATA")
    program_files = os.environ.get("ProgramFiles") or os.environ.get("PROGRAMFILES")
    candidates = []
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe")
    if program_files:
        candidates.append(Path(program_files) / "Ollama" / "ollama.exe")
    for candidate in candidates:
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue
    return None


def _ollama_bind_address(host: str) -> str | None:
    """Convert a verified loopback HTTP URL to Ollama's bind-address format."""
    try:
        parsed = urlsplit(host if "://" in host else f"http://{host}")
        if parsed.scheme != "http" or parsed.username or parsed.password:
            return None
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            return None
        if not is_local_engine(SimpleNamespace(_host=host), "ollama"):
            return None
        hostname = parsed.hostname
        if not hostname:
            return None
        port = parsed.port or 80
    except ValueError:
        return None

    if ":" in hostname:
        hostname = f"[{hostname}]"
    return f"{hostname}:{port}"


def start_local_ollama(host: str) -> bool:
    """Start Ollama for a loopback endpoint without shell execution.

    Only called by the opt-in local-only engine resolver after a failed health
    check. The process is intentionally left running when ACE exits so other
    local clients can use the same runtime.
    """
    bind_address = _ollama_bind_address(host)
    if bind_address is None:
        logger.warning("Refusing to auto-start Ollama for a non-loopback host")
        return False

    executable = find_local_ollama_executable()
    if not executable:
        logger.info("Ollama auto-start requested but the CLI is not installed")
        return False

    with _START_LOCK:
        current = _STARTED_PROCESSES.get(bind_address)
        if current is not None and current.poll() is None:
            return True

        env = os.environ.copy()
        # Override inherited OLLAMA_HOST so the service cannot bind to a
        # different (possibly network-visible) interface than the verified URL.
        env["OLLAMA_HOST"] = bind_address
        options: dict[str, object] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": env,
            "close_fds": True,
        }
        if os.name == "nt":
            options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            options["start_new_session"] = True

        try:
            process = subprocess.Popen([executable, "serve"], **options)
        except OSError as exc:
            logger.warning("Could not start local Ollama runtime: %s", exc)
            return False

        _STARTED_PROCESSES[bind_address] = process
        logger.info("Started local Ollama runtime for %s", bind_address)
        return True


__all__ = ["find_local_ollama_executable", "start_local_ollama"]
