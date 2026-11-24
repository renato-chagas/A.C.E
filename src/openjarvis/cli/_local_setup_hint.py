"""Actionable setup guidance for local-only inference failures."""

from __future__ import annotations

import re
import sys
from typing import Any

_SAFE_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*\Z")


def local_model_setup_hint(
    config: Any,
    *,
    requested_model: str | None = None,
) -> str:
    """Explain how to prepare the selected local runtime without downloading.

    Model IDs are shown as pull commands only when they match the runtime's
    ordinary identifier characters; configuration text is never executed.
    """
    if (getattr(config.intelligence, "provider", "") or "").strip().lower() != "local":
        return ""

    selected_engine = (
        (
            getattr(config.intelligence, "preferred_engine", "")
            or getattr(config.engine, "default", "")
            or ""
        )
        .strip()
        .lower()
    )
    if selected_engine != "ollama":
        return (
            "This profile is local-only, so cloud providers will not be used. "
            "Start the selected local runtime with a compatible model and run "
            "`jarvis doctor` to inspect availability."
        )

    lines = ["This profile is local-only, so cloud providers will not be used."]
    if getattr(config.engine, "auto_start_local", False):
        lines.append(
            "ACE can start loopback Ollama automatically, but it does not "
            "download models."
        )
    else:
        lines.append(
            "Start Ollama Desktop or run `ollama serve` in a separate terminal."
        )
    if sys.platform == "win32":
        lines.append(
            "If Ollama is missing, `jarvis model setup --install-runtime` can "
            "install it through WinGet."
        )
    else:
        lines.append(
            "Install Ollama from the official ollama.com/download page if needed."
        )

    model = requested_model or getattr(config.intelligence, "default_model", "")
    fallback = getattr(config.intelligence, "fallback_model", "")
    model_ids = list(dict.fromkeys(item for item in (model, fallback) if item))
    commands = [
        f"`ollama pull {model_id}`"
        for model_id in model_ids
        if _SAFE_MODEL_ID.fullmatch(model_id)
    ]
    if commands:
        message = ", ".join(commands)
        lines.append(f"Download the configured model(s) manually: {message}.")
        lines.append(
            "Choose another model or download one interactively with "
            "`jarvis model setup`."
        )
    else:
        lines.append(
            "Run `jarvis model setup` to choose a local Ollama model and "
            "optionally download it."
        )
    lines.append("Run `jarvis doctor` to check runtime and model availability.")
    return " ".join(lines)


__all__ = ["local_model_setup_hint"]
