"""Strict endpoint checks for engines selected in local-only mode."""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlsplit

_IN_PROCESS_LOCAL_ENGINES = frozenset({"afm", "gemma_cpp"})
_LOOPBACK_DEFAULT_ENGINES = frozenset(
    {
        "apple_fm",
        "exo",
        "llamacpp",
        "lemonade",
        "mlx",
        "nexa",
        "ollama",
        "sglang",
        "uzu",
        "vllm",
    }
)
LOCAL_ENGINE_KEYS = _IN_PROCESS_LOCAL_ENGINES | _LOOPBACK_DEFAULT_ENGINES


def is_local_engine(engine: Any, key: str = "") -> bool:
    """Return true only for loopback HTTP engines or known in-process engines.

    Custom in-process engines can opt in with ``is_local = True``. A backend
    marked as cloud is never local, even if it happens to use a loopback proxy.
    """
    if bool(getattr(engine, "is_cloud", False)):
        return False
    host = getattr(engine, "_host", "") or ""
    if isinstance(host, str) and host.strip():
        try:
            parsed = urlsplit(host if "://" in host else f"http://{host}")
            if (
                parsed.scheme not in {"http", "https"}
                or parsed.username
                or parsed.password
            ):
                return False
            try:
                if parsed.port == 0:
                    return False
            except ValueError:
                return False
            hostname = (parsed.hostname or "").casefold().rstrip(".")
            if not hostname:
                return False
            if hostname == "localhost":
                return True
            try:
                return ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                return False
        except ValueError:
            return False

    engine_key = (key or getattr(engine, "engine_id", "") or "").strip().casefold()
    return bool(getattr(engine, "is_local", False) or engine_key in LOCAL_ENGINE_KEYS)


def is_local_engine_for_model(engine: Any, model: str, key: str = "") -> bool:
    """Check the concrete backend selected for a model by a routing wrapper."""
    resolver = getattr(engine, "is_local_for_model", None)
    if callable(resolver):
        try:
            return bool(resolver(model))
        except Exception:
            return False
    return is_local_engine(engine, key)


__all__ = ["LOCAL_ENGINE_KEYS", "is_local_engine", "is_local_engine_for_model"]
