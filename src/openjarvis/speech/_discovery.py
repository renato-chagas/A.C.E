"""Auto-discover available speech-to-text backends."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from openjarvis.core.config import JarvisConfig
    from openjarvis.speech._stubs import SpeechBackend

# Priority order: local first, then cloud
DISCOVERY_ORDER = [
    "faster-whisper",
    "openai",
    "deepgram",
]
LOCAL_BACKENDS = ("faster-whisper",)


def _create_backend(
    key: str,
    config: "JarvisConfig",
) -> Optional["SpeechBackend"]:
    """Try to instantiate a speech backend by registry key."""
    from openjarvis.core.registry import SpeechRegistry

    if not SpeechRegistry.contains(key):
        return None

    try:
        backend_cls = SpeechRegistry.get(key)

        if key == "faster-whisper":
            return backend_cls(
                model_size=config.speech.model,
                device=config.speech.device,
                compute_type=config.speech.compute_type,
                local_files_only=bool(getattr(config.speech, "offline_only", False)),
            )
        elif key == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return None
            return backend_cls(api_key=api_key)
        elif key == "deepgram":
            api_key = os.environ.get("DEEPGRAM_API_KEY", "")
            if not api_key:
                return None
            return backend_cls(api_key=api_key)
        else:
            return backend_cls()
    except Exception:
        return None


def get_speech_backend(config: "JarvisConfig") -> Optional["SpeechBackend"]:
    """Resolve the speech backend from config.

    If ``config.speech.backend`` is ``"auto"``, tries backends in
    priority order and returns the first healthy one.
    """
    # Trigger registration of built-in backends
    import openjarvis.speech  # noqa: F401

    backend_key = config.speech.backend
    offline_only = bool(getattr(config.speech, "offline_only", False))

    if backend_key != "auto":
        if offline_only and backend_key not in LOCAL_BACKENDS:
            return None
        backend = _create_backend(backend_key, config)
        if backend is None:
            return None
        try:
            return backend if backend.health() else None
        except Exception:
            return None

    # Auto-discovery: try each in priority order
    discovery_order = LOCAL_BACKENDS if offline_only else DISCOVERY_ORDER
    for key in discovery_order:
        backend = _create_backend(key, config)
        if backend is None:
            continue
        try:
            if backend.health():
                return backend
        except Exception:
            continue

    return None
