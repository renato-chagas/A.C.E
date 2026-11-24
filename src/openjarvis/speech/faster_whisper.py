"""Faster-Whisper speech-to-text backend (local, CTranslate2-based)."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional

from openjarvis.core.registry import SpeechRegistry
from openjarvis.speech._stubs import Segment, SpeechBackend, TranscriptionResult

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment, misc]

try:
    import ctranslate2
except ImportError:
    ctranslate2 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def missing_cached_model_files(path: Path) -> tuple[str, ...]:
    """Return the files required to load a Faster-Whisper bundle offline."""
    # Faster-Whisper fetches a pretrained tokenizer if this file is absent.
    return tuple(
        name
        for name in ("config.json", "model.bin", "tokenizer.json")
        if not (path / name).is_file()
    )


def _cached_model_directory(model_size: str) -> str:
    """Resolve a local model bundle without asking the Hub for updates."""
    path = Path(model_size).expanduser()
    if not path.is_dir():
        from faster_whisper.utils import download_model

        try:
            path = Path(download_model(model_size, local_files_only=True))
        except Exception as exc:
            raise FileNotFoundError(
                f"Faster-Whisper model {model_size!r} is unavailable in the local "
                "cache. Prepare the model separately or set [speech].model to "
                "a local model directory."
            ) from exc

    missing = missing_cached_model_files(path)
    if missing:
        raise FileNotFoundError(
            "The local Faster-Whisper bundle is incomplete. Missing: "
            + ", ".join(missing)
            + "."
        )
    return str(path.resolve())


@SpeechRegistry.register("faster-whisper")
class FasterWhisperBackend(SpeechBackend):
    """Local speech-to-text using Faster-Whisper (CTranslate2)."""

    backend_id = "faster-whisper"

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "float16",
        local_files_only: bool = False,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._local_files_only = local_files_only
        self._model: Optional[WhisperModel] = None
        self._last_error: Optional[str] = None

    def _resolve_compute_type(self) -> str:
        """Pick a CTranslate2 compute type supported by the configured device."""
        if ctranslate2 is None:
            return self._compute_type

        device = self._device
        try:
            if device == "auto":
                device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
            supported = set(ctranslate2.get_supported_compute_types(device))
        except Exception as exc:
            logger.debug(
                "Could not inspect CTranslate2 compute types for %s: %s",
                self._device,
                exc,
            )
            return self._compute_type

        if self._compute_type in supported:
            return self._compute_type

        preferences = (
            ("int8", "float32", "int8_float32", "int16")
            if self._compute_type == "float16"
            else ("float32", "int8", "int8_float32", "int16")
        )
        fallback = next((value for value in preferences if value in supported), None)
        if fallback is None:
            return self._compute_type

        logger.warning(
            "CTranslate2 compute_type=%r is not supported on device=%r; "
            "using %r instead",
            self._compute_type,
            device,
            fallback,
        )
        return fallback

    def _ensure_model(self) -> WhisperModel:
        """Lazy-load the Whisper model on first use."""
        if self._model is None:
            if WhisperModel is None:
                self._last_error = (
                    "faster-whisper is not installed. "
                    "Install with: uv sync --extra speech "
                    "(or --extra desktop for microphone support)."
                )
                raise ImportError(self._last_error)
            compute_type = self._resolve_compute_type()
            kwargs: dict[str, Any] = {
                "device": self._device,
                "compute_type": compute_type,
            }
            model_source = self._model_size
            if self._local_files_only:
                model_source = _cached_model_directory(model_source)
                kwargs["local_files_only"] = True
            self._model = WhisperModel(model_source, **kwargs)
        self._last_error = None
        return self._model

    def transcribe(
        self,
        audio: bytes,
        *,
        format: str = "wav",
        language: Optional[str] = None,
        max_text_chars: Optional[int] = None,
    ) -> TranscriptionResult:
        """Transcribe audio bytes using Faster-Whisper."""
        if max_text_chars is not None and max_text_chars <= 0:
            raise ValueError("max_text_chars must be greater than zero.")
        try:
            model = self._ensure_model()

            # PyAV detects the container from the bytes. Keep the stream open
            # until Faster-Whisper has finished iterating over its segments.
            with BytesIO(audio) as stream:
                kwargs = {"language": language} if language else {}
                segments_iter, info = model.transcribe(stream, **kwargs)
                segments_list = []
                text_chars = 0
                truncated = False
                for segment in segments_iter:
                    segment_text = segment.text
                    if max_text_chars is not None:
                        remaining = max_text_chars - text_chars
                        if remaining <= 0:
                            truncated = True
                            break
                        if len(segment_text) > remaining:
                            segment_text = segment_text[:remaining]
                            truncated = True
                    segments_list.append((segment_text, segment.start, segment.end))
                    text_chars += len(segment_text)
                    if truncated:
                        break
        except Exception as exc:
            self._last_error = str(exc)
            raise

        # Build result
        text = "".join(seg_text for seg_text, _, _ in segments_list).strip()
        segments = [
            Segment(
                text=seg_text.strip(),
                start=start,
                end=end,
                confidence=None,
            )
            for seg_text, start, end in segments_list
        ]

        self._last_error = None
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            confidence=getattr(info, "language_probability", None),
            duration_seconds=getattr(info, "duration", 0.0),
            segments=segments,
            truncated=truncated,
        )

    def health(self) -> bool:
        """Check if model is loaded or loadable."""
        try:
            self._ensure_model()
            return True
        except Exception as exc:
            self._last_error = str(exc)
            logger.debug("Faster-Whisper health check failed: %s", exc)
            return False

    def last_error(self) -> Optional[str]:
        """Return the last model load or transcription error, if any."""
        return self._last_error

    def supported_formats(self) -> List[str]:
        """Audio formats and video containers whose audio track can be read."""
        return [
            "3gp",
            "aac",
            "avi",
            "flv",
            "flac",
            "m4a",
            "m4v",
            "mkv",
            "mov",
            "mp3",
            "mp4",
            "mpeg",
            "mpg",
            "oga",
            "ogg",
            "opus",
            "wav",
            "webm",
            "wma",
            "wmv",
        ]
