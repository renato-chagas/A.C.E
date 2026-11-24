"""Local-only transcription helpers for indexing audio and video files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_MAX_MEDIA_BYTES = 50 * 1024 * 1024
_MAX_MEDIA_SECONDS = 30 * 60
_MAX_TRANSCRIPT_CHARS = 100_000
_TRUNCATION_MARKER = "\n[Media transcript truncated at 100,000 characters.]"

_AUDIO_EXTS = frozenset(
    {".aac", ".flac", ".m4a", ".mp3", ".oga", ".ogg", ".opus", ".wav", ".wma"}
)
_VIDEO_EXTS = frozenset(
    {
        ".3gp",
        ".avi",
        ".flv",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".webm",
        ".wmv",
    }
)


class MediaTranscriptionUnavailableError(RuntimeError):
    """Raised when the local model or its offline dependencies are unavailable."""


class MediaProcessingError(RuntimeError):
    """Raised when a supported local media file cannot be transcribed."""


def detect_media_type(path: str | Path) -> str | None:
    """Return audio/video for supported suffixes, or None for other files."""
    suffix = Path(path).suffix.lower()
    if suffix in _AUDIO_EXTS:
        return "audio"
    if suffix in _VIDEO_EXTS:
        return "video"
    return None


def transcribe_media_file(
    path: str | Path,
    *,
    speech_config: Any = None,
    language: str | None = None,
) -> str:
    """Transcribe one audio track with a cached local Faster-Whisper model.

    Model downloads are disabled for this indexing path. Video files contribute
    their audio track only; video frames are not analyzed.
    """
    media_path = Path(path).expanduser().resolve()
    media_type = detect_media_type(media_path)
    if media_type is None:
        raise ValueError(f"Unsupported media format: {media_path.suffix or '(none)'}")
    if not media_path.is_file():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    size = media_path.stat().st_size
    if size <= 0:
        raise ValueError("The media file is empty.")
    if size > _MAX_MEDIA_BYTES:
        raise ValueError("Local media transcription is limited to 50 MiB per file.")

    try:
        with media_path.open("rb") as media_stream:
            audio = media_stream.read(_MAX_MEDIA_BYTES + 1)
    except OSError as exc:
        raise MediaProcessingError(
            f"Could not read local media file {media_path.name}."
        ) from exc
    if len(audio) > _MAX_MEDIA_BYTES:
        raise ValueError("Local media transcription is limited to 50 MiB per file.")

    try:
        import av
    except ImportError as exc:
        raise MediaTranscriptionUnavailableError(
            "Local media transcription requires Faster-Whisper and PyAV. "
            "Install the speech extra and make the model available in the "
            "local cache."
        ) from exc

    try:
        with av.open(str(media_path)) as container:
            if not any(stream.type == "audio" for stream in container.streams):
                raise MediaProcessingError("The media file has no audio track.")
            duration = getattr(container, "duration", None)
            if duration is not None and duration / av.time_base > _MAX_MEDIA_SECONDS:
                raise ValueError("Local media transcription is limited to 30 minutes.")
    except (MediaProcessingError, ValueError):
        raise
    except Exception as exc:
        raise MediaProcessingError(
            "PyAV could not inspect the local media file."
        ) from exc

    if speech_config is None:
        from openjarvis.core.config import load_config

        speech_config = load_config().speech

    try:
        from openjarvis.speech.faster_whisper import FasterWhisperBackend
    except ImportError as exc:
        raise MediaTranscriptionUnavailableError(
            "Local media transcription requires the Faster-Whisper speech extra."
        ) from exc

    backend = FasterWhisperBackend(
        model_size=getattr(speech_config, "model", "base"),
        device=getattr(speech_config, "device", "auto"),
        compute_type=getattr(speech_config, "compute_type", "float16"),
        local_files_only=True,
    )
    if not backend.health():
        detail = backend.last_error() or "the local model could not be loaded"
        raise MediaTranscriptionUnavailableError(
            "The local Faster-Whisper model is unavailable or not cached: "
            f"{detail}"
        )

    try:
        result = backend.transcribe(
            audio,
            format=media_path.suffix.lstrip(".").lower(),
            language=language or getattr(speech_config, "language", "") or None,
            max_text_chars=_MAX_TRANSCRIPT_CHARS - len(_TRUNCATION_MARKER),
        )
    except (ImportError, OSError) as exc:
        raise MediaTranscriptionUnavailableError(
            f"Local media transcription could not start: {exc}"
        ) from exc
    except Exception as exc:
        raise MediaProcessingError(
            f"Local transcription failed for {media_path.name}: {exc}"
        ) from exc

    text = result.text.strip()
    if result.truncated:
        text = text[: _MAX_TRANSCRIPT_CHARS - len(_TRUNCATION_MARKER)]
        text += _TRUNCATION_MARKER
    return text


__all__ = [
    "MediaProcessingError",
    "MediaTranscriptionUnavailableError",
    "detect_media_type",
    "transcribe_media_file",
]
