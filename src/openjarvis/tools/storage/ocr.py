"""Offline OCR for image files and in-memory screen captures."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_TEXT_CHARS = 100_000
_OCR_TIMEOUT_SECONDS = 30
_LANGUAGE_TIMEOUT_SECONDS = 10
_LANGUAGE_NAME = re.compile(r"^[a-z0-9_]+$")


class OcrUnavailableError(RuntimeError):
    """Raised when a local OCR executable or language pack is unavailable."""


class OcrProcessingError(RuntimeError):
    """Raised when local OCR cannot process an otherwise accepted image."""


def _tesseract_path() -> str:
    executable = shutil.which("tesseract")
    if not executable:
        raise OcrUnavailableError(
            "Image OCR requires the local Tesseract executable on PATH. "
            "Install Tesseract with Portuguese or English language data, then retry."
        )
    return executable


def _select_language(executable: str) -> str:
    try:
        result = subprocess.run(
            [executable, "--list-langs"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_LANGUAGE_TIMEOUT_SECONDS,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OcrUnavailableError(
            "Tesseract language data could not be checked locally."
        ) from exc

    languages = {
        line.strip()
        for line in result.stdout.splitlines()
        if _LANGUAGE_NAME.fullmatch(line.strip()) and line.strip() != "osd"
    }
    if result.returncode != 0 or not languages:
        raise OcrUnavailableError(
            "Tesseract has no readable OCR language data installed."
        )

    preferred = [language for language in ("por", "eng") if language in languages]
    if preferred:
        return "+".join(preferred)
    return sorted(languages)[0]


def select_ocr_language() -> str:
    """Select an installed local Tesseract language for a batch of OCR calls."""
    return _select_language(_tesseract_path())


def ocr_image(
    path: str | Path,
    *,
    timeout_seconds: float = _OCR_TIMEOUT_SECONDS,
    language: str | None = None,
) -> str:
    """Extract bounded text from one image using local Tesseract only."""
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    size = image_path.stat().st_size
    if size <= 0:
        raise ValueError("The image file is empty.")
    if size > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image OCR is limited to {_MAX_IMAGE_BYTES // (1024 * 1024)} MiB per file."
        )
    return _run_tesseract(
        str(image_path),
        language=language,
        timeout_seconds=timeout_seconds,
    )


def screen_ocr_reader() -> Callable[[bytes], str]:
    """Prepare local OCR before capturing an in-memory screen image."""
    try:
        executable = _tesseract_path()
        language = _select_language(executable)
    except OcrUnavailableError:
        if sys.platform != "win32":
            raise
        from openjarvis.tools.storage._windows_ocr import windows_ocr_reader

        reader = windows_ocr_reader()
    else:

        def reader(image: bytes) -> str:
            return _run_tesseract(
                "stdin",
                executable=executable,
                language=language,
                image=image,
                page_mode="11",
            )

    def recognize(image: bytes) -> str:
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Screen OCR requires an in-memory PNG image.")
        if len(image) > _MAX_IMAGE_BYTES:
            raise ValueError(
                f"Screen OCR is limited to {_MAX_IMAGE_BYTES // (1024 * 1024)} MiB."
            )
        return reader(image).strip()[:_MAX_TEXT_CHARS]

    return recognize


def _run_tesseract(
    source: str,
    *,
    executable: str | None = None,
    language: str | None,
    timeout_seconds: float = _OCR_TIMEOUT_SECONDS,
    image: bytes | None = None,
    page_mode: str = "3",
) -> str:
    if timeout_seconds <= 0:
        raise OcrProcessingError("Local image OCR exceeded its time limit.")
    executable = executable or _tesseract_path()
    language = language or _select_language(executable)
    if any(
        not _LANGUAGE_NAME.fullmatch(part) or part == "osd"
        for part in language.split("+")
    ):
        raise ValueError("Invalid local OCR language selection.")
    try:
        result = subprocess.run(
            [
                executable,
                source,
                "stdout",
                "--psm",
                page_mode,
                "-l",
                language,
            ],
            input=image,
            capture_output=True,
            timeout=min(timeout_seconds, _OCR_TIMEOUT_SECONDS),
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        if timeout_seconds >= _OCR_TIMEOUT_SECONDS:
            message = "Local image OCR exceeded its 30-second limit."
        else:
            message = "Local image OCR exceeded the remaining document time budget."
        raise OcrProcessingError(message) from exc
    except OSError as exc:
        raise OcrProcessingError(
            "The local Tesseract process could not be started."
        ) from exc

    if result.returncode != 0:
        raise OcrProcessingError("Local Tesseract could not read this image.")
    text = result.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")
    return text.replace("\r", "\n").strip()[:_MAX_TEXT_CHARS]


__all__ = [
    "OcrProcessingError",
    "OcrUnavailableError",
    "ocr_image",
    "screen_ocr_reader",
    "select_ocr_language",
]
