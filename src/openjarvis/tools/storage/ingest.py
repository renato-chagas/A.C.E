"""Document ingestion — file reading, type detection, directory walking."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from openjarvis.tools.storage.chunking import Chunk, ChunkConfig, chunk_text
from openjarvis.tools.storage.media import detect_media_type, transcribe_media_file
from openjarvis.tools.storage.ocr import (
    OcrProcessingError,
    OcrUnavailableError,
    ocr_image,
    select_ocr_language,
)

_MAX_PDF_OCR_BYTES = 50 * 1024 * 1024
_MAX_PDF_OCR_PAGES = 20
_MAX_PDF_OCR_SECONDS = 180
_MAX_PDF_OCR_TEXT_CHARS = 100_000
_PDF_PAGE_RENDER_TIMEOUT_SECONDS = 20
_PDF_PAGE_RENDER_SIZE = 1800

# Directories to skip when walking a tree
_SKIP_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "__pypackages__",
        ".eggs",
        "*.egg-info",
    }
)

# Extension -> file-type mapping
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})

_CODE_EXTS = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".rs",
        ".go",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".rb",
        ".sh",
        ".bash",
        ".zsh",
        ".lua",
        ".swift",
        ".kt",
        ".scala",
        ".cs",
        ".r",
        ".sql",
        ".yaml",
        ".yml",
        ".toml",
        ".json",
        ".xml",
        ".html",
        ".css",
    }
)


@dataclass(slots=True)
class DocumentMeta:
    """Metadata about an ingested document."""

    path: str
    file_type: str
    size_bytes: int
    line_count: int


def detect_file_type(path: Path) -> str:
    """Map an extension to a supported document, image, audio, or video type."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".mdx"}:
        return "markdown"
    if suffix == ".pdf":
        return "pdf"
    if suffix in _IMAGE_EXTS:
        return "image"
    media_type = detect_media_type(path)
    if media_type:
        return media_type
    if suffix in _CODE_EXTS:
        return "code"
    return "text"


def read_document(
    path: Path,
    *,
    ocr: bool = False,
    transcribe_media: bool = False,
    speech_config=None,
) -> Tuple[str, DocumentMeta]:
    """Read a file and return ``(text, metadata)``.

    Raises
    ------
    ImportError
        If the file is a PDF and ``pdfplumber`` is not installed.
    FileNotFoundError
        If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ftype = detect_file_type(path)

    if ftype == "image":
        if not ocr:
            raise ValueError(
                "Image indexing requires OCR. Enable it with --ocr or the "
                "Use local OCR option."
            )
        text = ocr_image(path)
    elif ftype == "pdf":
        try:
            import pdfplumber  # noqa: F401
        except ImportError:
            raise ImportError(
                "PDF support requires pdfplumber. "
                "Install it with: uv sync --extra memory-pdf"
            ) from None

        text = _read_pdf(path, ocr=ocr)
    elif ftype in {"audio", "video"}:
        if not transcribe_media:
            raise ValueError(
                "Audio/video indexing requires local transcription. "
                "Enable --transcribe-media to opt in."
            )
        text = transcribe_media_file(path, speech_config=speech_config)
    else:
        text = _read_text(path)

    line_count = text.count("\n") + 1 if text else 0
    meta = DocumentMeta(
        path=str(path),
        file_type=ftype,
        size_bytes=path.stat().st_size,
        line_count=line_count,
    )
    return text, meta


def _read_text(path: Path) -> str:
    """Read a text file with UTF-8, falling back to latin-1."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _read_pdf(path: Path, *, ocr: bool = False) -> str:
    """Extract PDF text, optionally OCRing image-only pages on this machine."""
    import pdfplumber

    pages: List[str] = []
    ocr_started = time.monotonic()
    ocr_pages = 0
    ocr_text_chars = 0
    language: str | None = None
    renderer: str | None = None
    size_checked = False
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(text)
                continue

            # Empty text layers are common in scans. Ignore blank pages so a
            # normal PDF does not require OCR tools just because it contains
            # an intentionally blank page.
            if not ocr or not getattr(page, "images", None):
                continue

            if not size_checked:
                if path.stat().st_size > _MAX_PDF_OCR_BYTES:
                    raise ValueError("Scanned PDF OCR is limited to 50 MiB per file.")
                size_checked = True
            ocr_pages += 1
            if ocr_pages > _MAX_PDF_OCR_PAGES:
                raise ValueError(
                    "Scanned PDF OCR is limited to 20 image-only pages per file."
                )
            remaining = _MAX_PDF_OCR_SECONDS - (time.monotonic() - ocr_started)
            if remaining <= 0:
                raise OcrProcessingError(
                    "Scanned PDF OCR exceeded its 180-second time limit."
                )

            if renderer is None:
                renderer = shutil.which("pdftoppm")
                if not renderer:
                    raise OcrUnavailableError(
                        "Scanned PDF OCR requires the local Poppler "
                        "'pdftoppm' executable on PATH."
                    )
            if language is None:
                language = select_ocr_language()
            remaining = _MAX_PDF_OCR_SECONDS - (time.monotonic() - ocr_started)
            if remaining <= 0:
                raise OcrProcessingError(
                    "Scanned PDF OCR exceeded its 180-second time limit."
                )

            with tempfile.TemporaryDirectory(prefix="openjarvis-pdf-ocr-") as temp:
                output_prefix = Path(temp) / "page"
                command = [
                    renderer,
                    "-f",
                    str(page_number),
                    "-l",
                    str(page_number),
                    "-singlefile",
                    "-png",
                    "-scale-to",
                    str(_PDF_PAGE_RENDER_SIZE),
                    str(path.expanduser().resolve()),
                    str(output_prefix),
                ]
                try:
                    rendered = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=min(remaining, _PDF_PAGE_RENDER_TIMEOUT_SECONDS),
                        check=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                except subprocess.TimeoutExpired as exc:
                    raise OcrProcessingError(
                        "Rendering a scanned PDF page exceeded its 20-second limit."
                    ) from exc
                except OSError as exc:
                    raise OcrProcessingError(
                        "The local Poppler PDF renderer could not be started."
                    ) from exc

                if rendered.returncode != 0:
                    raise OcrProcessingError(
                        f"Poppler could not render PDF page {page_number}."
                    )
                image_path = output_prefix.with_suffix(".png")
                if not image_path.is_file():
                    raise OcrProcessingError(
                        f"Poppler produced no image for PDF page {page_number}."
                    )

                remaining = _MAX_PDF_OCR_SECONDS - (
                    time.monotonic() - ocr_started
                )
                recognized = ocr_image(
                    image_path,
                    timeout_seconds=remaining,
                    language=language,
                )

            if recognized:
                remaining_chars = _MAX_PDF_OCR_TEXT_CHARS - ocr_text_chars
                if remaining_chars <= 0:
                    break
                if len(recognized) > remaining_chars:
                    marker = "\n[PDF OCR output truncated at 100,000 characters.]"
                    if remaining_chars > len(marker):
                        recognized = (
                            recognized[: remaining_chars - len(marker)] + marker
                        )
                    else:
                        recognized = recognized[:remaining_chars]
                    pages.append(recognized)
                    break
                ocr_text_chars += len(recognized)
                pages.append(recognized)
    return "\n\n".join(pages)


def _should_skip_dir(name: str) -> bool:
    """Return True if directory *name* should be skipped."""
    if name.startswith("."):
        return True
    if name in _SKIP_DIRS:
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def ingest_path(
    path: Path,
    *,
    config: Optional[ChunkConfig] = None,
    ocr: bool = False,
    transcribe_media: bool = False,
    speech_config=None,
) -> List[Chunk]:
    """Ingest a file or directory into chunks.

    If *path* is a file, reads and chunks it.
    If *path* is a directory, recursively walks it (skipping hidden and
    common non-content directories) and chunks each file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        text, _meta = read_document(
            path,
            ocr=ocr,
            transcribe_media=transcribe_media,
            speech_config=speech_config,
        )
        return chunk_text(text, source=str(path), config=config)

    # Directory: recursive walk
    all_chunks: List[Chunk] = []
    for child in sorted(path.rglob("*")):
        # Skip directories themselves — rglob yields files too
        if child.is_dir():
            continue

        # Check if any parent directory should be skipped
        rel = child.relative_to(path)
        skip = False
        for part in rel.parts[:-1]:
            if _should_skip_dir(part):
                skip = True
                break
        if skip:
            continue

        # Skip hidden files
        if child.name.startswith("."):
            continue

        # Skip sensitive files (secrets, credentials, keys)
        from openjarvis.security.file_policy import is_sensitive_file

        if is_sensitive_file(child):
            continue

        suffix = child.suffix.lower()
        if suffix in _IMAGE_EXTS and not ocr:
            continue
        if detect_media_type(child) and not transcribe_media:
            continue
        # Skip unsupported binary formats. Supported media is handled above
        # only when local transcription was explicitly enabled.
        if suffix in {
            ".gif",
            ".ico",
            ".webp",
            ".zip",
            ".tar",
            ".gz",
            ".bz2",
            ".7z",
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            ".o",
            ".pyc",
            ".pyo",
            ".class",
            ".wasm",
        }:
            continue

        try:
            text, _meta = read_document(
                child,
                ocr=ocr,
                transcribe_media=transcribe_media,
                speech_config=speech_config,
            )
            chunks = chunk_text(text, source=str(child), config=config)
            all_chunks.extend(chunks)
        except (ImportError, OSError):
            # Skip files we can't read (e.g. PDF without pdfplumber)
            continue

    return all_chunks


__all__ = ["DocumentMeta", "detect_file_type", "ingest_path", "read_document"]
