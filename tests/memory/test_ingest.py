"""Tests for document ingestion and file type detection."""

from __future__ import annotations

import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from openjarvis.tools.storage import ingest
from openjarvis.tools.storage.ingest import (
    detect_file_type,
    ingest_path,
    read_document,
)


class _FakePdfPage:
    def __init__(self, text=None, images=None):
        self.text = text
        self.images = images or []

    def extract_text(self):
        return self.text


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _mock_pdfplumber(monkeypatch, pages):
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda _path: _FakePdf(pages)),
    )


def test_detect_file_type_markdown(tmp_path: Path):
    p = tmp_path / "readme.md"
    assert detect_file_type(p) == "markdown"
    p2 = tmp_path / "doc.markdown"
    assert detect_file_type(p2) == "markdown"


def test_detect_file_type_python(tmp_path: Path):
    p = tmp_path / "main.py"
    assert detect_file_type(p) == "code"


def test_detect_file_type_pdf(tmp_path: Path):
    p = tmp_path / "paper.pdf"
    assert detect_file_type(p) == "pdf"


def test_detect_file_type_unknown(tmp_path: Path):
    p = tmp_path / "notes.txt"
    assert detect_file_type(p) == "text"
    p2 = tmp_path / "data.csv"
    assert detect_file_type(p2) == "text"


def test_detect_file_type_image(tmp_path: Path):
    assert detect_file_type(tmp_path / "scan.png") == "image"


def test_detect_file_type_audio_and_video(tmp_path: Path):
    assert detect_file_type(tmp_path / "meeting.wav") == "audio"
    assert detect_file_type(tmp_path / "recording.mp4") == "video"


def test_read_media_requires_explicit_local_transcription(tmp_path: Path):
    media = tmp_path / "meeting.wav"
    media.write_bytes(b"fake audio")

    with pytest.raises(ValueError, match="transcribe-media"):
        read_document(media)


def test_directory_media_indexing_is_opt_in_and_uses_local_transcription(
    tmp_path: Path, monkeypatch
):
    media = tmp_path / "meeting.mp4"
    media.write_bytes(b"fake video")
    extracted = " ".join(f"spoken{i}" for i in range(100))
    calls = []

    def fake_transcribe(path, *, speech_config=None):
        calls.append((path, speech_config))
        return extracted

    monkeypatch.setattr(ingest, "transcribe_media_file", fake_transcribe)

    assert ingest_path(tmp_path) == []
    chunks = ingest_path(tmp_path, transcribe_media=True, speech_config="local-config")

    assert chunks
    assert chunks[0].source == str(media)
    assert calls == [(media, "local-config")]


def test_read_image_requires_explicit_ocr(tmp_path: Path):
    image = tmp_path / "scan.png"
    image.write_bytes(b"image data")

    with pytest.raises(ValueError, match="requires OCR"):
        read_document(image)


def test_image_directory_ingestion_is_opt_in_and_uses_local_ocr(
    tmp_path: Path, monkeypatch
):
    image = tmp_path / "scan.png"
    image.write_bytes(b"image data")
    extracted = " ".join(f"recognized{i}" for i in range(100))
    calls = []

    def fake_ocr(path):
        calls.append(path)
        return extracted

    monkeypatch.setattr("openjarvis.tools.storage.ingest.ocr_image", fake_ocr)

    assert ingest_path(tmp_path) == []
    chunks = ingest_path(tmp_path, ocr=True)

    assert chunks
    assert chunks[0].source == str(image)
    assert len(calls) == 1


def test_read_image_records_ocr_text_metadata(tmp_path: Path, monkeypatch):
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"image data")
    monkeypatch.setattr(
        "openjarvis.tools.storage.ingest.ocr_image", lambda _path: "Recognized text"
    )

    text, metadata = read_document(image, ocr=True)

    assert text == "Recognized text"
    assert metadata.file_type == "image"


def test_read_document_text(tmp_path: Path):
    p = tmp_path / "hello.txt"
    p.write_text("Hello, world!\nLine two.", encoding="utf-8")
    text, meta = read_document(p)
    assert "Hello, world!" in text
    assert meta.file_type == "text"
    assert meta.line_count == 2
    assert meta.size_bytes > 0


def test_read_document_utf8_fallback(tmp_path: Path):
    p = tmp_path / "latin.txt"
    p.write_bytes(b"caf\xe9 au lait")
    text, meta = read_document(p)
    assert "caf" in text


def test_read_document_pdf_missing_dep(tmp_path: Path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake pdf content")
    # Should raise ImportError when pdfplumber not installed
    # or succeed if it IS installed — either way just check it's handled
    try:
        read_document(p)
    except ImportError as exc:
        assert "pdfplumber" in str(exc)


def test_scanned_pdf_ocr_renders_page_and_uses_local_ocr(
    tmp_path: Path, monkeypatch
):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    _mock_pdfplumber(monkeypatch, [_FakePdfPage(images=[{"name": "scan"}])])
    calls = []
    render_dir = tmp_path / "rendered"
    render_dir.mkdir()

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[-1] + ".png").write_bytes(b"rendered page")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        ingest.tempfile,
        "TemporaryDirectory",
        lambda **_kwargs: nullcontext(str(render_dir)),
    )
    monkeypatch.setattr(ingest.shutil, "which", lambda name: "/local/pdftoppm")
    monkeypatch.setattr(ingest, "select_ocr_language", lambda: "por+eng")
    monkeypatch.setattr(ingest.subprocess, "run", fake_run)
    monkeypatch.setattr(
        ingest,
        "ocr_image",
        lambda path, *, timeout_seconds, language: (
            "Texto do PDF escaneado"
            if Path(path).is_file() and language == "por+eng" and timeout_seconds > 0
            else ""
        ),
    )

    text, metadata = read_document(pdf, ocr=True)

    assert text == "Texto do PDF escaneado"
    assert metadata.file_type == "pdf"
    command, options = calls[0]
    assert command[0] == "/local/pdftoppm"
    assert command[command.index("-f") + 1] == "1"
    assert command[command.index("-l") + 1] == "1"
    assert command[command.index("-scale-to") + 1] == "1800"
    assert options["timeout"] <= ingest._PDF_PAGE_RENDER_TIMEOUT_SECONDS


def test_scanned_pdf_ocr_is_opt_in(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    _mock_pdfplumber(monkeypatch, [_FakePdfPage(images=[{"name": "scan"}])])

    text, _ = read_document(pdf)

    assert text == ""


def test_text_pdf_does_not_require_ocr_even_when_enabled(
    tmp_path: Path, monkeypatch
):
    pdf = tmp_path / "digital.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    _mock_pdfplumber(
        monkeypatch, [_FakePdfPage(text="Selectable PDF text", images=[{"logo": 1}])]
    )
    monkeypatch.setattr(
        ingest.shutil,
        "which",
        lambda _name: pytest.fail("OCR renderer should not be resolved"),
    )

    text, _ = read_document(pdf, ocr=True)

    assert text == "Selectable PDF text"


def test_read_document_not_found(tmp_path: Path):
    p = tmp_path / "nope.txt"
    with pytest.raises(FileNotFoundError):
        read_document(p)


def test_ingest_single_file(tmp_path: Path):
    p = tmp_path / "doc.txt"
    content = " ".join(f"word{i}" for i in range(100))
    p.write_text(content, encoding="utf-8")
    chunks = ingest_path(p)
    assert len(chunks) >= 1
    assert chunks[0].source == str(p)


def test_ingest_directory_recursive(tmp_path: Path):
    sub = tmp_path / "docs"
    sub.mkdir()
    (sub / "a.txt").write_text(
        " ".join(f"a{i}" for i in range(100)),
        encoding="utf-8",
    )
    (sub / "b.md").write_text(
        " ".join(f"b{i}" for i in range(100)),
        encoding="utf-8",
    )
    chunks = ingest_path(tmp_path)
    sources = {c.source for c in chunks}
    assert any("a.txt" in s for s in sources)
    assert any("b.md" in s for s in sources)


def test_ingest_skips_hidden_dirs(tmp_path: Path):
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.txt").write_text(
        " ".join(f"s{i}" for i in range(100)),
        encoding="utf-8",
    )
    (tmp_path / "visible.txt").write_text(
        " ".join(f"v{i}" for i in range(100)),
        encoding="utf-8",
    )
    chunks = ingest_path(tmp_path)
    sources = {c.source for c in chunks}
    assert not any(".hidden" in s for s in sources)


def test_ingest_skips_pycache(tmp_path: Path):
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "mod.cpython-310.pyc").write_bytes(b"\x00\x00")
    (tmp_path / "real.py").write_text(
        " ".join(f"r{i}" for i in range(100)),
        encoding="utf-8",
    )
    chunks = ingest_path(tmp_path)
    sources = {c.source for c in chunks}
    assert not any("__pycache__" in s for s in sources)


def test_ingest_nonexistent_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ingest_path(tmp_path / "nope")


def test_ingest_empty_dir(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    chunks = ingest_path(empty)
    assert chunks == []


def test_ingest_skips_env_files(tmp_path: Path):
    (tmp_path / ".env").write_text(
        " ".join(f"x{i}" for i in range(100)), encoding="utf-8"
    )
    (tmp_path / "readme.txt").write_text(
        " ".join(f"x{i}" for i in range(100)), encoding="utf-8"
    )
    chunks = ingest_path(tmp_path)
    sources = {c.source for c in chunks}
    assert not any(".env" in s for s in sources)
    assert any("readme.txt" in s for s in sources)


def test_ingest_skips_key_files(tmp_path: Path):
    (tmp_path / "server.key").write_text(
        " ".join(f"x{i}" for i in range(100)), encoding="utf-8"
    )
    (tmp_path / "notes.txt").write_text(
        " ".join(f"x{i}" for i in range(100)), encoding="utf-8"
    )
    chunks = ingest_path(tmp_path)
    sources = {c.source for c in chunks}
    assert not any("server.key" in s for s in sources)
    assert any("notes.txt" in s for s in sources)


def test_ingest_processes_normal_files(tmp_path: Path):
    (tmp_path / "app.py").write_text(
        " ".join(f"x{i}" for i in range(100)), encoding="utf-8"
    )
    (tmp_path / "doc.md").write_text(
        " ".join(f"x{i}" for i in range(100)), encoding="utf-8"
    )
    chunks = ingest_path(tmp_path)
    sources = {c.source for c in chunks}
    assert any("app.py" in s for s in sources)
    assert any("doc.md" in s for s in sources)
