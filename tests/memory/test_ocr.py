"""Tests for optional offline Tesseract OCR."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from openjarvis.tools.storage import ocr


def test_ocr_image_uses_local_tesseract_and_installed_languages(
    tmp_path: Path, monkeypatch
) -> None:
    image = tmp_path / "note image.png"
    image.write_bytes(b"fake image")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "--list-langs":
            return subprocess.CompletedProcess(
                command, 0, stdout="Languages:\neng\npor\nosd\n", stderr=""
            )
        return subprocess.CompletedProcess(
            command, 0, stdout=b"  Texto reconhecido localmente.\r\n", stderr=b""
        )

    monkeypatch.setattr(ocr.shutil, "which", lambda name: "/local/tesseract")
    monkeypatch.setattr(ocr.subprocess, "run", fake_run)

    text = ocr.ocr_image(image)

    assert text == "Texto reconhecido localmente."
    assert calls[0][0] == ["/local/tesseract", "--list-langs"]
    command, options = calls[1]
    assert command[0] == "/local/tesseract"
    assert command[1] == str(image.resolve())
    assert command[2] == "stdout"
    assert command[command.index("-l") + 1] == "por+eng"
    assert options["timeout"] == ocr._OCR_TIMEOUT_SECONDS
    assert options["check"] is False


def test_ocr_falls_back_to_an_installed_language(monkeypatch) -> None:
    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout="Languages:\neng\nosd\n", stderr=""
        )

    monkeypatch.setattr(ocr.subprocess, "run", fake_run)

    assert ocr._select_language("/local/tesseract") == "eng"


def test_ocr_requires_tesseract_on_path(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "note.png"
    image.write_bytes(b"fake image")
    monkeypatch.setattr(ocr.shutil, "which", lambda _name: None)

    with pytest.raises(ocr.OcrUnavailableError, match="Tesseract"):
        ocr.ocr_image(image)


def test_ocr_timeout_is_reported(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "note.png"
    image.write_bytes(b"fake image")
    calls = 0

    def fake_run(command, **_kwargs):
        nonlocal calls
        calls += 1
        if command[1] == "--list-langs":
            return subprocess.CompletedProcess(
                command, 0, stdout="Languages:\neng\n", stderr=""
            )
        raise subprocess.TimeoutExpired(command, 30)

    monkeypatch.setattr(ocr.shutil, "which", lambda _name: "/local/tesseract")
    monkeypatch.setattr(ocr.subprocess, "run", fake_run)

    with pytest.raises(ocr.OcrProcessingError, match="30-second limit"):
        ocr.ocr_image(image)
    assert calls == 2


def test_screen_ocr_pipes_png_to_local_tesseract(monkeypatch) -> None:
    image = b"\x89PNG\r\n\x1a\nfixture"
    calls = []

    def fake_run(command, **options):
        calls.append((command, options))
        if command[1] == "--list-langs":
            return subprocess.CompletedProcess(
                command, 0, stdout="Languages:\neng\npor\n", stderr=""
            )
        return subprocess.CompletedProcess(
            command, 0, stdout=b"  Text from the screen.  ", stderr=b""
        )

    monkeypatch.setattr(ocr.shutil, "which", lambda _name: "/local/tesseract")
    monkeypatch.setattr(ocr.subprocess, "run", fake_run)

    recognize = ocr.screen_ocr_reader()
    assert len(calls) == 1
    assert recognize(image) == "Text from the screen."
    command, options = calls[1]
    assert command == [
        "/local/tesseract",
        "stdin",
        "stdout",
        "--psm",
        "11",
        "-l",
        "por+eng",
    ]
    assert options["input"] == image
    assert options["timeout"] == ocr._OCR_TIMEOUT_SECONDS
    assert options.get("shell", False) is False


def test_screen_ocr_uses_native_windows_without_tesseract(monkeypatch) -> None:
    calls = []
    image = b"\x89PNG\r\n\x1a\nfixture"

    def prepare_windows():
        calls.append("prepared")

        def recognize(payload):
            calls.append(payload)
            return "  Local Windows OCR.  "

        return recognize

    monkeypatch.setattr(ocr.sys, "platform", "win32")
    monkeypatch.setattr(ocr.shutil, "which", lambda _name: None)
    monkeypatch.setitem(
        ocr.sys.modules,
        "openjarvis.tools.storage._windows_ocr",
        SimpleNamespace(windows_ocr_reader=prepare_windows),
    )

    recognize = ocr.screen_ocr_reader()
    with pytest.raises(ValueError, match="PNG"):
        recognize(b"not a png")
    monkeypatch.setattr(ocr, "_MAX_IMAGE_BYTES", len(image))
    with pytest.raises(ValueError, match="is limited to"):
        recognize(image + b"too big")
    assert recognize(image) == "Local Windows OCR."
    assert calls == ["prepared", image]
