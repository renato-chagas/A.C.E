"""Tests for bounded, offline audio/video transcription."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openjarvis.tools.storage import media


class _FakeContainer:
    def __init__(self, *, has_audio=True, duration=60_000_000):
        self.streams = [SimpleNamespace(type="audio")] if has_audio else []
        self.duration = duration

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _install_fake_av(monkeypatch, *, has_audio=True, duration=60_000_000):
    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(
            open=lambda _path: _FakeContainer(
                has_audio=has_audio,
                duration=duration,
            ),
            time_base=1_000_000,
        ),
    )


def test_detect_media_type_for_audio_and_video():
    assert media.detect_media_type("call.wav") == "audio"
    assert media.detect_media_type("clip.mp4") == "video"
    assert media.detect_media_type("notes.txt") is None


def test_transcribe_media_uses_cached_local_model_and_config(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "clip.mp4"
    source = b"video with audio track"
    path.write_bytes(source)
    _install_fake_av(monkeypatch)
    calls = {}

    class _Backend:
        def __init__(self, **kwargs):
            calls["backend"] = kwargs

        def health(self):
            return True

        def last_error(self):
            return None

        def transcribe(self, audio, **kwargs):
            calls["audio"] = audio
            calls["transcribe"] = kwargs
            return SimpleNamespace(text="Recognized speech", truncated=False)

    from openjarvis.speech import faster_whisper

    monkeypatch.setattr(faster_whisper, "FasterWhisperBackend", _Backend)
    speech_config = SimpleNamespace(
        model="small",
        device="cpu",
        compute_type="int8",
        language="pt",
    )

    result = media.transcribe_media_file(path, speech_config=speech_config)

    assert result == "Recognized speech"
    assert calls["backend"] == {
        "model_size": "small",
        "device": "cpu",
        "compute_type": "int8",
        "local_files_only": True,
    }
    assert calls["audio"] == source
    assert calls["transcribe"] == {
        "format": "mp4",
        "language": "pt",
        "max_text_chars": media._MAX_TRANSCRIPT_CHARS
        - len(media._TRUNCATION_MARKER),
    }


def test_transcribe_media_rejects_files_without_audio(tmp_path: Path, monkeypatch):
    path = tmp_path / "silent.mp4"
    path.write_bytes(b"video without audio")
    _install_fake_av(monkeypatch, has_audio=False)

    with pytest.raises(media.MediaProcessingError, match="no audio track"):
        media.transcribe_media_file(path, speech_config=SimpleNamespace())


def test_transcribe_media_rejects_long_files_before_loading_model(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "long.wav"
    path.write_bytes(b"audio")
    _install_fake_av(
        monkeypatch,
        duration=(media._MAX_MEDIA_SECONDS + 1) * 1_000_000,
    )

    with pytest.raises(ValueError, match="30 minutes"):
        media.transcribe_media_file(path, speech_config=SimpleNamespace())


def test_transcribe_media_requires_a_cached_local_model(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"audio")
    _install_fake_av(monkeypatch)

    class _Backend:
        def __init__(self, **_kwargs):
            pass

        def health(self):
            return False

        def last_error(self):
            return "model files not found"

    from openjarvis.speech import faster_whisper

    monkeypatch.setattr(faster_whisper, "FasterWhisperBackend", _Backend)

    with pytest.raises(media.MediaTranscriptionUnavailableError, match="not cached"):
        media.transcribe_media_file(path, speech_config=SimpleNamespace())


def test_transcribe_media_caps_file_size(tmp_path: Path, monkeypatch):
    path = tmp_path / "large.mp3"
    with path.open("wb") as stream:
        stream.truncate(media._MAX_MEDIA_BYTES + 1)

    with pytest.raises(ValueError, match="50 MiB"):
        media.transcribe_media_file(path, speech_config=SimpleNamespace())
