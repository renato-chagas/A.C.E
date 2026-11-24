"""Tests for Faster-Whisper speech backend."""

import sys
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from openjarvis.core.registry import SpeechRegistry
from openjarvis.speech.faster_whisper import FasterWhisperBackend


@pytest.fixture(autouse=True)
def _register_faster_whisper():
    """Re-register after any registry clear."""
    if not SpeechRegistry.contains("faster-whisper"):
        SpeechRegistry.register_value("faster-whisper", FasterWhisperBackend)


def test_faster_whisper_backend_registers():
    """Backend registers itself in SpeechRegistry."""
    assert SpeechRegistry.contains("faster-whisper")


def test_faster_whisper_transcribe():
    """Transcribe returns a TranscriptionResult."""
    from openjarvis.speech._stubs import TranscriptionResult

    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = " Hello world"
    mock_segment.start = 0.0
    mock_segment.end = 1.2
    mock_segment.avg_logprob = -0.3

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.95
    mock_info.duration = 1.5

    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    with patch(
        "openjarvis.speech.faster_whisper.WhisperModel",
        return_value=mock_model,
    ):
        from openjarvis.speech.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend(model_size="base", device="cpu")
        result = backend.transcribe(b"fake audio bytes")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration_seconds == 1.5


def test_faster_whisper_keeps_audio_in_memory_during_lazy_transcription():
    mock_info = SimpleNamespace(language="en", duration=1.5)
    seen = []

    def fake_transcribe(stream, *, language):
        assert isinstance(stream, BytesIO)
        assert language == "en"
        seen.append(stream)

        def segments():
            assert stream.read() == b"fake audio bytes"
            yield SimpleNamespace(text="Words", start=0.0, end=1.0)

        return segments(), mock_info

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = fake_transcribe
    with patch(
        "openjarvis.speech.faster_whisper.WhisperModel", return_value=mock_model
    ):
        backend = FasterWhisperBackend(model_size="base", device="cpu")
        result = backend.transcribe(b"fake audio bytes", language="en")

    assert result.text == "Words"
    assert len(seen) == 1 and seen[0].closed


def test_faster_whisper_closes_audio_stream_on_error():
    seen = []

    def fake_transcribe(stream, **_kwargs):
        seen.append(stream)
        raise RuntimeError("decode failed")

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = fake_transcribe
    with patch(
        "openjarvis.speech.faster_whisper.WhisperModel", return_value=mock_model
    ):
        backend = FasterWhisperBackend(model_size="base", device="cpu")
        with pytest.raises(RuntimeError, match="decode failed"):
            backend.transcribe(b"fake audio bytes")

    assert len(seen) == 1 and isinstance(seen[0], BytesIO) and seen[0].closed
    assert "decode failed" in (backend.last_error() or "")


def test_faster_whisper_falls_back_from_unsupported_float16():
    mock_model = MagicMock()

    with (
        patch(
            "openjarvis.speech.faster_whisper.WhisperModel",
            return_value=mock_model,
        ) as mock_whisper,
        patch(
            "openjarvis.speech.faster_whisper.ctranslate2",
            MagicMock(
                get_supported_compute_types=MagicMock(return_value={"float32", "int8"})
            ),
        ),
    ):
        backend = FasterWhisperBackend(
            model_size="base",
            device="cpu",
            compute_type="float16",
        )
        assert backend._ensure_model() is mock_model

    mock_whisper.assert_called_once_with("base", device="cpu", compute_type="int8")


@pytest.mark.parametrize(
    ("cuda_count", "device", "supported", "expected"),
    [
        (0, "cpu", {"float32", "int8"}, "int8"),
        (1, "cuda", {"float16", "int8_float16"}, "float16"),
    ],
)
def test_faster_whisper_auto_checks_compute_types_for_selected_device(
    cuda_count, device, supported, expected
):
    runtime = MagicMock()
    runtime.get_cuda_device_count.return_value = cuda_count
    runtime.get_supported_compute_types.return_value = supported
    with (
        patch("openjarvis.speech.faster_whisper.ctranslate2", runtime),
        patch("openjarvis.speech.faster_whisper.WhisperModel") as load,
    ):
        FasterWhisperBackend(device="auto")._ensure_model()

    runtime.get_supported_compute_types.assert_called_once_with(device)
    load.assert_called_once_with("base", device="auto", compute_type=expected)


def test_faster_whisper_only_resolves_model_in_local_cache(tmp_path, monkeypatch):
    for name in ("config.json", "model.bin", "tokenizer.json"):
        (tmp_path / name).touch()
    resolve = MagicMock(return_value=str(tmp_path))
    monkeypatch.setitem(
        sys.modules, "faster_whisper.utils", SimpleNamespace(download_model=resolve)
    )
    mock_model = MagicMock()

    with patch(
        "openjarvis.speech.faster_whisper.WhisperModel",
        return_value=mock_model,
    ) as mock_whisper:
        backend = FasterWhisperBackend(
            model_size="faster-whisper-model-id",
            device="cpu",
            local_files_only=True,
        )
        assert backend._ensure_model() is mock_model

    resolve.assert_called_once_with("faster-whisper-model-id", local_files_only=True)
    mock_whisper.assert_called_once_with(
        str(tmp_path.resolve()),
        device="cpu",
        compute_type="float16",
        local_files_only=True,
    )


def test_faster_whisper_offline_model_requires_local_tokenizer(tmp_path, monkeypatch):
    (tmp_path / "config.json").touch()
    (tmp_path / "model.bin").touch()
    resolve = MagicMock(side_effect=AssertionError("explicit local path"))
    monkeypatch.setitem(
        sys.modules, "faster_whisper.utils", SimpleNamespace(download_model=resolve)
    )

    with patch("openjarvis.speech.faster_whisper.WhisperModel") as mock_whisper:
        backend = FasterWhisperBackend(
            model_size=str(tmp_path),
            device="cpu",
            local_files_only=True,
        )
        assert not backend.health()

    assert "Missing: tokenizer.json." in (backend.last_error() or "")
    mock_whisper.assert_not_called()
    resolve.assert_not_called()


def test_faster_whisper_stops_at_requested_transcript_limit():
    mock_info = SimpleNamespace(
        language="en",
        language_probability=0.9,
        duration=20.0,
    )
    consumed = []

    def segments():
        consumed.append("first")
        yield SimpleNamespace(text="abcdef", start=0.0, end=1.0)
        consumed.append("second")
        yield SimpleNamespace(text="later speech", start=1.0, end=2.0)

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (segments(), mock_info)
    backend = FasterWhisperBackend()
    backend._model = mock_model

    result = backend.transcribe(b"audio", max_text_chars=4)

    assert result.text == "abcd"
    assert result.truncated is True
    assert [segment.text for segment in result.segments] == ["abcd"]
    assert consumed == ["first"]


def test_faster_whisper_missing_dependency_hint_documents_speech_extras():
    with patch("openjarvis.speech.faster_whisper.WhisperModel", new=None):
        backend = FasterWhisperBackend()

        with pytest.raises(ImportError) as excinfo:
            backend._ensure_model()

    assert "uv sync --extra speech" in str(excinfo.value)
    assert "--extra desktop" in str(excinfo.value)


def test_faster_whisper_health_no_model():
    """Health returns False before model is loaded."""
    with patch(
        "openjarvis.speech.faster_whisper.WhisperModel",
        new=None,
    ):
        backend = FasterWhisperBackend()
        assert backend.health() is False
        assert "uv sync --extra speech" in (backend.last_error() or "")


def test_faster_whisper_health_captures_load_error():
    with patch(
        "openjarvis.speech.faster_whisper.WhisperModel",
        side_effect=RuntimeError("missing cublas64_12.dll"),
    ):
        backend = FasterWhisperBackend()
        assert backend.health() is False
        assert "missing cublas64_12.dll" in (backend.last_error() or "")


def test_faster_whisper_supported_formats():
    """Backend supports standard audio formats."""
    with patch("openjarvis.speech.faster_whisper.WhisperModel"):
        from openjarvis.speech.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend.__new__(FasterWhisperBackend)
        formats = backend.supported_formats()
        assert "wav" in formats
        assert "mp3" in formats
        assert "webm" in formats
        assert "mp4" in formats
