"""Timing and local-device behavior for speech audio capture."""

from __future__ import annotations

import builtins
import struct
import sys
import types
from types import SimpleNamespace

import pytest

from openjarvis.speech import voice_io
from openjarvis.speech.voice_io import record_until_silence


class _FakeStream:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = iter(frames)
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def read(self, chunk: int):
        self.reads += 1
        return next(self.frames), False


def _install_audio(monkeypatch, stream: _FakeStream) -> None:
    fake_sd = SimpleNamespace(RawInputStream=lambda **kwargs: stream)
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)


class _FakeAudioBlock:
    size = 4
    ndim = 2

    def mean(self, axis: int):
        assert axis == 1
        return self

    def __mul__(self, _scale: float):
        return self

    def astype(self, dtype):
        assert dtype == "int16"
        return self

    def tobytes(self) -> bytes:
        return struct.pack("2h", 1000, -1000)


class _FakeRecorder:
    def __init__(self) -> None:
        self.closed = False
        self.record_args = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        self.closed = True

    def record(self, *, numframes: int):
        self.record_args = numframes
        return object()


def _force_windows_audio_path(monkeypatch) -> None:
    original_import = builtins.__import__

    def import_with_windows_os(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "os":
            return SimpleNamespace(name="nt")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_windows_os)


def _install_loopback_audio(monkeypatch, speaker, loopback) -> _FakeRecorder:
    _force_windows_audio_path(monkeypatch)
    recorder = _FakeRecorder()

    def open_recorder(**kwargs):
        assert kwargs == {
            "samplerate": 48000,
            "channels": 2,
            "blocksize": 2048,
        }
        return recorder

    loopback.recorder = open_recorder

    fake_soundcard = types.ModuleType("soundcard")
    fake_soundcard.default_speaker = lambda: speaker
    fake_soundcard.get_microphone = lambda *, id, include_loopback: (
        loopback if include_loopback and id == speaker.name else None
    )
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.asarray = lambda *_args, **_kwargs: _FakeAudioBlock()
    fake_numpy.clip = lambda block, *_args: block
    monkeypatch.setitem(sys.modules, "soundcard", fake_soundcard)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    return recorder


def test_initial_silence_stops_at_startup_timeout(monkeypatch) -> None:
    silence = bytes(1024 * 2)
    stream = _FakeStream([silence] * 30)
    _install_audio(monkeypatch, stream)

    record_until_silence(
        sample_rate=1024,
        startup_silence_seconds=2,
        max_seconds=30,
    )

    assert stream.reads == 2


def test_post_speech_uses_normal_silence_window(monkeypatch) -> None:
    speech = struct.pack("1024h", *([1000] * 1024))
    silence = bytes(1024 * 2)
    stream = _FakeStream([speech, silence, silence, silence])
    _install_audio(monkeypatch, stream)

    record_until_silence(
        sample_rate=1024,
        silence_seconds=2,
        startup_silence_seconds=1,
        max_seconds=30,
    )

    assert stream.reads == 3


def test_system_audio_uses_default_speaker_loopback_and_closes_recorder(
    monkeypatch,
) -> None:
    speaker = SimpleNamespace(name="Speakers", channels=2)
    loopback = SimpleNamespace(isloopback=True)
    recorder = _install_loopback_audio(monkeypatch, speaker, loopback)
    captured = {}

    def collect_first_chunk(chunks, **kwargs):
        captured.update(kwargs)
        iterator = iter(chunks)
        try:
            return next(iterator)
        finally:
            iterator.close()

    monkeypatch.setattr(voice_io, "_collect_speech_chunks", collect_first_chunk)

    audio = voice_io.record_loopback_speech_until_silence()

    assert audio == struct.pack("2h", 1000, -1000)
    assert captured["sample_rate"] == 48000
    assert captured["chunk_frames"] == 2048
    assert recorder.record_args == 2048
    assert recorder.closed


def test_system_audio_fails_closed_if_loopback_device_is_not_available(
    monkeypatch,
) -> None:
    speaker = SimpleNamespace(name="Speakers", channels=2)
    recorder = _install_loopback_audio(
        monkeypatch, speaker, SimpleNamespace(isloopback=False)
    )

    with pytest.raises(
        RuntimeError,
        match="Could not open the default speaker loopback",
    ):
        voice_io.record_loopback_speech_until_silence()

    assert recorder.record_args is None
    assert not recorder.closed


def test_system_audio_reports_missing_optional_dependencies(monkeypatch) -> None:
    _force_windows_audio_path(monkeypatch)
    original_import = builtins.__import__

    def import_without_soundcard(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"numpy", "soundcard"}:
            raise ImportError(f"{name} is not installed")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_soundcard)

    with pytest.raises(RuntimeError, match="audio-loopback"):
        voice_io.record_loopback_speech_until_silence()
