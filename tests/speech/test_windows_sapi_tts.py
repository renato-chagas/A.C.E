"""Tests for the optional offline Windows SAPI TTS adapter."""

from __future__ import annotations

import base64
import io
import json
import wave
from types import SimpleNamespace

import pytest

from openjarvis.speech import windows_sapi_tts
from openjarvis.speech.windows_sapi_tts import WindowsSapiTTSBackend


def _wav_bytes(sample_rate: int = 24_000, frames: int = 240) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\0\0" * frames)
    return buffer.getvalue()


def test_windows_sapi_registered_and_unavailable_without_powershell(monkeypatch):
    from openjarvis.core.registry import TTSRegistry

    monkeypatch.setattr(windows_sapi_tts, "_powershell_executable", lambda: None)
    if not TTSRegistry.contains("windows_sapi"):
        TTSRegistry.register_value("windows_sapi", WindowsSapiTTSBackend)

    assert TTSRegistry.contains("windows_sapi")
    assert WindowsSapiTTSBackend().health() is False


def test_available_voices_are_cached_and_normalized(monkeypatch):
    calls = []

    def run_sapi(request):
        calls.append(request)
        return json.dumps(
            [
                {"name": "Microsoft Maria", "culture": "pt-BR"},
                {"name": "Microsoft Zira", "culture": "en-US"},
            ]
        )

    monkeypatch.setattr(windows_sapi_tts, "_run_sapi", run_sapi)
    backend = WindowsSapiTTSBackend()

    assert backend.health() is True
    assert backend.available_voices() == ["Microsoft Maria", "Microsoft Zira"]
    assert calls == [{"operation": "voices"}]


def test_synthesize_prefers_installed_brazilian_portuguese_voice(monkeypatch):
    voices = json.dumps(
        [
            {"name": "English voice", "culture": "en-US"},
            {"name": "Maria", "culture": "pt-BR"},
        ]
    )
    requests = []

    def run_sapi(request, **kwargs):
        requests.append((request, kwargs))
        if request["operation"] == "voices":
            return voices
        return base64.b64encode(_wav_bytes()).decode("ascii")

    monkeypatch.setattr(windows_sapi_tts, "_run_sapi", run_sapi)
    result = WindowsSapiTTSBackend().synthesize("Olá, ACE.", speed=1.0)

    assert requests[1][0] == {
        "operation": "synthesize",
        "text": "Olá, ACE.",
        "voice_id": "",
        "preferred_voice": "Maria",
        "rate": 0,
    }
    assert requests[1][1] == {"timeout": windows_sapi_tts._PROCESS_TIMEOUT_SECONDS}
    assert result.format == "wav"
    assert result.sample_rate == 24_000
    assert result.duration_seconds == pytest.approx(0.01)
    assert result.voice_id == "Maria"
    assert result.metadata == {"backend": "windows_sapi", "offline": True}


def test_synthesize_uses_explicit_installed_voice_and_maps_speed(monkeypatch):
    calls = []

    def run_sapi(request, **kwargs):
        calls.append(request)
        if request["operation"] == "voices":
            return '[{"name":"Zira","culture":"en-US"}]'
        return base64.b64encode(_wav_bytes()).decode("ascii")

    monkeypatch.setattr(windows_sapi_tts, "_run_sapi", run_sapi)
    result = WindowsSapiTTSBackend().synthesize("Hello.", voice_id="Zira", speed=2.0)

    assert calls[1]["voice_id"] == "Zira"
    assert calls[1]["preferred_voice"] == ""
    assert calls[1]["rate"] == 10
    assert result.voice_id == "Zira"


def test_play_audio_uses_windows_player_and_bounded_timeout(monkeypatch):
    calls = []
    audio = _wav_bytes(sample_rate=24_000, frames=24_000)

    def play_sapi(audio_base64, *, timeout):
        calls.append((audio_base64, timeout))

    monkeypatch.setattr(windows_sapi_tts, "_play_sapi_audio", play_sapi)

    WindowsSapiTTSBackend().play_audio(audio)

    assert base64.b64decode(calls[0][0], validate=True) == audio
    assert calls[0][1] == pytest.approx(11.0)


def test_play_audio_rejects_invalid_or_oversized_data(monkeypatch):
    backend = WindowsSapiTTSBackend()
    calls = []
    monkeypatch.setattr(
        windows_sapi_tts,
        "_play_sapi_audio",
        lambda *a, **k: calls.append(a),
    )

    with pytest.raises(ValueError, match="valid WAV"):
        backend.play_audio(b"not a wave")
    with pytest.raises(ValueError, match="exceeds"):
        backend.play_audio(b"x" * (windows_sapi_tts._MAX_PLAYBACK_BYTES + 1))

    assert calls == []


def test_synthesize_rejects_invalid_or_oversized_input():
    backend = WindowsSapiTTSBackend()

    with pytest.raises(ValueError, match="WAV output only"):
        backend.synthesize("Hello", output_format="mp3")
    with pytest.raises(ValueError, match="must not be empty"):
        backend.synthesize("  ")
    with pytest.raises(ValueError, match="finite value"):
        backend.synthesize("Hello", speed=0)
    with pytest.raises(ValueError, match="exceeds"):
        backend.synthesize("x" * (windows_sapi_tts._MAX_TEXT_CHARS + 1))


def test_run_sapi_passes_text_as_base64_json_not_shell_source(monkeypatch):
    request = {"operation": "synthesize", "text": "hi'; Remove-Item *; '"}

    def fake_run(args, **kwargs):
        payload = base64.b64decode(kwargs["input"], validate=True)
        assert json.loads(payload) == request
        assert args[-2] == "-EncodedCommand"
        assert kwargs["capture_output"] is True
        assert "shell" not in kwargs
        return SimpleNamespace(returncode=0, stdout=b"YWJj", stderr=b"")

    monkeypatch.setattr(
        windows_sapi_tts, "_powershell_executable", lambda: "powershell.exe"
    )
    monkeypatch.setattr(windows_sapi_tts.subprocess, "run", fake_run)

    assert windows_sapi_tts._run_sapi(request) == "YWJj"


def test_playback_uses_static_powershell_and_json_stdin(monkeypatch):
    captured = {}

    class FakeProcess:
        returncode = 0

        def communicate(self, payload, *, timeout):
            captured["request"] = json.loads(base64.b64decode(payload, validate=True))
            captured["timeout"] = timeout
            return b"", b""

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        windows_sapi_tts, "_powershell_executable", lambda: "powershell.exe"
    )
    monkeypatch.setattr(windows_sapi_tts.subprocess, "Popen", fake_popen)

    windows_sapi_tts._play_sapi_audio("YWJj", timeout=12.5)

    assert captured["request"] == {"operation": "play", "audio": "YWJj"}
    assert captured["timeout"] == 12.5
    assert captured["args"][-2] == "-EncodedCommand"
    assert captured["kwargs"]["stdin"] == windows_sapi_tts.subprocess.PIPE
    assert captured["kwargs"]["stdout"] == windows_sapi_tts.subprocess.DEVNULL
    assert "shell" not in captured["kwargs"]


def test_playback_terminates_child_when_interrupted(monkeypatch):
    class FakeProcess:
        returncode = None

        def __init__(self):
            self.killed = False
            self.calls = 0

        def communicate(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            return None, b""

        def kill(self):
            self.killed = True

    process = FakeProcess()
    monkeypatch.setattr(
        windows_sapi_tts, "_powershell_executable", lambda: "powershell.exe"
    )
    monkeypatch.setattr(
        windows_sapi_tts.subprocess, "Popen", lambda *_args, **_kwargs: process
    )

    with pytest.raises(KeyboardInterrupt):
        windows_sapi_tts._play_sapi_audio("YWJj", timeout=10)

    assert process.killed is True
    assert process.calls == 2


def test_offline_resolution_uses_sapi_after_kokoro_fails(monkeypatch):
    from openjarvis.core.registry import TTSRegistry
    from openjarvis.speech._tts_discovery import get_tts_backend

    checked = []

    class Backend:
        def __init__(self, backend_id: str) -> None:
            self.backend_id = backend_id

        def health(self) -> bool:
            checked.append(self.backend_id)
            return self.backend_id == "windows_sapi"

    monkeypatch.setattr(TTSRegistry, "contains", lambda _key: True)
    monkeypatch.setattr(
        TTSRegistry,
        "get",
        lambda key: lambda: Backend(key),
    )

    backend = get_tts_backend("openai_tts", offline_only=True)

    assert backend.backend_id == "windows_sapi"
    assert checked == ["kokoro", "windows_sapi"]
