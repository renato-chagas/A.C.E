"""Offline Windows text-to-speech through the built-in System.Speech voices.

This adapter uses Windows PowerShell and the .NET ``System.Speech`` assembly;
it has no Python package or network dependency. Audio is synthesized to a WAV
stream in memory and returned through the regular TTS backend interface.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import shutil
import subprocess
import threading
import wave
from typing import Any, List

from openjarvis.core.registry import TTSRegistry
from openjarvis.speech.tts import TTSBackend, TTSResult

logger = logging.getLogger(__name__)

_POWERSHELL_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $OutputEncoding
$synth = $null
$stream = $null
$player = $null
try {
    $encoded = [Console]::In.ReadToEnd().Trim()
    $json = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($encoded))
    $request = ConvertFrom-Json -InputObject $json
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

    if ($request.operation -eq 'voices') {
        $records = @($synth.GetInstalledVoices() |
            Where-Object { $_.Enabled } |
            ForEach-Object {
                @{ name = $_.VoiceInfo.Name; culture = $_.VoiceInfo.Culture.Name }
            })
        [Console]::Out.Write((ConvertTo-Json -InputObject $records -Compress))
    } elseif ($request.operation -eq 'synthesize') {
        if ($request.voice_id) {
            $synth.SelectVoice([string]$request.voice_id)
        } elseif ($request.preferred_voice) {
            $synth.SelectVoice([string]$request.preferred_voice)
        }
        $synth.Rate = [Math]::Max(-10, [Math]::Min(10, [int]$request.rate))
        $stream = New-Object System.IO.MemoryStream
        $synth.SetOutputToWaveStream($stream)
        $synth.Speak([string]$request.text)
        $synth.SetOutputToNull()
        [Console]::Out.Write([Convert]::ToBase64String($stream.ToArray()))
        $stream.Dispose()
    } elseif ($request.operation -eq 'play') {
        Add-Type -AssemblyName System
        $audio = [Convert]::FromBase64String([string]$request.audio)
        $stream = New-Object System.IO.MemoryStream
        $stream.Write($audio, 0, $audio.Length)
        $stream.Position = 0
        $player = [System.Media.SoundPlayer]::new($stream)
        $player.PlaySync()
    } else {
        throw 'Unsupported speech operation.'
    }
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
} finally {
    if ($null -ne $player) { $player.Dispose() }
    if ($null -ne $stream) { $stream.Dispose() }
    if ($null -ne $synth) { $synth.Dispose() }
}
"""
_POWERSHELL_COMMAND = base64.b64encode(_POWERSHELL_SCRIPT.encode("utf-16le")).decode(
    "ascii"
)
_MAX_TEXT_CHARS = 20_000
_MAX_PLAYBACK_BYTES = 64 * 1024 * 1024
_MAX_PLAYBACK_SECONDS = 30 * 60
_PROCESS_TIMEOUT_SECONDS = 120


def _powershell_executable() -> str | None:
    if os.name != "nt":
        return None
    return shutil.which("powershell.exe") or shutil.which("powershell")


def _run_sapi(request: dict[str, Any], *, timeout: float = 8.0) -> str:
    executable = _powershell_executable()
    if executable is None:
        raise RuntimeError("Windows PowerShell is unavailable.")

    payload = base64.b64encode(
        json.dumps(request, ensure_ascii=True).encode("utf-8")
    ).decode("ascii")
    try:
        result = subprocess.run(
            [
                executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                _POWERSHELL_COMMAND,
            ],
            input=payload.encode("ascii"),
            capture_output=True,
            timeout=timeout,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Windows speech synthesis timed out.") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not start Windows PowerShell: {exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[:500]
        detail = detail or "System.Speech returned an error."
        raise RuntimeError(detail)
    try:
        return result.stdout.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeError("Windows PowerShell returned invalid UTF-8 output.") from exc


def _play_sapi_audio(audio_base64: str, *, timeout: float) -> None:
    """Play in-memory WAV data and terminate playback promptly on Ctrl+C."""
    executable = _powershell_executable()
    if executable is None:
        raise RuntimeError("Windows PowerShell is unavailable.")
    payload = base64.b64encode(
        json.dumps(
            {"operation": "play", "audio": audio_base64}, ensure_ascii=True
        ).encode("utf-8")
    ).decode("ascii")
    try:
        process = subprocess.Popen(
            [
                executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                _POWERSHELL_COMMAND,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        raise RuntimeError(f"Could not start Windows audio playback: {exc}") from exc

    try:
        _stdout, stderr = process.communicate(payload.encode("ascii"), timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.communicate()
        raise RuntimeError("Windows audio playback timed out.") from exc
    except KeyboardInterrupt:
        try:
            process.kill()
        except OSError:
            pass
        process.communicate()
        raise

    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()[:500]
        detail = detail or "Windows audio playback returned an error."
        raise RuntimeError(detail)


@TTSRegistry.register("windows_sapi")
class WindowsSapiTTSBackend(TTSBackend):
    """Synthesize speech offline with voices installed in Windows."""

    backend_id = "windows_sapi"

    def __init__(self) -> None:
        self._voices: list[dict[str, str]] | None = None
        self._voice_lock = threading.Lock()

    def _installed_voices(self) -> list[dict[str, str]]:
        with self._voice_lock:
            if self._voices is None:
                raw = _run_sapi({"operation": "voices"})
                try:
                    records = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "Windows returned an invalid voice list."
                    ) from exc
                if isinstance(records, dict):
                    records = [records]
                if not isinstance(records, list):
                    raise RuntimeError("Windows returned an invalid voice list.")
                self._voices = [
                    {
                        "name": str(item["name"]),
                        "culture": str(item.get("culture", "")),
                    }
                    for item in records
                    if isinstance(item, dict) and item.get("name")
                ]
            return list(self._voices)

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str = "",
        speed: float = 1.0,
        output_format: str = "wav",
    ) -> TTSResult:
        if output_format.lower() != "wav":
            raise ValueError("Windows SAPI currently supports WAV output only.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Speech text must not be empty.")
        if len(text) > _MAX_TEXT_CHARS:
            raise ValueError(f"Speech text exceeds {_MAX_TEXT_CHARS} characters.")
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Speech speed must be a finite value greater than zero.")

        voices = self._installed_voices()
        names = {voice["name"] for voice in voices}
        if voice_id and voice_id not in names:
            raise ValueError(f"Windows SAPI voice is not installed: {voice_id}")
        preferred_voice = next(
            (voice["name"] for voice in voices if voice["culture"].lower() == "pt-br"),
            "",
        )
        rate = max(-10, min(10, round(math.log2(speed) * 10)))
        raw = _run_sapi(
            {
                "operation": "synthesize",
                "text": text,
                "voice_id": voice_id,
                "preferred_voice": preferred_voice,
                "rate": rate,
            },
            timeout=_PROCESS_TIMEOUT_SECONDS,
        )
        try:
            audio = base64.b64decode(raw, validate=True)
            with wave.open(io.BytesIO(audio), "rb") as wav:
                sample_rate = wav.getframerate()
                duration = wav.getnframes() / sample_rate
        except (ValueError, wave.Error, ZeroDivisionError) as exc:
            raise RuntimeError("Windows SAPI returned invalid WAV audio.") from exc
        return TTSResult(
            audio=audio,
            format="wav",
            voice_id=voice_id or preferred_voice,
            sample_rate=sample_rate,
            duration_seconds=duration,
            metadata={"backend": self.backend_id, "offline": True},
        )

    def play_audio(self, audio: bytes) -> None:
        """Play synthesized PCM WAV through Windows without Python audio packages."""
        if not isinstance(audio, bytes) or not audio:
            raise ValueError("Windows SAPI playback requires non-empty WAV bytes.")
        if len(audio) > _MAX_PLAYBACK_BYTES:
            raise ValueError(
                f"Windows SAPI playback exceeds {_MAX_PLAYBACK_BYTES} bytes."
            )
        try:
            with wave.open(io.BytesIO(audio), "rb") as wav:
                if wav.getcomptype() != "NONE":
                    raise ValueError(
                        "Windows SAPI playback supports uncompressed WAV only."
                    )
                duration = wav.getnframes() / wav.getframerate()
        except (wave.Error, ZeroDivisionError) as exc:
            raise ValueError(
                "Windows SAPI playback requires a valid WAV file."
            ) from exc
        if duration > _MAX_PLAYBACK_SECONDS:
            raise ValueError(
                f"Windows SAPI playback exceeds {_MAX_PLAYBACK_SECONDS} seconds."
            )

        _play_sapi_audio(
            base64.b64encode(audio).decode("ascii"),
            timeout=max(8.0, duration + 10.0),
        )

    def available_voices(self) -> List[str]:
        try:
            return [voice["name"] for voice in self._installed_voices()]
        except RuntimeError:
            return []

    def health(self) -> bool:
        try:
            return bool(self._installed_voices())
        except Exception:
            logger.debug("Windows SAPI TTS is unavailable", exc_info=True)
            return False


__all__ = ["WindowsSapiTTSBackend"]
