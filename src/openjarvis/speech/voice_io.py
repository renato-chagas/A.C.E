"""Microphone recording with silence detection and audio playback helpers."""

from __future__ import annotations

import io
import wave
from collections import deque
from collections.abc import Iterable

_SAMPLE_RATE = 16000
_CHANNELS = 1
_CHUNK = 1024
_SILENCE_THRESHOLD = 500  # RMS below this → silence
_SILENCE_SECONDS = 1.5  # seconds of silence before auto-stop
_STARTUP_SILENCE_SECONDS = 5.0  # give up early if speech never begins
_MAX_RECORD_SECONDS = 30  # safety ceiling


def _rms(data: bytes) -> float:
    """Compute RMS amplitude of 16-bit PCM bytes."""
    import struct

    n = len(data) // 2
    if n == 0:
        return 0.0
    shorts = struct.unpack(f"{n}h", data[: n * 2])
    return (sum(s * s for s in shorts) / n) ** 0.5


def record_until_silence(
    *,
    sample_rate: int = _SAMPLE_RATE,
    silence_threshold: int = _SILENCE_THRESHOLD,
    silence_seconds: float = _SILENCE_SECONDS,
    startup_silence_seconds: float = _STARTUP_SILENCE_SECONDS,
    max_seconds: float = _MAX_RECORD_SECONDS,
) -> bytes:
    """Record from the default microphone until silence is detected.

    Returns raw WAV bytes (16-bit mono).
    Raises RuntimeError if sounddevice is not installed.
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError(
            "sounddevice is required for voice input. "
            "Install with: pip install sounddevice"
        )

    chunks_per_second = sample_rate / _CHUNK
    silence_chunks = int(silence_seconds * chunks_per_second)
    startup_silence_chunks = max(1, int(startup_silence_seconds * chunks_per_second))
    max_chunks = int(max_seconds * chunks_per_second)

    frames: list[bytes] = []
    silence_count = 0
    has_speech = False

    with sd.RawInputStream(
        samplerate=sample_rate,
        channels=_CHANNELS,
        dtype="int16",
        blocksize=_CHUNK,
    ) as stream:
        for _ in range(max_chunks):
            raw, _ = stream.read(_CHUNK)
            data = bytes(raw)
            frames.append(data)

            amplitude = _rms(data)
            if amplitude > silence_threshold:
                has_speech = True
                silence_count = 0
            elif not has_speech and len(frames) >= startup_silence_chunks:
                break
            elif has_speech:
                silence_count += 1
                if silence_count >= silence_chunks:
                    break

    return _frames_to_wav(frames, sample_rate)


def record_speech_until_silence(
    *,
    sample_rate: int = _SAMPLE_RATE,
    silence_threshold: int = _SILENCE_THRESHOLD,
    silence_seconds: float = _SILENCE_SECONDS,
    max_phrase_seconds: float = _MAX_RECORD_SECONDS,
    preroll_seconds: float = 0.4,
) -> bytes:
    """Wait for speech, record one phrase, and stop after silence.

    This function keeps the microphone open while hands-free mode is active.
    Before voice activity is detected, it retains only a short in-memory
    pre-roll. It never writes audio to disk.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is required for hands-free voice input. "
            "Install with: pip install 'OpenJarvis[voice]'"
        ) from exc

    def chunks() -> Iterable[bytes]:
        with sd.RawInputStream(
            samplerate=sample_rate,
            channels=_CHANNELS,
            dtype="int16",
            blocksize=_CHUNK,
        ) as stream:
            while True:
                raw, _overflowed = stream.read(_CHUNK)
                yield bytes(raw)

    return _collect_speech_chunks(
        chunks(),
        sample_rate=sample_rate,
        silence_threshold=silence_threshold,
        silence_seconds=silence_seconds,
        max_phrase_seconds=max_phrase_seconds,
        preroll_seconds=preroll_seconds,
        chunk_frames=_CHUNK,
    )


def record_loopback_speech_until_silence(
    *,
    sample_rate: int = 48000,
    silence_threshold: int = 500,
    silence_seconds: float = _SILENCE_SECONDS,
    max_phrase_seconds: float = _MAX_RECORD_SECONDS,
    preroll_seconds: float = 0.4,
) -> bytes:
    """Capture one phrase from the default Windows speaker output.

    Sound is retained only in memory here. The local speech backend may create
    a short-lived temporary audio file while transcribing the returned WAV.
    """
    import os

    if os.name != "nt":
        raise RuntimeError("System audio loopback is currently supported on Windows.")
    try:
        import numpy as np
        import soundcard as sc
    except ImportError as exc:
        raise RuntimeError(
            "System audio capture needs the optional dependency. Install with: "
            "pip install 'OpenJarvis[audio-loopback]'"
        ) from exc

    try:
        speaker = sc.default_speaker()
        if speaker is None:
            raise RuntimeError("No default audio output device is available.")
        loopback = sc.get_microphone(
            id=str(speaker.name),
            include_loopback=True,
        )
        if not getattr(loopback, "isloopback", False):
            raise RuntimeError("Could not open the default speaker loopback device.")
        available_channels = getattr(speaker, "channels", 2)
        channel_count = (
            int(available_channels)
            if isinstance(available_channels, int)
            else len(available_channels)
        )
        channel_count = max(1, min(2, channel_count))
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not open system audio loopback: {exc}") from exc

    blocksize = 2048

    def chunks() -> Iterable[bytes]:
        with loopback.recorder(
            samplerate=sample_rate,
            channels=channel_count,
            blocksize=blocksize,
        ) as recorder:
            while True:
                block = np.asarray(
                    recorder.record(numframes=blocksize), dtype="float32"
                )
                if block.size == 0:
                    continue
                mono = block if block.ndim == 1 else block.mean(axis=1)
                pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("int16")
                yield pcm.tobytes()

    return _collect_speech_chunks(
        chunks(),
        sample_rate=sample_rate,
        silence_threshold=silence_threshold,
        silence_seconds=silence_seconds,
        max_phrase_seconds=max_phrase_seconds,
        preroll_seconds=preroll_seconds,
        chunk_frames=blocksize,
    )


def _collect_speech_chunks(
    chunks: Iterable[bytes],
    *,
    sample_rate: int,
    silence_threshold: int,
    silence_seconds: float,
    max_phrase_seconds: float,
    preroll_seconds: float,
    chunk_frames: int,
) -> bytes:
    chunks_per_second = sample_rate / chunk_frames
    silence_chunks = max(1, int(silence_seconds * chunks_per_second))
    preroll_chunks = max(1, int(preroll_seconds * chunks_per_second))
    max_phrase_chunks = max(1, int(max_phrase_seconds * chunks_per_second))
    preroll: deque[bytes] = deque(maxlen=preroll_chunks)
    frames: list[bytes] = []
    has_speech = False
    silence_count = 0
    iterator = iter(chunks)
    try:
        for data in iterator:
            amplitude = _rms(data)
            if not has_speech:
                preroll.append(data)
                if amplitude > silence_threshold:
                    has_speech = True
                    frames.extend(preroll)
                continue

            frames.append(data)
            if amplitude > silence_threshold:
                silence_count = 0
            else:
                silence_count += 1
            if silence_count >= silence_chunks or len(frames) >= max_phrase_chunks:
                break
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()

    return _frames_to_wav(frames, sample_rate) if frames else b""


def _frames_to_wav(frames: list[bytes], sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def play_wav(audio: bytes, sample_rate: int = 24000) -> None:
    """Play raw WAV bytes through the default output device.

    If the bytes are a valid WAV file, sample rate is read from the header;
    otherwise ``sample_rate`` is used as a fallback.
    """
    try:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            "sounddevice, numpy, and soundfile are required for voice output. "
            "Install with: pip install sounddevice numpy soundfile"
        )

    buf = io.BytesIO(audio)
    try:
        data, sr = sf.read(buf, dtype="float32")
    except Exception:
        # Fall back: treat as raw PCM
        import struct

        n = len(audio) // 2
        data = (
            np.array(struct.unpack(f"{n}h", audio[: n * 2]), dtype="float32") / 32768.0
        )
        sr = sample_rate

    sd.play(data, sr)
    sd.wait()


__all__ = [
    "play_wav",
    "record_loopback_speech_until_silence",
    "record_speech_until_silence",
    "record_until_silence",
]
