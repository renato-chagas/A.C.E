"""Voice input/output helpers for the interactive chat session."""

from __future__ import annotations

import re
from typing import Any, Optional

from rich.markup import escape

VOICE_EXIT = object()

# Backend selection is shared with the API server and lives in
# openjarvis.speech._tts_discovery. Importing that at module level would pull
# the whole speech stack -- numpy included -- into every `jarvis` invocation,
# which tests/cli/test_cli.py guards against, so these aliases resolve lazily.
_LAZY_TTS_NAMES = {
    "_TTS_BACKEND_ORDER": "TTS_BACKEND_ORDER",
    "_BACKEND_DEFAULT_VOICE": "BACKEND_DEFAULT_VOICE",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_TTS_NAMES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from openjarvis.speech import _tts_discovery

    return getattr(_tts_discovery, target)


def _terminal_safe_text(value: object) -> str:
    """Remove terminal controls and escape Rich markup from dynamic text."""
    printable = "".join(
        char for char in str(value) if char in ("\n", "\t") or char.isprintable()
    )
    return escape(printable)


class VoiceSession:
    """Cache healthy speech backends for one interactive chat session."""

    def __init__(self, config: object | None = None) -> None:
        self._config = config
        self._stt_resolved = False
        self._stt_backend: Any = None
        self._tts_backend: Any = None
        self._tts_attempted: set[str] = set()
        self._voice_prefs: tuple[str, str, float] | None = None
        self._voice_warned: set[str] = set()

    def get_stt_backend(self) -> Any:
        """Resolve and health-check STT once, then reuse the loaded backend."""
        if not self._stt_resolved:
            from openjarvis.core.config import load_config
            from openjarvis.speech._discovery import get_speech_backend

            config = self._config if self._config is not None else load_config()
            self._stt_backend = get_speech_backend(config)
            self._stt_resolved = True
        return self._stt_backend

    def get_stt_language(self) -> Optional[str]:
        """Return the configured ``speech.language``, or None to auto-detect."""
        from openjarvis.core.config import load_config

        config = self._config if self._config is not None else load_config()
        return (getattr(config.speech, "language", "") or "").strip() or None

    def get_tts_backend(self) -> Any:
        """Return a cached healthy TTS backend, falling through once per key."""
        if self._tts_backend is not None:
            return self._tts_backend

        from openjarvis.speech._tts_discovery import get_tts_backend

        preferred, _, _ = self.get_voice_preferences()
        from openjarvis.core.config import load_config

        config = self._config if self._config is not None else load_config()
        offline_only = bool(
            getattr(getattr(config, "speech", None), "offline_only", False)
        )
        self._tts_backend = get_tts_backend(
            preferred,
            attempted=self._tts_attempted,
            offline_only=offline_only,
        )
        return self._tts_backend

    def get_voice_preferences(self) -> tuple[str, str, float]:
        """Resolve configured (tts_backend, voice_id, speed), cached per session."""
        if self._voice_prefs is None:
            from openjarvis.core.config import load_config
            from openjarvis.speech._tts_discovery import voice_preferences

            config = self._config if self._config is not None else load_config()
            self._voice_prefs = voice_preferences(config)
        return self._voice_prefs

    def voice_for_backend(self, backend: Any, console: Any = None) -> tuple[str, float]:
        """Return the voice ID valid for ``backend``, plus the configured speed.

        Voice IDs are not portable across backends, so the configured
        ``speech.voice_id`` is honored only when ``backend`` is the configured
        ``speech.tts_backend``. Otherwise the fallback backend's own default is
        used and the substitution is reported once per session.
        """
        want_backend, voice_id, speed = self.get_voice_preferences()
        active = getattr(backend, "backend_id", "") or ""
        if active == want_backend:
            return voice_id, speed

        from openjarvis.speech._tts_discovery import default_voice_for

        substitute = default_voice_for(active)
        if console is not None and active not in self._voice_warned:
            self._voice_warned.add(active)
            console.print(
                f"[dim yellow]Voice: {want_backend!r} unavailable — using "
                f"{active!r} with its default voice "
                f"({substitute or 'backend default'}).[/dim yellow]"
            )
        return substitute, speed

    def discard_tts_backend(self) -> None:
        """Forget a backend that failed synthesis and allow the next fallback."""
        self._tts_backend = None


def read_voice_input(console: Any, session: VoiceSession) -> Optional[str] | object:
    """Accept a typed command/message, or record after an empty submission."""
    try:
        typed = input("You> [type, or press Enter to speak] ")
    except (EOFError, KeyboardInterrupt):
        return VOICE_EXIT
    typed = typed.strip()
    return typed if typed else record_voice(console, session)


def record_voice(
    console: Any,
    session: VoiceSession | None = None,
) -> Optional[str] | object:
    """Record from mic, transcribe, and return text or a loop sentinel."""
    from openjarvis.speech.voice_io import record_until_silence

    active_session = session or VoiceSession()
    backend = active_session.get_stt_backend()
    if backend is None:
        console.print(
            "[yellow]Speech recognition is unavailable. "
            "You can keep typing. Run jarvis doctor to check the "
            "voice dependencies and local model cache.[/yellow]"
        )
        return None

    console.print("[dim cyan]Listening… (speak now, stops on silence)[/dim cyan]")
    try:
        audio_bytes = record_until_silence()
    except KeyboardInterrupt:
        return VOICE_EXIT
    except Exception as exc:
        # PortAudio failures are regular Exceptions (and can surface as
        # OSError), so report them without letting terminal control sequences
        # through. SystemExit deliberately continues to propagate, while
        # Ctrl-C maps to the chat loop's graceful-exit sentinel above.
        console.print(f"[red]Mic error: {_terminal_safe_text(exc)}[/red]")
        console.print("[dim]You can keep typing.[/dim]")
        return None

    console.print("[dim]Transcribing…[/dim]")
    try:
        result = backend.transcribe(
            audio_bytes, format="wav", language=active_session.get_stt_language()
        )
        text = result.text.strip()
        if text:
            console.print(f"[bold]You (voice):[/bold] {_terminal_safe_text(text)}")
            return text
        console.print("[dim]Nothing heard — try again.[/dim]")
        return None
    except Exception as exc:
        console.print(f"[red]Transcription error: {_terminal_safe_text(exc)}[/red]")
        return None


def _wake_word_command(text: str, wake_words: list[str]) -> str | None:
    """Return the text after a configured leading wake phrase, if present."""
    tokens = list(re.finditer(r"[\w]+", text, flags=re.UNICODE))
    if not tokens:
        return None
    normalized_words = [
        "".join(char for char in word.casefold() if char.isalnum())
        for word in wake_words
        if word.strip()
    ]
    normalized_words = [word for word in normalized_words if word]
    starts = [0]
    if tokens[0].group(0).casefold() in {"hey", "ok", "okay"}:
        starts.append(1)
    for start in starts:
        for wake_word in normalized_words:
            combined = ""
            for end in range(start, len(tokens)):
                combined += "".join(
                    char for char in tokens[end].group(0).casefold() if char.isalnum()
                )
                if combined == wake_word:
                    return text[tokens[end].end() :].lstrip(" \t,.;:!?—-–")
                if len(combined) >= len(wake_word):
                    break
    return None


def read_hands_free_input(
    console: Any,
    session: VoiceSession,
    wake_words: list[str] | None = None,
    *,
    source: str = "microphone",
) -> Optional[str] | object:
    """Listen for local speech and return only requests prefixed by a wake word.

    Speech that does not begin with a configured wake phrase is transcribed
    locally and discarded without being printed or added to chat history.
    """
    if source == "system":
        from openjarvis.speech.voice_io import record_loopback_speech_until_silence

        record_phrase = record_loopback_speech_until_silence
        source_label = "the default speaker output"
    elif source == "microphone":
        from openjarvis.speech.voice_io import record_speech_until_silence

        record_phrase = record_speech_until_silence
        source_label = "the microphone"
    else:
        raise ValueError(f"Unsupported hands-free audio source: {source}")

    backend = session.get_stt_backend()
    if backend is None or getattr(backend, "backend_id", "") != "faster-whisper":
        console.print(
            "[red]Hands-free mode needs the local faster-whisper backend. "
            "Install voice dependencies and set [speech].offline_only = true.[/red]"
        )
        return VOICE_EXIT

    if wake_words is None:
        config = session._config
        speech_config = getattr(config, "speech", None)
        wake_words = getattr(speech_config, "wake_words", ["jarvis"])
    if isinstance(wake_words, str):
        wake_words = [word.strip() for word in wake_words.split(",") if word.strip()]
    if not isinstance(wake_words, (list, tuple)):
        wake_words = ["jarvis"]
    wake_words = [str(word).strip() for word in wake_words if str(word).strip()]
    if not wake_words:
        console.print("[red]Configure at least one [speech].wake_words entry.[/red]")
        return VOICE_EXIT

    if not getattr(session, "_hands_free_banner_shown", False):
        safe_words = ", ".join(_terminal_safe_text(word) for word in wake_words)
        console.print(
            f"[dim cyan]Hands-free listening active on {source_label}. "
            f"Say {safe_words} "
            "followed by a request; press Ctrl+C to stop.[/dim cyan]"
        )
        session._hands_free_banner_shown = True

    try:
        audio_bytes = record_phrase()
    except KeyboardInterrupt:
        return VOICE_EXIT
    except Exception as exc:
        console.print(f"[red]Mic error: {_terminal_safe_text(exc)}[/red]")
        return VOICE_EXIT

    try:
        result = backend.transcribe(
            audio_bytes,
            format="wav",
            language=session.get_stt_language(),
        )
        text = result.text.strip()
    except Exception as exc:
        console.print(
            f"[dim yellow]Local transcription failed: "
            f"{_terminal_safe_text(exc)}[/dim yellow]"
        )
        return None

    if getattr(session, "_wake_pending", False):
        command = text
        session._wake_pending = False
    else:
        command = _wake_word_command(text, wake_words)
    if command is None:
        return None
    if not command:
        session._wake_pending = True
        console.print("[dim]Activation detected; say your request.[/dim]")
        return None

    console.print(f"[bold]You (voice):[/bold] {_terminal_safe_text(command)}")
    return command


def speak(text: str, console: Any, session: VoiceSession | None = None) -> None:
    """Synthesize and play text, reusing a healthy backend for the session."""
    from openjarvis.speech.voice_io import play_wav

    active_session = session or VoiceSession()

    # Resolved before the loop: a bad config value must surface as a config
    # error, not be swallowed by the per-backend fallback handler below.
    try:
        active_session.get_voice_preferences()
    except Exception as exc:
        console.print(f"[red]Invalid speech config: {_terminal_safe_text(exc)}[/red]")
        return

    while (backend := active_session.get_tts_backend()) is not None:
        try:
            voice_id, speed = active_session.voice_for_backend(backend, console)
            synth_kwargs: dict[str, Any] = {"output_format": "wav", "speed": speed}
            if voice_id:
                synth_kwargs["voice_id"] = voice_id
            result = backend.synthesize(text, **synth_kwargs)
            if result.audio:
                play_audio = getattr(type(backend), "play_audio", None)
                if callable(play_audio):
                    play_audio(backend, result.audio)
                else:
                    play_wav(result.audio, sample_rate=result.sample_rate)
            return
        except Exception as exc:
            console.print(
                f"[dim yellow]Voice backend "
                f"{getattr(backend, 'backend_id', '?')!r} failed: "
                f"{_terminal_safe_text(exc)}[/dim yellow]"
            )
            active_session.discard_tts_backend()

    console.print(
        "[dim yellow]No TTS backend available — install Kokoro or check for an "
        "installed Windows speech voice with `jarvis doctor`.[/dim yellow]"
    )


__all__ = [
    "VOICE_EXIT",
    "VoiceSession",
    "read_hands_free_input",
    "read_voice_input",
    "record_voice",
    "speak",
]
