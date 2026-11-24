# A.C.E. Local Speech on Windows

## Offline SAPI playback

The `windows_sapi` text-to-speech backend uses the installed Windows
`System.Speech` voice to synthesize a WAV in memory. Interactive CLI chat now
passes that WAV to Windows `System.Media.SoundPlayer`, so this fallback does
not require Python audio playback packages. Other TTS backends keep using the
existing `sounddevice` playback path. The server TTS endpoint continues to
return synthesized audio to its client.

The PowerShell program is static and sent using `-EncodedCommand`; request data
travels through standard input as base64-encoded JSON. Text and audio are never
interpolated into shell source or written to a temporary file. Playback accepts
only valid, uncompressed WAV data up to 64 MiB and 30 minutes. The player runs
synchronously, and Ctrl+C terminates its PowerShell child process so speech
does not continue after the chat command is interrupted.

The backend prefers an installed Portuguese (Brazil) voice. Its voice listing,
WAV synthesis, and playback all use local Windows components. Microphone input
still requires a local speech-recognition backend and the optional audio input
dependencies; SAPI output does not provide speech recognition.

## Offline speech recognition

For `[speech].offline_only = true`, speech discovery checks Faster-Whisper
only. Model lookup uses the existing local cache or a configured model
directory. It requires `config.json`, `model.bin`, and `tokenizer.json`
before loading. Faster-Whisper can fetch a tokenizer separately if the bundle
lacks one. `jarvis doctor` checks the same files as the runtime.

Microphone and loopback audio reach the decoder through an in-memory stream,
which stays open until transcription finishes. `jarvis chat --voice` keeps
typed input available if recognition or microphone capture fails. On a CPU
without CUDA, automatic device selection checks the CPU's supported compute
types and can substitute `int8` for an unsupported `float16` request.

## Validation and limits

On 2026-11-18, the focused Windows SAPI and interactive chat suites passed 41
tests. They cover voice selection, WAV validation, no-shell process invocation,
interrupt cleanup, and CLI routing to the native playback method. A Windows
smoke check played a 10 ms silent WAV through the real `System.Media` path.
Ruff and `git diff --check` passed. The smoke check verifies the playback path
and device availability, not voice quality or microphone input.

In the current ACE environment, no Faster-Whisper, Kokoro, `sounddevice`, or
PyAV package is installed, and no SAPI speech recognizer is available. Local
SAPI output can work independently of those Python audio packages; voice input
still cannot until a recognizer and microphone capture dependency are present.
