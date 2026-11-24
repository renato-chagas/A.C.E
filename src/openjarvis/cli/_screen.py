"""Screen capture for vision input (``jarvis ask --screen``).

Captures the primary monitor as PNG bytes or to a temporary file. On Windows
the in-memory path uses the built-in .NET ``System.Drawing`` stack (no
third-party dependency). Other platforms use ``mss`` or ``Pillow`` if installed.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
from binascii import Error as Base64Error
from io import BytesIO
from pathlib import Path

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_PNG_BYTES = 20 * 1024 * 1024
_MAX_CAPTURE_PIXELS = 7680 * 4320

_PS_CAPTURE_BYTES = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$screen = [System.Windows.Forms.Screen]::PrimaryScreen
if ($null -eq $screen) { throw 'No interactive primary screen is accessible.' }
$bounds = $screen.Bounds
if (([long]$bounds.Width * $bounds.Height) -gt MAX_CAPTURE_PIXELS) {
    throw 'The primary screen exceeds the capture pixel limit.'
}
$bitmap = $null
$graphics = $null
$stream = $null
try {
    $bitmap = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.CopyFromScreen($bounds.X, $bounds.Y, 0, 0, $bitmap.Size)
    $stream = New-Object System.IO.MemoryStream
    $bitmap.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
    if ($stream.Length -gt MAX_PNG_BYTES) { throw 'The screen PNG exceeds 20 MiB.' }
    [Console]::Out.WriteLine([Convert]::ToBase64String($stream.ToArray()))
} finally {
    if ($null -ne $stream) { $stream.Dispose() }
    if ($null -ne $graphics) { $graphics.Dispose() }
    if ($null -ne $bitmap) { $bitmap.Dispose() }
}
""".replace("MAX_CAPTURE_PIXELS", str(_MAX_CAPTURE_PIXELS)).replace(
    "MAX_PNG_BYTES", str(_MAX_PNG_BYTES)
)


def _validate_screen_png(image: bytes) -> bytes:
    if not image.startswith(_PNG_SIGNATURE):
        raise RuntimeError("screen capture did not return a PNG image")
    if len(image) > _MAX_PNG_BYTES:
        raise RuntimeError("screen capture exceeds the 20 MiB image limit")
    return image


def _capture_windows_screen() -> bytes:
    executable = (
        Path(os.environ.get("SystemRoot", "C:/Windows"))
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )
    encoded = base64.b64encode(_PS_CAPTURE_BYTES.encode("utf-16-le")).decode("ascii")
    try:
        proc = subprocess.run(
            [
                str(executable),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded,
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("screen capture exceeded 30 seconds") from exc
    except OSError as exc:
        raise RuntimeError("Windows PowerShell could not start screen capture") from exc
    output = proc.stdout.strip()
    if proc.returncode != 0 or not output:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"screen capture failed: {detail or 'empty image returned'}")
    if len(output) > 4 * ((_MAX_PNG_BYTES + 2) // 3):
        raise RuntimeError("screen capture exceeds the 20 MiB image limit")
    try:
        image = base64.b64decode(output, validate=True)
    except Base64Error as exc:
        raise RuntimeError("screen capture returned invalid image bytes") from exc
    return _validate_screen_png(image)


def capture_screen_bytes() -> bytes:
    """Capture the primary screen as PNG bytes without saving it to disk."""
    if sys.platform.startswith("win"):
        return _capture_windows_screen()

    try:
        import mss
        from mss.tools import to_png
    except ImportError:
        mss = None

    if mss is not None:
        try:
            with mss.mss() as capture:
                if len(capture.monitors) < 2:
                    raise RuntimeError("no primary screen is available")
                screenshot = capture.grab(capture.monitors[1])
                image = to_png(screenshot.rgb, screenshot.size)
        except Exception as exc:
            raise RuntimeError("screen capture failed") from exc
        return _validate_screen_png(image)

    try:
        from PIL import ImageGrab
    except ImportError as exc:
        raise RuntimeError(
            "screen capture on this platform needs 'mss' or 'Pillow' "
            "(try: pip install mss)"
        ) from exc

    try:
        buffer = BytesIO()
        ImageGrab.grab().save(buffer, format="PNG")
        image = buffer.getvalue()
    except Exception as exc:
        raise RuntimeError("screen capture failed") from exc
    return _validate_screen_png(image)


def capture_screen_to_temp() -> str:
    """Capture the primary screen and write it to a caller-owned temporary PNG."""
    image = capture_screen_bytes()
    fd, path = tempfile.mkstemp(prefix="jarvis_screen_", suffix=".png")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(image)
    except OSError:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


__all__ = ["capture_screen_bytes", "capture_screen_to_temp"]
