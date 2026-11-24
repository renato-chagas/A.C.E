"""Use the installed Windows OCR language packs with images passed over stdin."""

from __future__ import annotations

import base64
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from openjarvis.tools.storage.ocr import (
    _MAX_TEXT_CHARS,
    OcrProcessingError,
    OcrUnavailableError,
)

_OCR_ENGINE = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
function Get-LocalRuntimeType($name) {
    return [type]("$name, Windows.Foundation, ContentType=WindowsRuntime")
}
$ocrEngine = Get-LocalRuntimeType 'Windows.Media.Ocr.OcrEngine'
$engine = $ocrEngine::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) { exit 12 }
"""
_OCR_IMAGE = (
    _OCR_ENGINE
    + r"""
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$bitmapDecoder = Get-LocalRuntimeType 'Windows.Graphics.Imaging.BitmapDecoder'
$softwareBitmap = Get-LocalRuntimeType 'Windows.Graphics.Imaging.SoftwareBitmap'
$ocrResult = Get-LocalRuntimeType 'Windows.Media.Ocr.OcrResult'
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $parameters = $_.GetParameters()
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $parameters.Count -eq 1 -and
    $parameters[0].ParameterType.Name -eq ('IAsyncOperation' + [char]96 + '1')
} | Select-Object -First 1
function Wait-LocalOperation($operation, [Type]$type) {
    $task = $asTask.MakeGenericMethod($type).Invoke($null, @($operation))
    return $task.GetAwaiter().GetResult()
}
$memory = [IO.MemoryStream]::new()
[Console]::OpenStandardInput().CopyTo($memory)
$memory.Position = 0
$stream = [System.IO.WindowsRuntimeStreamExtensions]::AsRandomAccessStream($memory)
$decoder = Wait-LocalOperation ($bitmapDecoder::CreateAsync($stream)) $bitmapDecoder
$max = [double]$ocrEngine::MaxImageDimension
$edge = [Math]::Max([double]$decoder.PixelWidth, [double]$decoder.PixelHeight)
if ($edge -gt $max) {
    $transformType = Get-LocalRuntimeType 'Windows.Graphics.Imaging.BitmapTransform'
    $formatType = Get-LocalRuntimeType 'Windows.Graphics.Imaging.BitmapPixelFormat'
    $alphaType = Get-LocalRuntimeType 'Windows.Graphics.Imaging.BitmapAlphaMode'
    $exifType = Get-LocalRuntimeType 'Windows.Graphics.Imaging.ExifOrientationMode'
    $colorType = Get-LocalRuntimeType 'Windows.Graphics.Imaging.ColorManagementMode'
    $transform = $transformType::new()
    $ratio = $max / $edge
    $width = [Math]::Floor($decoder.PixelWidth * $ratio)
    $height = [Math]::Floor($decoder.PixelHeight * $ratio)
    $transform.ScaledWidth = [uint32][Math]::Max(1, $width)
    $transform.ScaledHeight = [uint32][Math]::Max(1, $height)
    $decode = $decoder.GetSoftwareBitmapAsync(
        $formatType::Bgra8, $alphaType::Ignore, $transform,
        $exifType::IgnoreExifOrientation, $colorType::DoNotColorManage
    )
} else {
    $decode = $decoder.GetSoftwareBitmapAsync()
}
$bitmap = Wait-LocalOperation $decode $softwareBitmap
$result = Wait-LocalOperation ($engine.RecognizeAsync($bitmap)) $ocrResult
$lines = $result.Lines | ForEach-Object { $_.Text }
[Console]::Write($lines -join [Environment]::NewLine)
"""
)


def _run_windows_ocr(
    executable: Path, script: str, image: bytes, *, timeout: float
) -> subprocess.CompletedProcess[bytes]:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        [
            str(executable),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encoded,
        ],
        input=image,
        capture_output=True,
        timeout=timeout,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def windows_ocr_reader() -> Callable[[bytes], str]:
    """Check the installed Windows OCR engine, then prepare an in-memory reader."""
    executable = (
        Path(os.environ.get("SystemRoot", "C:/Windows"))
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )
    if not executable.is_file():
        raise OcrUnavailableError("Windows OCR requires Windows PowerShell.")
    unavailable = "Windows OCR has no installed language available for this user."
    try:
        ready = _run_windows_ocr(executable, _OCR_ENGINE, b"", timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OcrUnavailableError("Windows OCR could not be checked locally.") from exc
    if ready.returncode != 0:
        raise OcrUnavailableError(unavailable)

    def read(image: bytes) -> str:
        try:
            result = _run_windows_ocr(executable, _OCR_IMAGE, image, timeout=30)
        except subprocess.TimeoutExpired as exc:
            raise OcrProcessingError(
                "Windows OCR exceeded its 30-second limit."
            ) from exc
        except OSError as exc:
            raise OcrProcessingError("Windows OCR could not be started.") from exc
        if result.returncode == 12:
            raise OcrUnavailableError(unavailable)
        if result.returncode != 0:
            raise OcrProcessingError("Windows OCR could not read the screenshot.")
        return result.stdout.decode("utf-8", errors="replace").strip()[:_MAX_TEXT_CHARS]

    return read
