"""Explicit, confirmation-gated controls for the current desktop session."""

from __future__ import annotations

import ctypes
import os
import re
import time
from ctypes import wintypes
from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

_MAX_TEXT_CHARS = 2000
_ALLOWED_KEYS = {
    "enter",
    "tab",
    "esc",
    "backspace",
    "delete",
    "space",
    "up",
    "down",
    "left",
    "right",
    "home",
    "end",
    "pageup",
    "pagedown",
    "ctrl+a",
    "ctrl+z",
    "ctrl+y",
    "ctrl+s",
    "ctrl+f",
    "ctrl+l",
    "alt+left",
    "alt+right",
}
_KEY_ALIASES = {
    "escape": "esc",
    "return": "enter",
    "pgup": "pageup",
    "pgdn": "pagedown",
}
_VIRTUAL_KEYS = {
    "enter": 0x0D,
    "tab": 0x09,
    "esc": 0x1B,
    "backspace": 0x08,
    "delete": 0x2E,
    "space": 0x20,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    "ctrl": 0x11,
    "alt": 0x12,
    "a": 0x41,
    "f": 0x46,
    "l": 0x4C,
    "s": 0x53,
    "x": 0x58,
    "y": 0x59,
    "z": 0x5A,
}
_MOUSE_FLAGS = {
    "left": (0x0002, 0x0004),
    "right": (0x0008, 0x0010),
    "middle": (0x0020, 0x0040),
}


def _windows_api():
    if os.name != "nt":
        raise RuntimeError("Native desktop control is currently supported on Windows.")

    class _MouseInput(ctypes.Structure):
        _fields_ = [
            ("x", wintypes.LONG),
            ("y", wintypes.LONG),
            ("mouse_data", wintypes.DWORD),
            ("flags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("extra_info", ctypes.c_size_t),
        ]

    class _KeyboardInput(ctypes.Structure):
        _fields_ = [
            ("virtual_key", wintypes.WORD),
            ("scan_code", wintypes.WORD),
            ("flags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("extra_info", ctypes.c_size_t),
        ]

    class _InputUnion(ctypes.Union):
        _fields_ = [("mouse", _MouseInput), ("keyboard", _KeyboardInput)]

    class _Input(ctypes.Structure):
        _anonymous_ = ("payload",)
        _fields_ = [("input_type", wintypes.DWORD), ("payload", _InputUnion)]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SetCursorPos.argtypes = [wintypes.INT, wintypes.INT]
    user32.SetCursorPos.restype = wintypes.BOOL
    user32.GetSystemMetrics.argtypes = [wintypes.INT]
    user32.GetSystemMetrics.restype = wintypes.INT
    return user32, _Input, _InputUnion, _MouseInput, _KeyboardInput


def _send_inputs(*events: Any) -> None:
    if not events:
        return
    user32, *_ = _windows_api()
    input_type = type(events[0])
    user32.SendInput.argtypes = [
        wintypes.UINT,
        ctypes.POINTER(input_type),
        ctypes.c_int,
    ]
    user32.SendInput.restype = wintypes.UINT
    inputs = (input_type * len(events))(*events)
    sent = user32.SendInput(len(inputs), inputs, ctypes.sizeof(input_type))
    if sent != len(inputs):
        raise OSError(ctypes.get_last_error(), "Windows could not complete the input.")


def _check_coordinates(x: Any, y: Any) -> tuple[int, int]:
    if isinstance(x, bool) or not isinstance(x, int):
        raise ValueError("x must be an integer screen coordinate.")
    if isinstance(y, bool) or not isinstance(y, int):
        raise ValueError("y must be an integer screen coordinate.")
    user32, *_ = _windows_api()
    width = user32.GetSystemMetrics(0)
    height = user32.GetSystemMetrics(1)
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError(f"Coordinates must be inside the {width}x{height} screen.")
    return x, y


def _send_windows_unicode(text: str) -> None:
    """Type UTF-16 text through SendInput without reading/changing clipboard."""

    _, input_type, input_union, _, keyboard_type = _windows_api()
    key_up = 0x0002
    unicode_key = 0x0004
    events: list[Any] = []
    encoded = text.encode("utf-16-le", errors="strict")
    for index in range(0, len(encoded), 2):
        code_unit = int.from_bytes(encoded[index : index + 2], "little")
        events.append(
            input_type(
                input_type=1,
                payload=input_union(
                    keyboard=keyboard_type(0, code_unit, unicode_key, 0, 0)
                ),
            )
        )
        events.append(
            input_type(
                input_type=1,
                payload=input_union(
                    keyboard=keyboard_type(0, code_unit, unicode_key | key_up, 0, 0)
                ),
            )
        )
    if events:
        _send_inputs(*events)


def _type_text(text: str, interval: float) -> None:
    segments = re.split(r"([\r\n\t])", text)
    for segment in segments:
        if segment in ("\r", "\n", "\t"):
            key = "enter" if segment in ("\r", "\n") else "tab"
            _press_keys((key,))
        elif segment:
            if interval:
                for char in segment:
                    _send_windows_unicode(char)
                    time.sleep(interval)
            else:
                _send_windows_unicode(segment)


def _press_keys(keys: tuple[str, ...]) -> None:
    _, input_type, input_union, _, keyboard_type = _windows_api()
    key_up = 0x0002
    events = []
    for key in keys:
        events.append(
            input_type(
                input_type=1,
                payload=input_union(
                    keyboard=keyboard_type(_VIRTUAL_KEYS[key], 0, 0, 0, 0)
                ),
            )
        )
    for key in reversed(keys):
        events.append(
            input_type(
                input_type=1,
                payload=input_union(
                    keyboard=keyboard_type(_VIRTUAL_KEYS[key], 0, key_up, 0, 0)
                ),
            )
        )
    _send_inputs(*events)


@ToolRegistry.register("desktop_click")
class DesktopClickTool(BaseTool):
    """Click a coordinate on the user's current primary desktop."""

    tool_id = "desktop_click"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="desktop_click",
            description=(
                "Click one coordinate on the current primary desktop. Use only "
                "when the user explicitly requested a desktop action; obtain "
                "confirmation for this exact click before execution."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "Horizontal pixel coordinate.",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Vertical pixel coordinate.",
                    },
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button; defaults to left.",
                    },
                    "clicks": {
                        "type": "integer",
                        "enum": [1, 2],
                        "description": "One click, or a double-click.",
                    },
                },
                "required": ["x", "y"],
            },
            category="desktop",
            required_capabilities=["desktop:control"],
            requires_confirmation=True,
            timeout_seconds=15.0,
        )

    def execute(self, **params: Any) -> ToolResult:
        try:
            x, y = _check_coordinates(params.get("x"), params.get("y"))
            button = str(params.get("button", "left")).lower()
            clicks = params.get("clicks", 1)
            if button not in ("left", "right", "middle"):
                raise ValueError("button must be left, right, or middle.")
            if (
                isinstance(clicks, bool)
                or not isinstance(clicks, int)
                or clicks not in (1, 2)
            ):
                raise ValueError("clicks must be 1 or 2.")
            user32, input_type, input_union, mouse_type, _keyboard_type = _windows_api()
            if not user32.SetCursorPos(x, y):
                raise OSError(
                    ctypes.get_last_error(), "Could not position the pointer."
                )
            down_flag, up_flag = _MOUSE_FLAGS[button]
            for click_index in range(clicks):
                _send_inputs(
                    input_type(
                        input_type=0,
                        payload=input_union(mouse=mouse_type(0, 0, 0, down_flag, 0, 0)),
                    ),
                    input_type(
                        input_type=0,
                        payload=input_union(mouse=mouse_type(0, 0, 0, up_flag, 0, 0)),
                    ),
                )
                if clicks == 2 and click_index == 0:
                    time.sleep(0.1)
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Clicked at ({x}, {y}) with {button} button.",
                success=True,
                metadata={"x": x, "y": y, "button": button, "clicks": clicks},
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Desktop click failed: {exc}",
                success=False,
            )


@ToolRegistry.register("desktop_type")
class DesktopTypeTool(BaseTool):
    """Type explicitly supplied text into the focused desktop application."""

    tool_id = "desktop_type"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="desktop_type",
            description=(
                "Type exact text into the currently focused desktop application. "
                "Use only when the user explicitly requested this action. The "
                "terminal shows the exact text before asking for confirmation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Exact text to type; maximum 2000 characters.",
                    },
                    "interval": {
                        "type": "number",
                        "description": "Delay between characters (0-0.1 seconds).",
                    },
                },
                "required": ["text"],
            },
            category="desktop",
            required_capabilities=["desktop:control"],
            requires_confirmation=True,
            timeout_seconds=30.0,
            sensitive_parameters=["text"],
        )

    def execute(self, **params: Any) -> ToolResult:
        text = params.get("text", "")
        interval = params.get("interval", 0.0)
        try:
            if not isinstance(text, str) or not text:
                raise ValueError("text must be a non-empty string.")
            if len(text) > _MAX_TEXT_CHARS:
                raise ValueError(f"text cannot exceed {_MAX_TEXT_CHARS} characters.")
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            if any(ord(char) < 32 and char not in "\r\n\t" for char in text):
                raise ValueError("text contains unsupported control characters.")
            if isinstance(interval, bool) or not isinstance(interval, (int, float)):
                raise ValueError("interval must be a number from 0 to 0.1 seconds.")
            interval = float(interval)
            if not 0 <= interval <= 0.1:
                raise ValueError("interval must be between 0 and 0.1 seconds.")
            _type_text(text, interval)
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Typed {len(text)} characters into the focused application.",
                success=True,
                metadata={"characters": len(text)},
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Desktop typing failed: {exc}",
                success=False,
            )


@ToolRegistry.register("desktop_press")
class DesktopPressTool(BaseTool):
    """Press one narrowly allowlisted key or shortcut."""

    tool_id = "desktop_press"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="desktop_press",
            description=(
                "Press one allowlisted key or shortcut on the desktop. Each "
                "press requires confirmation; do not submit forms, send messages, "
                "or confirm purchases unless the user requested that exact action."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "enum": sorted(_ALLOWED_KEYS),
                        "description": (
                            "One allowed key or shortcut, such as tab or ctrl+f."
                        ),
                    }
                },
                "required": ["key"],
            },
            category="desktop",
            required_capabilities=["desktop:control"],
            requires_confirmation=True,
            timeout_seconds=15.0,
        )

    def execute(self, **params: Any) -> ToolResult:
        key = str(params.get("key", "")).strip().lower()
        key = _KEY_ALIASES.get(key, key)
        try:
            if key not in _ALLOWED_KEYS:
                raise ValueError("key is not in the desktop shortcut allowlist.")
            parts = key.split("+")
            _press_keys(tuple(parts))
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Pressed {key}.",
                success=True,
                metadata={"key": key},
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Desktop key action failed: {exc}",
                success=False,
            )


__all__ = ["DesktopClickTool", "DesktopPressTool", "DesktopTypeTool"]
