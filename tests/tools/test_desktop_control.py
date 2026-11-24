"""Safety tests for confirmation-gated desktop controls."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from openjarvis.core.types import ToolCall
from openjarvis.tools._stubs import ToolExecutor
from openjarvis.tools.desktop_control import (
    DesktopClickTool,
    DesktopPressTool,
    DesktopTypeTool,
    _check_coordinates,
)


def test_desktop_tools_require_capability_and_confirmation() -> None:
    for tool in (DesktopClickTool(), DesktopTypeTool(), DesktopPressTool()):
        assert tool.spec.required_capabilities == ["desktop:control"]
        assert tool.spec.requires_confirmation is True


def test_desktop_press_rejects_clipboard_shortcuts() -> None:
    tool = DesktopPressTool()
    with patch("openjarvis.tools.desktop_control._press_keys") as press:
        for key in ("ctrl+c", "ctrl+v"):
            result = tool.execute(key=key)
            assert result.success is False
            assert "allowlist" in result.content
        press.assert_not_called()


def test_desktop_type_confirmation_shows_text_and_denial_blocks_typing() -> None:
    prompts: list[str] = []
    tool = DesktopTypeTool()
    executor = ToolExecutor(
        [tool],
        interactive=True,
        confirm_callback=lambda prompt: prompts.append(prompt) or False,
    )
    call = ToolCall(
        id="desktop-type",
        name="desktop_type",
        arguments=json.dumps({"text": "Reviewed message"}),
    )

    with patch("openjarvis.tools.desktop_control._type_text") as type_text:
        result = executor.execute(call)

    assert result.success is False
    assert "Reviewed message" in prompts[0]
    type_text.assert_not_called()


def test_desktop_click_waits_for_approval_before_native_input() -> None:
    prompts: list[str] = []
    user32 = MagicMock()
    user32.SetCursorPos.return_value = True
    api = (user32, MagicMock(), MagicMock(), MagicMock(), MagicMock())
    executor = ToolExecutor(
        [DesktopClickTool()],
        interactive=True,
        confirm_callback=lambda prompt: prompts.append(prompt) or False,
    )
    call = ToolCall(
        id="desktop-click",
        name="desktop_click",
        arguments=json.dumps({"x": 12, "y": 34, "button": "left"}),
    )

    with (
        patch(
            "openjarvis.tools.desktop_control._check_coordinates",
            return_value=(12, 34),
        ),
        patch("openjarvis.tools.desktop_control._windows_api", return_value=api),
        patch("openjarvis.tools.desktop_control._send_inputs") as send_inputs,
    ):
        result = executor.execute(call)

    assert result.success is False
    assert "'x': 12" in prompts[0]
    assert "'y': 34" in prompts[0]
    user32.SetCursorPos.assert_not_called()
    send_inputs.assert_not_called()


def test_click_coordinates_must_be_inside_primary_screen() -> None:
    user32 = MagicMock()
    user32.GetSystemMetrics.side_effect = lambda index: 1920 if index == 0 else 1080
    api = (user32, MagicMock(), MagicMock(), MagicMock(), MagicMock())

    with patch("openjarvis.tools.desktop_control._windows_api", return_value=api):
        assert _check_coordinates(1919, 1079) == (1919, 1079)
        with pytest.raises(ValueError, match="inside the 1920x1080 screen"):
            _check_coordinates(1920, 1079)

    with patch("openjarvis.tools.desktop_control._windows_api") as windows_api:
        with pytest.raises(ValueError, match="x must be an integer"):
            _check_coordinates(True, 10)
        windows_api.assert_not_called()


def test_desktop_type_normalizes_newlines_without_clipboard(monkeypatch) -> None:
    tool = DesktopTypeTool()
    with patch("openjarvis.tools.desktop_control._type_text") as type_text:
        result = tool.execute(text="linha 1\r\nlinha 2\ttab", interval=0.02)

    assert result.success is True
    type_text.assert_called_once_with("linha 1\nlinha 2\ttab", 0.02)


def test_desktop_type_rejects_unsupported_control_chars() -> None:
    with patch("openjarvis.tools.desktop_control._type_text") as type_text:
        result = DesktopTypeTool().execute(text="hello\x01world")

    assert result.success is False
    assert "control characters" in result.content
    type_text.assert_not_called()
