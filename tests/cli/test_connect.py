"""Tests for ``jarvis connect`` CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from openjarvis.cli import cli


def test_connect_list_no_connectors() -> None:
    """--list with an empty registry shows a 'no connectors' message."""
    runner = CliRunner()
    with mock.patch(
        "openjarvis.cli.connect_cmd.connect.__wrapped__"
        if hasattr(cli, "__wrapped__")
        else "openjarvis.core.registry.ConnectorRegistry.items",
        return_value=(),
    ):
        with mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.items",
            return_value=(),
        ):
            result = runner.invoke(cli, ["connect", "--list"])

    assert result.exit_code == 0
    assert "No connectors registered" in result.output


def test_connect_list_with_connector(tmp_path: object) -> None:
    """--list with a connector registered shows it in the table."""
    runner = CliRunner()

    # Build a minimal mock connector class
    mock_cls = mock.MagicMock()
    mock_cls.auth_type = "filesystem"
    mock_instance = mock.MagicMock()
    mock_instance.is_connected.return_value = True
    mock_cls.return_value = mock_instance

    with mock.patch(
        "openjarvis.core.registry.ConnectorRegistry.items",
        return_value=(("obsidian", mock_cls),),
    ):
        result = runner.invoke(cli, ["connect", "--list"])

    assert result.exit_code == 0
    assert "obsidian" in result.output


def test_connect_help() -> None:
    """--help exits 0 and mentions the word 'connect'."""
    runner = CliRunner()
    result = runner.invoke(cli, ["connect", "--help"])
    assert result.exit_code == 0
    assert "connect" in result.output.lower()


def test_connect_specific_source(tmp_path: object) -> None:
    """connect --path /nonexistent obsidian shows an error gracefully."""
    runner = CliRunner()

    mock_cls = mock.MagicMock()
    mock_cls.auth_type = "filesystem"
    mock_instance = mock.MagicMock()
    # Path does not exist -> is_connected returns False
    mock_instance.is_connected.return_value = False
    mock_cls.return_value = mock_instance

    with (
        mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.contains",
            return_value=True,
        ),
        mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.get",
            return_value=mock_cls,
        ),
    ):
        # --path before the positional source arg (standard Click group behaviour)
        result = runner.invoke(cli, ["connect", "--path", "/nonexistent", "obsidian"])

    assert result.exit_code == 0
    # Should mention the source and give an indication something went wrong
    assert "obsidian" in result.output or "nonexistent" in result.output


def test_connect_disconnect() -> None:
    """--disconnect gmail exits 0."""
    runner = CliRunner()

    mock_cls = mock.MagicMock()
    mock_instance = mock.MagicMock()
    mock_cls.return_value = mock_instance

    with (
        mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.contains",
            return_value=True,
        ),
        mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.get",
            return_value=mock_cls,
        ),
    ):
        result = runner.invoke(cli, ["connect", "--disconnect", "gmail"])

    assert result.exit_code == 0
    mock_instance.disconnect.assert_called_once()


def test_connect_token_uses_connector_setter_without_duplicate_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Token setup delegates validation and secure persistence to the connector."""
    runner = CliRunner()
    mock_cls = mock.MagicMock()
    mock_cls.auth_type = "token"
    mock_instance = mock.MagicMock()
    mock_instance.is_connected.return_value = False
    mock_cls.return_value = mock_instance
    monkeypatch.chdir(tmp_path)

    with (
        mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.contains",
            return_value=True,
        ),
        mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.get",
            return_value=mock_cls,
        ),
        mock.patch(
            "openjarvis.cli.connect_cmd.click.prompt", return_value="local-test-token"
        ) as prompt,
    ):
        result = runner.invoke(cli, ["connect", "github_notifications"])

    assert result.exit_code == 0
    assert "local-test-token" not in result.output
    mock_instance.set_token.assert_called_once_with("local-test-token")
    prompt.assert_called_once_with(
        "Enter your github_notifications personal access token", hide_input=True
    )
    assert not (tmp_path / "connectors").exists()
    assert not (tmp_path / "github_notifications").exists()


def test_connect_weather_collects_location_and_uses_secure_configure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weather uses its API-key/location validator instead of a token file."""
    runner = CliRunner()
    mock_cls = mock.MagicMock()
    mock_cls.auth_type = "token"
    mock_instance = mock.MagicMock()
    mock_instance.is_connected.return_value = False
    mock_cls.return_value = mock_instance
    monkeypatch.chdir(tmp_path)

    with (
        mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.contains",
            return_value=True,
        ),
        mock.patch(
            "openjarvis.core.registry.ConnectorRegistry.get",
            return_value=mock_cls,
        ),
        mock.patch(
            "openjarvis.cli.connect_cmd.click.prompt",
            side_effect=["weather-test-key", "Porto Alegre"],
        ) as prompt,
    ):
        result = runner.invoke(cli, ["connect", "weather"])

    assert result.exit_code == 0
    assert "weather-test-key" not in result.output
    mock_instance.configure.assert_called_once_with(
        api_key="weather-test-key", location="Porto Alegre"
    )
    assert prompt.call_args_list[0].kwargs == {"hide_input": True}
    assert prompt.call_args_list[1].args == ("Enter your location",)
    assert not (tmp_path / "connectors").exists()
