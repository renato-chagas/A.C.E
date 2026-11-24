"""Tests for local persona files seeded by the ACE preset."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from openjarvis.cli import cli
from openjarvis.cli.init_cmd import _ACE_SOUL, _DEFAULT_SOUL
from openjarvis.core.config import load_config


class TestInitAcePersona:
    def test_ace_preset_seeds_persona_and_continuity_files(
        self, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / ".openjarvis"
        config_path = config_dir / "config.toml"
        with (
            mock.patch("openjarvis.cli.init_cmd.DEFAULT_CONFIG_DIR", config_dir),
            mock.patch("openjarvis.cli.init_cmd.DEFAULT_CONFIG_PATH", config_path),
            mock.patch("openjarvis.cli.init_cmd.detect_cloud_keys", return_value=None),
        ):
            result = CliRunner().invoke(
                cli, ["init", "--preset", "ace", "--no-analytics"]
            )

        assert result.exception is None, result.output
        assert result.exit_code == 0
        assert config_path.is_file()
        assert "facts_semantic_enabled = true" in config_path.read_text(
            encoding="utf-8"
        )
        assert "auto_start_local = true" in config_path.read_text(encoding="utf-8")
        ace_config = load_config(config_path)
        assert ace_config.engine.auto_start_local is True
        assert ace_config.intelligence.provider == "local"
        assert ace_config.intelligence.default_model == ""
        assert ace_config.intelligence.fallback_model == ""
        assert "jarvis model setup" in result.output
        assert (config_dir / "SOUL.md").read_text(encoding="utf-8") == _ACE_SOUL
        assert (config_dir / "MEMORY.md").read_text(
            encoding="utf-8"
        ) == "# Agent Memory\n\n"
        assert (config_dir / "USER.md").read_text(
            encoding="utf-8"
        ) == "# User Profile\n\n"
        assert (config_dir / "skills").is_dir()

    def test_ace_preset_replaces_stock_soul_but_preserves_user_files(
        self, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / ".openjarvis"
        config_dir.mkdir()
        (config_dir / "SOUL.md").write_text(_DEFAULT_SOUL, encoding="utf-8")
        (config_dir / "MEMORY.md").write_text("User memory\n", encoding="utf-8")
        (config_dir / "USER.md").write_text("User profile\n", encoding="utf-8")
        config_path = config_dir / "config.toml"

        with (
            mock.patch("openjarvis.cli.init_cmd.DEFAULT_CONFIG_DIR", config_dir),
            mock.patch("openjarvis.cli.init_cmd.DEFAULT_CONFIG_PATH", config_path),
            mock.patch("openjarvis.cli.init_cmd.detect_cloud_keys", return_value=None),
        ):
            result = CliRunner().invoke(
                cli, ["init", "--preset", "ace", "--no-analytics"]
            )

        assert result.exception is None, result.output
        assert result.exit_code == 0
        assert (config_dir / "SOUL.md").read_text(encoding="utf-8") == _ACE_SOUL
        assert (config_dir / "MEMORY.md").read_text(encoding="utf-8") == "User memory\n"
        assert (config_dir / "USER.md").read_text(encoding="utf-8") == "User profile\n"

    def test_ace_preset_preserves_custom_soul(self, tmp_path: Path) -> None:
        config_dir = tmp_path / ".openjarvis"
        config_dir.mkdir()
        custom_soul = "# My ACE\nKeep this custom persona.\n"
        (config_dir / "SOUL.md").write_text(custom_soul, encoding="utf-8")
        config_path = config_dir / "config.toml"

        with (
            mock.patch("openjarvis.cli.init_cmd.DEFAULT_CONFIG_DIR", config_dir),
            mock.patch("openjarvis.cli.init_cmd.DEFAULT_CONFIG_PATH", config_path),
            mock.patch("openjarvis.cli.init_cmd.detect_cloud_keys", return_value=None),
        ):
            result = CliRunner().invoke(
                cli, ["init", "--preset", "ace", "--no-analytics"]
            )

        assert result.exception is None, result.output
        assert result.exit_code == 0
        assert (config_dir / "SOUL.md").read_text(encoding="utf-8") == custom_soul
