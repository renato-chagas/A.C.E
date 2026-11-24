"""Tests for read-only local system diagnostics."""

from __future__ import annotations

from openjarvis.tools import system_health


def test_health_report_contains_local_resources_without_reading_files(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        system_health,
        "get_config_dir",
        lambda: tmp_path / "missing-config-dir",
    )
    monkeypatch.setattr(
        system_health,
        "_memory_bytes",
        lambda: (8 * 1024**3, 3 * 1024**3),
    )
    monkeypatch.setattr(
        system_health.shutil,
        "which",
        lambda name: "C:/local/ollama.exe" if name == "ollama" else None,
    )

    result = system_health.SystemHealthTool().execute()

    assert result.success
    assert result.metadata["disk_path"] == str(tmp_path)
    assert result.metadata["memory_total_bytes"] == 8 * 1024**3
    assert result.metadata["local_runtimes"]["ollama"]
    assert "Operating system:" in result.content
    assert "Memory available:" in result.content


def test_health_report_includes_graphics_adapter_names(monkeypatch):
    monkeypatch.setattr(system_health, "_graphics_adapters", lambda: ["AMD Radeon"])

    result = system_health.SystemHealthTool().execute()

    assert result.metadata["graphics_adapters"] == ["AMD Radeon"]
    assert "Graphics adapters" in result.content
