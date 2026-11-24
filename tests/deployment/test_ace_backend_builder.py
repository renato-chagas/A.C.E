"""Focused tests for staging the local ACE desktop runtime."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).parents[2] / "scripts/install/build_ace_backend.py"
_SPEC = importlib.util.spec_from_file_location("ace_backend_builder", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)


@pytest.fixture
def source_materials(tmp_path):
    python = tmp_path / "python"
    wheelhouse = tmp_path / "wheelhouse"
    files = (
        "python.exe",
        "pythonw.exe",
        "python313.dll",
        "vcruntime140.dll",
        "LICENSE.txt",
        "Lib/os.py",
        "Lib/__pycache__/os.pyc",
        "Lib/site-packages/pip/__init__.py",
        "Lib/site-packages/host_editable.pth",
        "DLLs/_sqlite3.pyd",
        "DLLs/sqlite3.dll",
        "obsidian.db",
    )
    for name in files:
        path = python / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test input", encoding="utf-8")
    wheelhouse.mkdir()
    for name in ("openjarvis", "fastapi", "uvicorn", "pydantic", "python_multipart"):
        (wheelhouse / f"{name}-1.0-py3-none-any.whl").touch()
    return python, wheelhouse


@pytest.fixture
def fake_subprocess(monkeypatch, source_materials):
    python, _ = source_materials
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-1] == builder._INSPECT_PYTHON:
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "prefix": str(python),
                        "implementation": "cpython",
                        "python": [3, 13],
                        "platform": "win-amd64",
                    }
                )
            )
        if "--target" in command:
            target = Path(command[command.index("--target") + 1])
            info = target / "openjarvis-1.0.dist-info"
            info.mkdir(parents=True)
            (info / "direct_url.json").write_text("absolute path", encoding="utf-8")
            (info / "METADATA").touch()
            (target / "openjarvis").mkdir()
            (target / "bin").mkdir()
            (target / "bin" / "jarvis.exe").touch()
        return SimpleNamespace()

    monkeypatch.setattr(builder.subprocess, "run", run)
    return calls


def test_stage_installs_local_wheels_and_excludes_host_state(
    tmp_path,
    source_materials,
    fake_subprocess,
):
    python, wheels = source_materials
    output = tmp_path / "runtime"
    builder.stage_backend(python, wheels, output)

    assert (output / "python.exe").is_file()
    assert (output / "Lib" / "os.py").is_file()
    assert (output / "DLLs" / "sqlite3.dll").is_file()
    assert not (output / "obsidian.db").exists()
    assert not (output / "Lib" / "__pycache__").exists()
    packages = output / "Lib" / "site-packages"
    assert (packages / "openjarvis").is_dir()
    assert not (packages / "host_editable.pth").exists()
    assert not (packages / "pip").exists()
    assert not (packages / "bin").exists()
    assert not (packages / "openjarvis-1.0.dist-info" / "direct_url.json").exists()
    assert (packages / "configs/openjarvis/examples/ace.toml").is_file()

    install = fake_subprocess[1][0]
    assert {"--isolated", "--no-index", "--no-deps", "--no-cache-dir"} <= set(install)
    assert sorted(Path(arg).name for arg in install if arg.endswith(".whl")) == sorted(
        path.name for path in wheels.iterdir()
    )
    for command, options in fake_subprocess[2:]:
        assert "-I" in command and "-B" in command
        profile = Path(options["env"]["OPENJARVIS_HOME"])
        assert profile.name == "ace-smoke-home"
        assert not profile.is_relative_to(output.parent)
        assert "OPENJARVIS_CONFIG" not in options["env"]
        assert command[0] != str(python / "python.exe")
    assert fake_subprocess[-2][0][-6:] == [
        "--quiet",
        "init",
        "--preset",
        "ace",
        "--no-analytics",
        "--no-download",
    ]
    assert fake_subprocess[-1][0][-2:] == ["serve", "--help"]


def test_missing_or_duplicate_wheels_are_rejected(tmp_path, source_materials):
    _, wheelhouse = source_materials
    (wheelhouse / "python_multipart-1.0-py3-none-any.whl").unlink()
    with pytest.raises(
        ValueError, match="Missing required runtime wheels: python-multipart"
    ):
        builder._runtime_wheels(wheelhouse)
    (wheelhouse / "pydantic-2.0-py3-none-any.whl").touch()
    with pytest.raises(ValueError, match="Provide just one runtime wheel for pydantic"):
        builder._runtime_wheels(wheelhouse)


def test_nested_installer_from_a_wheel_is_rejected(
    tmp_path, source_materials, fake_subprocess, monkeypatch
):
    python, wheelhouse = source_materials
    output = tmp_path / "runtime"
    fake_run = builder.subprocess.run

    def install_with_binary(command, **kwargs):
        result = fake_run(command, **kwargs)
        if "--target" in command:
            target = Path(command[command.index("--target") + 1])
            deploy = target / "_deploy" / "windows"
            deploy.mkdir(parents=True)
            (deploy / "Instalar ACE Offline.exe").touch()
        return result

    monkeypatch.setattr(builder.subprocess, "run", install_with_binary)
    with pytest.raises(ValueError, match="Windows installer was included"):
        builder.stage_backend(python, wheelhouse, output)
    assert not output.exists()


def test_existing_destination_is_preserved(tmp_path, source_materials):
    python, wheels = source_materials
    output = tmp_path / "runtime"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing work", encoding="utf-8")
    with pytest.raises(FileExistsError):
        builder.stage_backend(python, wheels, output)
    assert sentinel.read_text(encoding="utf-8") == "existing work"


def test_failure_never_publishes_partial_backend(
    tmp_path,
    monkeypatch,
    source_materials,
    fake_subprocess,
):
    python, wheels = source_materials
    output = tmp_path / "runtime"

    def failed_validation(*args):
        raise subprocess.CalledProcessError(1, "python smoke")

    monkeypatch.setattr(builder, "_validate", failed_validation)
    with pytest.raises(subprocess.CalledProcessError):
        builder.stage_backend(python, wheels, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".ace-backend-*"))
