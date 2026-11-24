"""Stage a relocatable Windows x64 OpenJarvis backend using local wheels."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT = _REPO_ROOT / "frontend" / "src-tauri" / "binaries" / "ace-backend"
_PRESET_DIRECTORY = Path("configs/openjarvis/examples")
_CLI = "from openjarvis.cli import main; main()"

_INSPECT_PYTHON = """
import json, sys, sysconfig
print(json.dumps({
    "prefix": sys.base_prefix,
    "python": list(sys.version_info[:2]),
    "implementation": sys.implementation.name,
    "platform": sysconfig.get_platform(),
}))
"""

_RUN_PIP_CHECK = """
import runpy, sys
sys.path.append(sys.argv.pop(1))
runpy.run_module("pip", run_name="__main__")
"""

_SMOKE_TEST = """
import sqlite3, sys
from pathlib import Path
import fastapi, python_multipart, uvicorn
from openjarvis.server.app import create_app

root = Path(sys.executable).resolve().parent
if Path(sys.base_prefix).resolve() != root:
    raise RuntimeError("Python used a base installation outside the staged runtime.")
with sqlite3.connect(":memory:") as conn:
    conn.execute("CREATE VIRTUAL TABLE ace_smoke USING fts5(body)")
"""


def _inspect_python(python_root: Path) -> None:
    if (
        (python_root / "pyvenv.cfg").exists()
        or not (python_root / "Lib" / "os.py").is_file()
        or not (python_root / "DLLs").is_dir()
    ):
        raise ValueError(
            "Provide a full relocatable CPython directory with Lib and DLLs."
        )
    python = python_root / "python.exe"
    result = subprocess.run(
        [str(python), "-I", "-B", "-c", _INSPECT_PYTHON],
        cwd=python_root,
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    if (
        info["implementation"] != "cpython"
        or info["platform"] != "win-amd64"
        or not (3, 10) <= tuple(info["python"]) <= (3, 13)
        or Path(info["prefix"]).resolve() != python_root
    ):
        raise ValueError("The Python input must be Windows x64 CPython 3.10–3.13.")
    if not (python_root / "Lib" / "site-packages" / "pip" / "__init__.py").is_file():
        raise ValueError("The CPython input must supply pip for the staging tool.")


def _runtime_wheels(wheelhouse: Path) -> list[Path]:
    wheels = sorted(wheelhouse.glob("*.whl"))
    by_project: dict[str, Path] = {}
    for wheel in wheels:
        parts = wheel.name.removesuffix(".whl").split("-")
        if len(parts) not in (5, 6) or not wheel.is_file():
            raise ValueError(f"Invalid wheel filename: {wheel.name}")
        name = re.sub(r"[-_.]+", "-", parts[0]).lower()
        if name in by_project:
            raise ValueError(f"Provide just one runtime wheel for {name}.")
        by_project[name] = wheel
    required = {"openjarvis", "fastapi", "uvicorn", "pydantic", "python-multipart"}
    missing = sorted(required - by_project.keys())
    if missing:
        raise ValueError("Missing required runtime wheels: " + ", ".join(missing))
    return wheels


def _copy_python(python_root: Path, stage: Path) -> None:
    stage.mkdir()
    for name in ("python.exe", "pythonw.exe", "LICENSE.txt"):
        shutil.copy2(python_root / name, stage / name)
    for pattern in ("python*.dll", "vcruntime*.dll"):
        for dll in python_root.glob(pattern):
            shutil.copy2(dll, stage / dll.name)
    exclude = shutil.ignore_patterns("site-packages", "__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(python_root / "Lib", stage / "Lib", ignore=exclude)
    shutil.copytree(python_root / "DLLs", stage / "DLLs", ignore=exclude)


def _stage_wheels(
    python_root: Path, wheels: list[Path], temp: Path, stage: Path
) -> None:
    unpacked = temp / "packages"
    subprocess.run(
        [
            str(python_root / "python.exe"),
            "-I",
            "-B",
            "-m",
            "pip",
            "--isolated",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "install",
            "--no-index",
            "--no-deps",
            "--ignore-installed",
            "--only-binary=:all:",
            "--no-compile",
            "--target",
            str(unpacked),
            *(str(wheel) for wheel in wheels),
        ],
        cwd=temp,
        check=True,
    )

    windows_deploy = unpacked / "_deploy" / "windows"
    if windows_deploy.is_dir() and any(
        path.is_file() and path.suffix.casefold() == ".exe"
        for path in windows_deploy.rglob("*")
    ):
        raise ValueError("A Windows installer was included in a runtime wheel.")

    # Generated script launchers embed the staging interpreter's absolute path.
    # The distributed CLI is invoked through its own python.exe.
    site_packages = stage / "Lib" / "site-packages"
    site_packages.mkdir()
    for child in unpacked.iterdir():
        if child.name in {"bin", "Scripts"}:
            continue
        if child.suffix == ".pth":
            raise ValueError(f"Runtime wheel installed a path override: {child.name}")
        if child.name.endswith(".dist-info"):
            (child / "direct_url.json").unlink(missing_ok=True)
        child.rename(site_packages / child.name)


def _validate(python_root: Path, temp: Path, stage: Path) -> None:
    python = str(stage / "python.exe")
    tools = temp / "build-tools"
    shutil.copytree(python_root / "Lib" / "site-packages" / "pip", tools / "pip")
    # The publish stage may live in the checkout; real profiles may not.
    with tempfile.TemporaryDirectory(prefix="ace-backend-profile-") as profile:
        profile_temp = Path(profile)
        env = {**os.environ, "OPENJARVIS_HOME": str(profile_temp / "ace-smoke-home")}
        env.pop("OPENJARVIS_CONFIG", None)
        subprocess.run(
            [
                python,
                "-I",
                "-B",
                "-c",
                _RUN_PIP_CHECK,
                str(tools),
                "--isolated",
                "--disable-pip-version-check",
                "check",
            ],
            cwd=temp,
            env=env,
            check=True,
        )
        for command in (
            ["-c", _SMOKE_TEST],
            [
                "-c",
                _CLI,
                "--quiet",
                "init",
                "--preset",
                "ace",
                "--no-analytics",
                "--no-download",
            ],
            ["-c", _CLI, "serve", "--help"],
        ):
            subprocess.run(
                [python, "-I", "-B", *command], cwd=temp, env=env, check=True
            )


def stage_backend(python_root: Path, wheelhouse: Path, output: Path) -> None:
    """Build in a sibling temporary directory and publish only after validation."""
    python_root = python_root.resolve(strict=True)
    wheelhouse = wheelhouse.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Destination already exists: {output}")
    output = output.resolve()
    if any(output.is_relative_to(source) for source in (python_root, wheelhouse)):
        raise ValueError(
            "The destination must be outside the Python input and wheelhouse."
        )
    _inspect_python(python_root)
    wheels = _runtime_wheels(wheelhouse)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ace-backend-", dir=output.parent) as work:
        temp = Path(work)
        stage = temp / "runtime"
        _copy_python(python_root, stage)
        _stage_wheels(python_root, wheels, temp, stage)
        presets = stage / "Lib" / "site-packages" / _PRESET_DIRECTORY
        presets.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO_ROOT / _PRESET_DIRECTORY / "ace.toml", presets / "ace.toml")
        _validate(python_root, temp, stage)
        stage.rename(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Run this builder on Windows.")
    try:
        stage_backend(args.python_root, args.wheelhouse, args.output)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Could not stage the ACE backend: {exc}", file=sys.stderr)
        return 1
    print(f"ACE backend staged at {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
