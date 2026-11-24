"""Read-only local host diagnostics for personal assistant profiles."""

from __future__ import annotations

import os
import platform
import shutil
import uuid
from pathlib import Path
from typing import Any

from openjarvis.core.paths import get_config_dir
from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


def _memory_bytes() -> tuple[int, int] | None:
    """Return total and available memory where the OS exposes it locally."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    except ImportError:
        pass
    except Exception:
        pass

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys), int(status.ullAvailPhys)
        except Exception:
            return None
    elif hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total_pages = int(os.sysconf("SC_PHYS_PAGES"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            return page_size * total_pages, page_size * available_pages
        except (OSError, ValueError, TypeError):
            return None
    return None


def _nearest_existing(path: Path) -> Path:
    candidate = path.expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _graphics_adapters() -> list[str]:
    """Return present Windows display-adapter names without starting a process.

    This reports only the adapter descriptions exposed by Windows SetupAPI. It
    does not report dedicated VRAM or prove that an inference runtime is using
    a particular adapter.
    """
    if os.name != "nt":
        return []

    try:
        import ctypes
        from ctypes import wintypes

        class _Guid(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        class _DeviceInfoData(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("ClassGuid", _Guid),
                ("DevInst", wintypes.DWORD),
                ("Reserved", ctypes.c_void_p),
            ]

        display_class = _Guid.from_buffer_copy(
            uuid.UUID("4d36e968-e325-11ce-bfc1-08002be10318").bytes_le
        )
        setupapi = ctypes.WinDLL("setupapi", use_last_error=True)
        get_class_devs = setupapi.SetupDiGetClassDevsW
        get_class_devs.argtypes = [
            ctypes.POINTER(_Guid),
            wintypes.LPCWSTR,
            wintypes.HWND,
            wintypes.DWORD,
        ]
        get_class_devs.restype = ctypes.c_void_p
        enum_device = setupapi.SetupDiEnumDeviceInfo
        enum_device.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(_DeviceInfoData),
        ]
        enum_device.restype = wintypes.BOOL
        get_property = setupapi.SetupDiGetDeviceRegistryPropertyW
        get_property.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_DeviceInfoData),
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        get_property.restype = wintypes.BOOL
        destroy_list = setupapi.SetupDiDestroyDeviceInfoList
        destroy_list.argtypes = [ctypes.c_void_p]
        destroy_list.restype = wintypes.BOOL

        devices = get_class_devs(ctypes.byref(display_class), None, None, 2)
        if devices == ctypes.c_void_p(-1).value:
            return []
    except Exception:
        return []

    names: list[str] = []
    seen: set[str] = set()
    try:
        for index in range(32):
            device = _DeviceInfoData()
            device.cbSize = ctypes.sizeof(device)
            if not enum_device(devices, index, ctypes.byref(device)):
                break
            description = ctypes.create_unicode_buffer(512)
            property_type = wintypes.DWORD()
            required_size = wintypes.DWORD()
            if not get_property(
                devices,
                ctypes.byref(device),
                0,  # SPDRP_DEVICEDESC
                ctypes.byref(property_type),
                description,
                ctypes.sizeof(description),
                ctypes.byref(required_size),
            ):
                continue
            name = description.value.strip()
            key = name.casefold()
            if name and key not in seen:
                names.append(name)
                seen.add(key)
    except Exception:
        return names
    finally:
        destroy_list(devices)
    return names


@ToolRegistry.register("system_health")
class SystemHealthTool(BaseTool):
    """Report local resources and Windows display-adapter names."""

    tool_id = "system_health"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="system_health",
            description=(
                "Read local OS, CPU, memory, disk-space, and runtime executable "
                "names on PATH. On Windows, report present display-adapter names. "
                "It reads no file contents, starts no process, and uses no network."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            category="system",
        )

    def execute(self, **params: Any) -> ToolResult:
        del params
        try:
            config_root = _nearest_existing(get_config_dir())
        except (OSError, RuntimeError):
            config_root = Path.cwd()
        try:
            disk = shutil.disk_usage(config_root)
        except OSError:
            config_root = Path.cwd()
            try:
                disk = shutil.disk_usage(config_root)
            except OSError:
                disk = None
        memory = _memory_bytes()
        graphics_adapters = _graphics_adapters()
        runtime_names = (
            "ollama",
            "llama-server",
            "vllm",
            "lemonade",
        )
        runtimes = {name: shutil.which(name) is not None for name in runtime_names}
        metadata: dict[str, Any] = {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "disk_path": str(config_root),
            "local_runtimes": runtimes,
            "graphics_adapters": graphics_adapters,
        }
        lines = [
            f"Operating system: {metadata['os']}",
            f"Python: {metadata['python']}",
            f"CPU cores reported by the OS: {metadata['cpu_count']}",
            "Local runtime executables on PATH: "
            + (
                ", ".join(name for name, found in runtimes.items() if found)
                or "none detected"
            ),
        ]
        if os.name == "nt":
            lines.append(
                "Graphics adapters reported by Windows: "
                + (", ".join(graphics_adapters) if graphics_adapters else "unavailable")
            )
        else:
            lines.append("Graphics adapters: unavailable from this local interface")
        if disk is None:
            metadata.pop("disk_path", None)
            lines.append("Disk space: unavailable from the current OS interface")
        else:
            metadata["disk_total_bytes"] = int(disk.total)
            metadata["disk_free_bytes"] = int(disk.free)
            lines.append(
                f"Disk free on {config_root}: {disk.free / 1024**3:.1f} GiB "
                f"of {disk.total / 1024**3:.1f} GiB"
            )
        if memory is None:
            lines.append("Memory: unavailable from the current OS interface")
        else:
            total, available = memory
            metadata["memory_total_bytes"] = total
            metadata["memory_available_bytes"] = available
            lines.append(
                f"Memory available: {available / 1024**3:.1f} GiB "
                f"of {total / 1024**3:.1f} GiB"
            )
        return ToolResult(
            tool_name=self.spec.name,
            content="\n".join(lines),
            success=True,
            metadata=metadata,
        )


__all__ = ["SystemHealthTool"]
