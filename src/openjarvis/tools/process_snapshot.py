"""Bounded, read-only process load samples for diagnosis and incident context."""

from __future__ import annotations

import time
from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

_SAMPLE_SECONDS = 0.25
_MAX_PROCESSES_SCANNED = 4096


def _safe_process_name(value: object) -> str:
    """Keep process names printable and bounded before passing them to a model."""
    name = "".join(char for char in str(value) if char.isprintable()).strip()
    return name[:128] or "unknown"


def sample_process_load(
    top_n: int = 5,
    *,
    include_pids: bool = True,
) -> dict[str, Any]:
    """Collect a short, bounded ranking of process CPU and memory use.

    The caller can omit PIDs for automatic incident context. This helper never
    reads command lines, executable paths, environment variables, or files.
    """
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError(
            "Process inspection is unavailable: install psutil."
        ) from exc

    try:
        top_n = min(max(int(top_n), 1), 10)
    except (TypeError, ValueError):
        top_n = 5

    processes = []
    inaccessible = 0
    try:
        scanned = 0
        for process in psutil.process_iter(
            attrs=["pid", "name", "memory_info"], ad_value=None
        ):
            if scanned >= _MAX_PROCESSES_SCANNED:
                break
            scanned += 1
            try:
                if not process.info.get("name"):
                    continue
                process.cpu_percent(interval=None)
                processes.append(process)
            except (psutil.Error, OSError):
                inaccessible += 1

        time.sleep(_SAMPLE_SECONDS)
        rows = []
        for process in processes:
            try:
                info = process.info
                memory = process.memory_info()
                row = {
                    "name": _safe_process_name(info.get("name")),
                    "cpu_percent": round(float(process.cpu_percent()), 1),
                    "memory_mib": round(memory.rss / 1024**2, 1),
                }
                if include_pids:
                    row["pid"] = int(info["pid"])
                rows.append(row)
            except (psutil.Error, OSError, TypeError, ValueError):
                inaccessible += 1
    except (psutil.Error, OSError) as exc:
        raise RuntimeError(
            f"Could not read the local process list: {type(exc).__name__}."
        ) from exc

    return {
        "sample_seconds": _SAMPLE_SECONDS,
        "processes_sampled": len(rows),
        "processes_inaccessible": inaccessible,
        "top_cpu": sorted(rows, key=lambda item: item["cpu_percent"], reverse=True)[
            :top_n
        ],
        "top_memory": sorted(rows, key=lambda item: item["memory_mib"], reverse=True)[
            :top_n
        ],
        "omitted_fields": [
            "command_line",
            "executable_path",
            "environment",
            "file_contents",
        ],
    }


@ToolRegistry.register("process_snapshot")
class ProcessSnapshotTool(BaseTool):
    """Summarize local process load without reading command lines or file paths."""

    tool_id = "process_snapshot"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="process_snapshot",
            description=(
                "On explicit request, inspect a short local process sample and "
                "report the busiest processes by CPU and resident memory. Returns "
                "only process name, PID, CPU percent, and memory MiB. It never "
                "reads command lines, executable paths, environment variables, "
                "file contents, or network data, and never starts or stops "
                "a process."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of processes per ranking (1-10).",
                        "minimum": 1,
                        "maximum": 10,
                    }
                },
                "required": [],
            },
            category="system",
        )

    def execute(self, **params: Any) -> ToolResult:
        try:
            top_n = min(max(int(params.get("top_n", 5)), 1), 10)
        except (TypeError, ValueError):
            top_n = 5
        try:
            snapshot = sample_process_load(top_n, include_pids=True)
        except RuntimeError as exc:
            return ToolResult(
                tool_name=self.spec.name,
                content=str(exc),
                success=False,
            )
        by_cpu = snapshot["top_cpu"]
        by_memory = snapshot["top_memory"]

        def render(title: str, entries: list[dict[str, Any]]) -> list[str]:
            if not entries:
                return [f"{title}: no process data available"]
            return [
                title,
                *[
                    f"- {entry['name']} (PID {entry['pid']}): "
                    f"CPU {entry['cpu_percent']:.1f}%, "
                    f"memory {entry['memory_mib']:.1f} MiB"
                    for entry in entries
                ],
            ]

        metadata = snapshot
        content = "\n".join(
            [
                "Read-only local process snapshot (CPU is a short 0.25-second sample).",
                *render("Highest CPU:", by_cpu),
                *render("Highest resident memory:", by_memory),
                f"Processes sampled: {snapshot['processes_sampled']}; "
                f"inaccessible or exited: {snapshot['processes_inaccessible']}.",
                "This snapshot is a clue for diagnosis, not proof of cause.",
            ]
        )
        return ToolResult(
            tool_name=self.spec.name,
            content=content,
            success=True,
            metadata=metadata,
        )


__all__ = ["ProcessSnapshotTool", "sample_process_load"]
