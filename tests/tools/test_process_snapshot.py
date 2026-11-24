"""Privacy and bounds tests for local process diagnostics."""

from __future__ import annotations

import psutil

from openjarvis.tools.process_snapshot import sample_process_load


class _FakeProcess:
    def __init__(self, pid: int, name: str, cpu: float, rss: int) -> None:
        self.info = {"pid": pid, "name": name, "memory_info": None}
        self._cpu_values = iter((0.0, cpu))
        self._rss = rss

    def cpu_percent(self, interval=None) -> float:  # noqa: ANN001
        return next(self._cpu_values)

    def memory_info(self):  # noqa: ANN201
        return type("Memory", (), {"rss": self._rss})()

    def cmdline(self):
        raise AssertionError("process command lines must not be read")

    def exe(self):
        raise AssertionError("process executable paths must not be read")


def test_automatic_snapshot_omits_pids_and_private_process_fields(monkeypatch) -> None:
    processes = [
        _FakeProcess(11, "worker\x1b[31m", 74.2, 80 * 1024**2),
        _FakeProcess(12, "browser", 12.1, 500 * 1024**2),
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda **kwargs: processes)
    monkeypatch.setattr("openjarvis.tools.process_snapshot.time.sleep", lambda _: None)

    snapshot = sample_process_load(top_n=99, include_pids=False)

    assert snapshot["processes_sampled"] == 2
    assert len(snapshot["top_cpu"]) == 2
    assert snapshot["top_cpu"][0]["name"] == "worker[31m"
    assert snapshot["top_cpu"][0]["cpu_percent"] == 74.2
    assert snapshot["top_memory"][0]["name"] == "browser"
    assert all("pid" not in process for process in snapshot["top_cpu"])
    assert all("pid" not in process for process in snapshot["top_memory"])
    assert "command_line" in snapshot["omitted_fields"]
    assert "executable_path" in snapshot["omitted_fields"]


def test_process_snapshot_bounds_top_n_and_includes_pids(monkeypatch) -> None:
    processes = [
        _FakeProcess(index, f"p{index}", float(index), index * 1024**2)
        for index in range(1, 13)
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda **kwargs: processes)
    monkeypatch.setattr("openjarvis.tools.process_snapshot.time.sleep", lambda _: None)

    result = sample_process_load(top_n=1000, include_pids=True)

    assert len(result["top_cpu"]) == 10
    assert len(result["top_memory"]) == 10
    assert all("pid" in process for process in result["top_cpu"])
