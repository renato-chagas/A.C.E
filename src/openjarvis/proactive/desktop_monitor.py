"""Manage explicitly started dashboard monitoring without changing the profile."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openjarvis.core.config import ProactiveConfig
from openjarvis.proactive.monitor_daemon import (
    MonitorStateConflict as DesktopMonitorConflict,
)
from openjarvis.proactive.monitor_daemon import (
    get_monitor_daemon_status,
    start_monitor_daemon,
    stop_monitor_daemon,
)

if TYPE_CHECKING:
    from openjarvis.core.config import JarvisConfig


class DesktopMonitorController:
    def __init__(self, config: JarvisConfig | None) -> None:
        proactive = config.proactive if config is not None else ProactiveConfig()
        self.interval_seconds = min(
            max(int(proactive.local_monitor_interval_seconds), 15), 3600
        )

    def status(self) -> tuple[bool, bool, bool, bool]:
        state = get_monitor_daemon_status(read_only=True)
        if state is None:
            return False, False, False, False
        phase = state["status"]
        desktop = state.get("desktop_resource_only", False)
        return (
            phase != "stopping",
            phase in {"running", "stopping"},
            not desktop,
            desktop
            and phase != "stopping"
            and state.get("desktop_notifications_enabled", False),
        )

    def set_enabled(
        self, enabled: bool, *, notifications_enabled: bool = False
    ) -> None:
        if enabled:
            start_monitor_daemon(
                self.interval_seconds,
                desktop_resource_only=True,
                desktop_notifications_enabled=notifications_enabled,
            )
        else:
            stop_monitor_daemon(only_desktop=True)


__all__ = ["DesktopMonitorConflict", "DesktopMonitorController"]
