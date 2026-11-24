from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

from openjarvis.core.config import JarvisConfig
from openjarvis.system.builder import SystemBuilder


def test_system_builder_binds_scheduler_without_enabling_proactive_agent(
    monkeypatch,
) -> None:
    config = JarvisConfig()
    scheduler = MagicMock()
    system = SimpleNamespace(scheduler=scheduler)
    builder = SystemBuilder(config)
    proactive_module = ModuleType("openjarvis.agents.proactive_agent")
    proactive_module.register_cron = MagicMock()
    monkeypatch.setitem(sys.modules, proactive_module.__name__, proactive_module)

    builder._configure_scheduler(system)

    scheduler.set_system.assert_called_once_with(system)
    proactive_module.register_cron.assert_not_called()


def test_system_builder_schedules_proactive_agent_only_after_explicit_enable(
    monkeypatch,
) -> None:
    config = JarvisConfig()
    config.proactive.enabled = True
    scheduler = MagicMock()
    system = SimpleNamespace(scheduler=scheduler)
    builder = SystemBuilder(config)
    proactive_module = ModuleType("openjarvis.agents.proactive_agent")
    proactive_module.register_cron = MagicMock(return_value=SimpleNamespace(id="daily"))
    monkeypatch.setitem(sys.modules, proactive_module.__name__, proactive_module)

    builder._configure_scheduler(system)

    scheduler.set_system.assert_called_once_with(system)
    proactive_module.register_cron.assert_called_once_with(scheduler)
