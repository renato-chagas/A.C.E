"""Engine resolution behavior for ``SystemBuilder``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from openjarvis.core.config import JarvisConfig
from openjarvis.system.builder import SystemBuilder


def test_local_preference_uses_shared_failover_resolver() -> None:
    config = JarvisConfig()
    config.intelligence.provider = "local"
    config.intelligence.preferred_engine = "ollama"
    engine = MagicMock()
    builder = SystemBuilder(config=config)

    with patch(
        "openjarvis.engine._discovery.get_configured_engine",
        return_value=("ollama", engine),
    ) as resolver:
        resolved_engine, resolved_key = builder._resolve_engine(config)

    assert resolved_engine is engine
    assert resolved_key == "ollama"
    resolver.assert_called_once_with(config, None, pin_default=True)


def test_local_profile_does_not_choose_first_installed_model() -> None:
    config = JarvisConfig()
    config.intelligence.provider = "local"
    config.intelligence.default_model = ""
    config.intelligence.fallback_model = ""
    engine = MagicMock()
    engine.list_models.return_value = ["installed-first", "installed-second"]

    model = SystemBuilder(config=config)._resolve_model(config, engine)

    assert model == ""
