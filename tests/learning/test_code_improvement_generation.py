"""Local patch generation reads only selected files and never chooses cloud."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

from openjarvis.learning.code_improvement import generate_local_patch


def test_local_patch_generation_forces_local_engine_and_hashes_exact_bytes(
    tmp_path, monkeypatch
) -> None:
    import openjarvis.core.config as config_module
    import openjarvis.engine._discovery as discovery_module
    import openjarvis.intelligence.model_catalog as catalog_module

    repository = tmp_path / "repo"
    source_path = repository / "src" / "sample.py"
    source_path.parent.mkdir(parents=True)
    source_bytes = b"def value():\r\n    return 1\r\n"
    source_path.write_bytes(source_bytes)
    config = SimpleNamespace(
        intelligence=SimpleNamespace(
            provider="cloud",
            default_model="local-model",
            fallback_model="fallback-model",
            max_tokens=1024,
        )
    )
    prompts = []

    class LocalEngine:
        is_cloud = False

        def list_models(self):
            return ["local-model"]

        def generate(self, messages, **kwargs):
            prompts.append((messages, kwargs))
            return {
                "content": (
                    "```diff\n--- a/src/sample.py\n+++ b/src/sample.py\n"
                    "@@ -1 +1 @@\n-old\n+new\n```"
                )
            }

        def close(self):
            self.closed = True

    engine = LocalEngine()
    observed_providers = []

    def resolve_engine(configured):
        observed_providers.append(configured.intelligence.provider)
        return "ollama", engine

    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(discovery_module, "get_configured_engine", resolve_engine)
    monkeypatch.setattr(catalog_module, "register_builtin_models", lambda: None)

    patch, hashes = generate_local_patch(
        "Change the returned value",
        ("src/sample.py",),
        repository,
    )

    assert observed_providers == ["local"]
    assert patch.startswith("--- a/src/sample.py")
    assert "```" not in patch
    assert hashes == {"src/sample.py": hashlib.sha256(source_bytes).hexdigest()}
    assert "return 1\n" in prompts[0][0][1].content
    assert engine.closed is True


def test_local_patch_generation_refuses_unselected_and_sensitive_paths(
    tmp_path, monkeypatch
) -> None:
    import openjarvis.core.config as config_module
    import openjarvis.engine._discovery as discovery_module
    import openjarvis.intelligence.model_catalog as catalog_module

    config = SimpleNamespace(
        intelligence=SimpleNamespace(
            provider="local",
            default_model="local-model",
            fallback_model="",
            max_tokens=1024,
        )
    )

    class LocalEngine:
        is_cloud = False

        def list_models(self):
            return ["local-model"]

        def generate(self, *_args, **_kwargs):
            raise AssertionError("model must not receive an unsafe path")

        def close(self):
            pass

    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(
        discovery_module,
        "get_configured_engine",
        lambda _config: ("ollama", LocalEngine()),
    )
    monkeypatch.setattr(catalog_module, "register_builtin_models", lambda: None)
    repository = tmp_path / "repo"
    (repository / ".ssh").mkdir(parents=True)
    (repository / ".ssh" / "id_rsa").write_text("private key", encoding="utf-8")

    try:
        generate_local_patch("Inspect selected source", (".ssh/id_rsa",), repository)
    except ValueError as exc:
        assert "sensitive" in str(exc).lower()
    else:
        raise AssertionError("sensitive paths must be refused")
