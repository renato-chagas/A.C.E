from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, Mock

from openjarvis.core.types import Message, Role
from openjarvis.engine._base import InferenceEngine
from openjarvis.engine._local_failover import LocalFailoverEngine
from openjarvis.engine._locality import is_local_engine, is_local_engine_for_model
from openjarvis.engine._stubs import StreamChunk


def _engine(
    *,
    healthy: bool = True,
    models: list[str] | None = None,
    is_cloud: bool = False,
) -> MagicMock:
    engine = MagicMock(spec=InferenceEngine)
    engine.is_cloud = is_cloud
    engine.is_local = not is_cloud
    engine.health.return_value = healthy
    engine.list_models.return_value = models or ["local-model"]
    engine.can_serve.return_value = True
    return engine


def test_failed_primary_refreshes_local_alternatives() -> None:
    primary = _engine(healthy=False)
    fallback = _engine()
    primary.generate.side_effect = RuntimeError("primary unavailable")
    fallback.generate.return_value = {"content": "recovered", "usage": {}}

    engine = LocalFailoverEngine(
        primary,
        [],
        primary_key="ollama",
        refresh_alternatives=lambda: [("lmstudio", fallback)],
    )
    result = engine.generate([Message(role=Role.USER, content="hello")], model="qwen")

    assert result["content"] == "recovered"
    fallback.generate.assert_called_once()


def test_failover_does_not_choose_an_unrelated_installed_model() -> None:
    primary = _engine(healthy=False)
    alternative = _engine(models=["unrelated:1b"])
    primary.generate.side_effect = RuntimeError("primary unavailable")
    alternative.can_serve.side_effect = lambda model: model == "unrelated:1b"

    engine = LocalFailoverEngine(
        primary,
        [("lmstudio", alternative)],
        primary_key="ollama",
    )

    try:
        engine.generate([Message(role=Role.USER, content="hello")], model="chosen:7b")
    except RuntimeError as exc:
        assert "primary unavailable" in str(exc)
    else:
        raise AssertionError("failover unexpectedly selected an unrelated model")

    alternative.generate.assert_not_called()


def test_failover_uses_only_an_explicitly_configured_fallback_model() -> None:
    primary = _engine(healthy=False)
    alternative = _engine(models=["backup:4b"])
    primary.generate.side_effect = RuntimeError("primary unavailable")
    alternative.can_serve.side_effect = lambda model: model == "backup:4b"
    alternative.generate.return_value = {"content": "recovered", "usage": {}}

    engine = LocalFailoverEngine(
        primary,
        [("lmstudio", alternative)],
        fallback_model="backup:4b",
        primary_key="ollama",
    )

    result = engine.generate(
        [Message(role=Role.USER, content="hello")], model="chosen:7b"
    )

    assert result["content"] == "recovered"
    assert alternative.generate.call_args.kwargs["model"] == "backup:4b"


def test_alternative_retries_with_explicit_backup_when_chosen_model_fails() -> None:
    primary = _engine(healthy=False)
    alternative = _engine(models=["chosen:7b", "backup:4b"])
    primary.generate.side_effect = RuntimeError("primary unavailable")
    alternative.generate.side_effect = [
        RuntimeError("selected model could not load"),
        {"content": "recovered", "usage": {}},
    ]
    engine = LocalFailoverEngine(
        primary,
        [("lmstudio", alternative)],
        fallback_model="backup:4b",
        primary_key="ollama",
    )

    result = engine.generate(
        [Message(role=Role.USER, content="hello")], model="chosen:7b"
    )

    assert result["content"] == "recovered"
    assert [call.kwargs["model"] for call in alternative.generate.call_args_list] == [
        "chosen:7b",
        "backup:4b",
    ]


def test_opted_in_runtime_recovery_retries_sync_primary_once() -> None:
    primary = _engine(healthy=False)
    primary.generate.side_effect = [
        RuntimeError("runtime stopped"),
        {"content": "runtime recovered", "usage": {}},
    ]
    recover = Mock(return_value=True)
    engine = LocalFailoverEngine(
        primary,
        [],
        primary_key="ollama",
        recover_primary=recover,
    )

    result = engine.generate([Message(role=Role.USER, content="hello")], model="qwen")

    assert result["content"] == "runtime recovered"
    assert primary.generate.call_count == 2
    recover.assert_called_once_with()


def test_runtime_recovery_does_not_restart_a_healthy_primary() -> None:
    primary = _engine(healthy=True)
    primary.generate.side_effect = RuntimeError("model request rejected")
    recover = Mock(return_value=True)
    engine = LocalFailoverEngine(
        primary,
        [],
        primary_key="ollama",
        recover_primary=recover,
    )

    try:
        engine.generate([Message(role=Role.USER, content="hello")], model="qwen")
    except RuntimeError as exc:
        assert "model request rejected" in str(exc)
    else:
        raise AssertionError("failed generation unexpectedly recovered")

    recover.assert_not_called()


def test_failed_runtime_recovery_is_rate_limited() -> None:
    primary = _engine(healthy=False)
    primary.generate.side_effect = RuntimeError("runtime stopped")
    recover = Mock(return_value=False)
    engine = LocalFailoverEngine(
        primary,
        [],
        primary_key="ollama",
        recover_primary=recover,
        recovery_cooldown_seconds=60.0,
    )

    for _ in range(2):
        try:
            engine.generate([Message(role=Role.USER, content="hello")], model="qwen")
        except RuntimeError as exc:
            assert "runtime stopped" in str(exc)
        else:
            raise AssertionError("unavailable runtime unexpectedly generated")

    recover.assert_called_once_with()


def test_opted_in_runtime_recovery_retries_async_stream() -> None:
    primary = _engine(healthy=False)
    stream_calls = 0

    async def _stream(*args, **kwargs):
        nonlocal stream_calls
        stream_calls += 1
        if stream_calls == 1:
            raise RuntimeError("runtime stopped")
        yield "runtime recovered"

    primary.stream = Mock(side_effect=lambda *args, **kwargs: _stream())
    recover = Mock(return_value=True)
    engine = LocalFailoverEngine(
        primary,
        [],
        primary_key="ollama",
        recover_primary=recover,
    )

    async def _collect() -> list[str]:
        return [
            token
            async for token in engine.stream(
                [Message(role=Role.USER, content="hello")], model="qwen"
            )
        ]

    assert asyncio.run(_collect()) == ["runtime recovered"]
    assert stream_calls == 2
    recover.assert_called_once_with()


def test_opted_in_runtime_recovery_retries_full_stream() -> None:
    primary = _engine(healthy=False)
    stream_calls = 0

    async def _stream_full(*args, **kwargs):
        nonlocal stream_calls
        stream_calls += 1
        if stream_calls == 1:
            raise RuntimeError("runtime stopped")
        yield StreamChunk(content="runtime recovered")

    primary.stream_full = Mock(side_effect=lambda *args, **kwargs: _stream_full())
    recover = Mock(return_value=True)
    engine = LocalFailoverEngine(
        primary,
        [],
        primary_key="ollama",
        recover_primary=recover,
    )

    async def _collect() -> list[StreamChunk]:
        return [
            chunk
            async for chunk in engine.stream_full(
                [Message(role=Role.USER, content="hello")], model="qwen"
            )
        ]

    chunks = asyncio.run(_collect())
    assert [chunk.content for chunk in chunks] == ["runtime recovered"]
    assert stream_calls == 2
    recover.assert_called_once_with()


def test_local_failover_never_uses_cloud_engines() -> None:
    primary = _engine(healthy=False)
    cloud = _engine(is_cloud=True)
    primary.generate.side_effect = RuntimeError("primary unavailable")

    engine = LocalFailoverEngine(
        primary,
        [],
        primary_key="ollama",
        refresh_alternatives=lambda: [("cloud", cloud)],
    )

    try:
        engine.generate([Message(role=Role.USER, content="hello")], model="qwen")
    except RuntimeError as exc:
        assert "primary unavailable" in str(exc)
    else:
        raise AssertionError("generation unexpectedly used a cloud engine")

    cloud.generate.assert_not_called()
    cloud.close.assert_called_once()


def test_refresh_does_not_duplicate_an_healthy_cached_engine() -> None:
    primary = _engine(healthy=False)
    cached = _engine()
    discovered = _engine()
    engine = LocalFailoverEngine(
        primary,
        [("lmstudio", cached)],
        primary_key="ollama",
        refresh_alternatives=lambda: [("lmstudio", discovered)],
    )

    engine._refresh_local_alternatives()

    assert engine._alternatives == [("lmstudio", cached)]
    discovered.close.assert_called_once()


def test_refresh_replaces_unhealthy_cached_engine_without_duplicates() -> None:
    primary = _engine(healthy=False)
    stale = _engine(healthy=False)
    recovered = _engine()
    later_discovery = _engine()
    discoveries = iter((recovered, later_discovery))
    engine = LocalFailoverEngine(
        primary,
        [("lmstudio", stale)],
        primary_key="ollama",
        refresh_alternatives=lambda: [("lmstudio", next(discoveries))],
    )

    engine._refresh_local_alternatives()

    assert engine._alternatives == [("lmstudio", recovered)]
    stale.close.assert_called_once()
    recovered.close.assert_not_called()

    engine._refresh_local_alternatives()

    assert engine._alternatives == [("lmstudio", recovered)]
    later_discovery.close.assert_called_once()
    recovered.close.assert_not_called()


def test_locality_requires_a_loopback_host_for_http_engines() -> None:
    loopback = _engine()
    loopback._host = "http://127.0.0.1:8000/v1"
    remote = _engine()
    remote._host = "https://inference.example/v1"
    remote.is_local = True
    cloud_loopback = _engine(is_cloud=True)
    cloud_loopback._host = "http://localhost:8000"

    assert is_local_engine(loopback, "vllm")
    assert not is_local_engine(remote, "vllm")
    assert not is_local_engine(cloud_loopback, "vllm")
    assert is_local_engine(_engine(), "gemma_cpp")


def test_failover_rejects_remote_primary_and_filters_remote_alternatives() -> None:
    primary = _engine()
    remote_primary = _engine()
    remote_primary._host = "https://api.example/v1"
    try:
        LocalFailoverEngine(remote_primary, [], primary_key="vllm")
    except ValueError as exc:
        assert "verified local primary" in str(exc)
    else:
        raise AssertionError("remote primary was accepted")

    remote = _engine()
    remote._host = "https://api.example/v1"
    fallback = _engine()
    engine = LocalFailoverEngine(
        primary,
        [("remote-vllm", remote), ("local-lmstudio", fallback)],
        primary_key="ollama",
    )
    assert engine._alternatives == [("local-lmstudio", fallback)]
    remote.close.assert_called_once()


def test_failover_preserves_explicit_locality_for_custom_in_process_engine() -> None:
    primary = _engine()
    primary.engine_id = "custom-local-engine"
    primary.is_local = True
    engine = LocalFailoverEngine(primary, [], primary_key="custom-local-engine")

    assert is_local_engine(engine)
    assert is_local_engine_for_model(engine, "custom-model")
