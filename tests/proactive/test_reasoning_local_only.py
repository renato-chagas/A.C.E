"""Privacy checks for model analysis of local monitor incidents."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openjarvis.proactive import monitor_daemon
from openjarvis.proactive.monitor_daemon import _ScreenAssessmentLog
from openjarvis.proactive.reasoning import (
    LocalIncidentReasoner,
    SerializedInferenceEngine,
    measured_incident_fallback,
)
from openjarvis.security.guardrails import GuardrailsEngine


def test_background_reasoner_uses_shared_local_engine_resolver(monkeypatch):
    import openjarvis.engine._discovery as discovery
    import openjarvis.proactive.reasoning as reasoning
    import openjarvis.security as security

    config = SimpleNamespace(
        proactive=SimpleNamespace(local_monitor_reasoning_enabled=True),
        intelligence=SimpleNamespace(
            provider="local",
            preferred_engine="ollama",
            default_model="primary-model",
            fallback_model="fallback-model",
        ),
        engine=SimpleNamespace(default="lmstudio"),
        memory=SimpleNamespace(default_backend=""),
    )
    engine = MagicMock()
    engine.engine_id = "ollama"
    engine.is_cloud = False
    engine.is_local = False
    engine._host = "http://127.0.0.1:11434"
    engine.health.return_value = True
    engine.list_models.return_value = ["primary-model"]
    engine.can_serve.side_effect = lambda model: model == "primary-model"

    calls = []

    def resolve(local_config):
        calls.append(local_config)
        return "ollama", engine

    reasoner_sentinel = object()
    reasoner_calls = []

    def make_reasoner(*args, **kwargs):
        reasoner_calls.append((args, kwargs))
        return reasoner_sentinel

    monkeypatch.setattr(discovery, "get_engine", resolve)
    monkeypatch.setattr(
        security,
        "setup_security",
        lambda config, selected: SimpleNamespace(engine=selected, audit_logger=None),
    )
    monkeypatch.setattr(reasoning, "LocalIncidentReasoner", make_reasoner)

    result, resources = monitor_daemon._build_local_reasoner(config)

    assert result is reasoner_sentinel
    assert resources == [engine]
    assert len(calls) == 1
    assert calls[0] is not config
    assert calls[0].intelligence.provider == "local"
    assert calls[0].engine.default == "ollama"
    assert config.engine.default == "lmstudio"
    assert reasoner_calls[0][0][:2] == (engine, "primary-model")


def test_background_screen_monitor_uses_verified_local_vision_route(monkeypatch):
    import openjarvis.engine._discovery as discovery
    import openjarvis.proactive.screen_monitor as screen_monitor
    import openjarvis.security as security

    config = SimpleNamespace(
        proactive=SimpleNamespace(
            local_screen_monitor_interval_seconds=30,
            local_screen_monitor_min_analysis_interval_seconds=120,
        ),
        intelligence=SimpleNamespace(
            provider="local",
            preferred_engine="ollama",
            default_model="vision-model",
            fallback_model="fallback-model",
        ),
        engine=SimpleNamespace(default="lmstudio"),
    )
    engine = MagicMock()
    engine.engine_id = "ollama"
    engine.is_cloud = False
    engine.is_local = False
    engine._host = "http://127.0.0.1:11434"
    engine.health.return_value = True
    engine.list_models.return_value = ["vision-model"]

    def can_serve(model):
        return model == "vision-model"

    engine.can_serve.side_effect = can_serve
    secured_engine = object()
    security_result = SimpleNamespace(engine=secured_engine, audit_logger=None)
    monitor_instance = object()
    monitor_calls = []

    monkeypatch.setattr(discovery, "get_engine", lambda _config: ("ollama", engine))
    monkeypatch.setattr(security, "setup_security", lambda *_args: security_result)

    def make_monitor(*args, **kwargs):
        monitor_calls.append((args, kwargs))
        return monitor_instance

    monkeypatch.setattr(screen_monitor, "LocalScreenMonitor", make_monitor)

    def assessment(_text):
        return None

    def error(_text):
        return None

    result, resources = monitor_daemon._build_local_screen_monitor(
        config,
        on_assessment=assessment,
        on_error=error,
    )

    assert result is monitor_instance
    assert resources == [secured_engine]
    assert config.engine.default == "lmstudio"
    assert monitor_calls[0][0][:2] == (secured_engine, "vision-model")
    assert monitor_calls[0][1]["interval_seconds"] == 30
    assert monitor_calls[0][1]["min_analysis_interval_seconds"] == 120
    assert monitor_calls[0][1]["on_assessment"] is assessment
    engine.generate.assert_not_called()


def test_background_screen_monitor_rejects_remote_route_before_health(monkeypatch):
    import openjarvis.engine._discovery as discovery
    import openjarvis.proactive.screen_monitor as screen_monitor

    config = SimpleNamespace(
        proactive=SimpleNamespace(
            local_screen_monitor_interval_seconds=30,
            local_screen_monitor_min_analysis_interval_seconds=120,
        ),
        intelligence=SimpleNamespace(
            provider="local",
            preferred_engine="remote-http",
            default_model="vision-model",
            fallback_model="",
        ),
        engine=SimpleNamespace(default="remote-http"),
    )
    engine = MagicMock()
    engine.engine_id = "custom-http"
    engine.is_cloud = False
    engine.is_local = True
    engine._host = "https://inference.example/v1"
    monitor_factory = MagicMock()

    monkeypatch.setattr(
        discovery,
        "get_engine",
        lambda _config: ("remote-http", engine),
    )
    monkeypatch.setattr(screen_monitor, "LocalScreenMonitor", monitor_factory)

    result, resources = monitor_daemon._build_local_screen_monitor(
        config,
        on_assessment=None,
        on_error=None,
    )

    assert result is None
    assert resources == []
    engine.health.assert_not_called()
    monitor_factory.assert_not_called()


def test_screen_assessment_log_strips_terminal_control_characters(capsys):
    write_assessment = _ScreenAssessmentLog(max_bytes=4096)

    write_assessment("Visible\x1b[31m text\x1b[0m")

    output = capsys.readouterr().out
    assert "Visible[31m text[0m" in output
    assert "\x1b" not in output


def test_screen_assessment_log_stops_persisting_at_its_budget(capsys):
    write_assessment = _ScreenAssessmentLog(max_bytes=0)

    write_assessment("private screen text")
    first_output = capsys.readouterr().out
    write_assessment("more private screen text")
    second_output = capsys.readouterr().out

    assert "private screen text" not in first_output
    assert "per-run limit" in first_output
    assert second_output == ""


def test_detached_command_includes_screen_flag_only_when_requested():
    screen_command = monitor_daemon._worker_command(screen_monitor_enabled=True)
    resource_command = monitor_daemon._worker_command(screen_monitor_enabled=False)

    assert screen_command[-1] == "--monitor-screen"
    assert "--monitor-screen" not in resource_command


def test_worker_state_validates_its_pid_and_launch_token(monkeypatch):
    import os
    from contextlib import nullcontext

    state = {
        "pid": os.getpid(),
        "launch_token": "private-token",
        "status": "starting",
    }
    monkeypatch.setattr(monitor_daemon, "_control_lock", lambda: nullcontext())
    monkeypatch.setattr(monitor_daemon, "_read_state_unlocked", lambda: state)
    monkeypatch.setattr(
        monitor_daemon,
        "_live_state_unlocked",
        lambda: pytest.fail("worker should not use external process verification"),
    )

    assert monitor_daemon._worker_state("private-token") == state
    assert monitor_daemon._worker_state("different-token") is None


def test_worker_claims_launcher_state_with_its_own_pid(monkeypatch):
    import os
    from contextlib import nullcontext

    state = {
        "pid": os.getppid(),
        "launch_token": "private-token",
        "status": "starting",
    }
    writes = []
    monkeypatch.setattr(monitor_daemon, "_control_lock", lambda: nullcontext())
    monkeypatch.setattr(monitor_daemon, "_read_state_unlocked", lambda: state)
    monkeypatch.setattr(monitor_daemon, "_state_path", lambda: "isolated-state")
    monkeypatch.setattr(
        monitor_daemon,
        "secure_write_json",
        lambda _path, payload: writes.append(dict(payload)),
    )

    result = monitor_daemon._set_worker_running("private-token")

    assert result is not None
    assert result["pid"] == os.getpid()
    assert result["status"] == "running"
    assert result["process_created"] > 0
    assert writes == [result]


def test_worker_logs_rejection_without_matching_launch_state(monkeypatch, capsys):
    import openjarvis.proactive.monitor_daemon as monitor_daemon

    monkeypatch.setattr(monitor_daemon, "_worker_state", lambda _token: None)

    assert monitor_daemon.run_monitor_worker("private-token") == 2
    assert "no matching active launch state" in capsys.readouterr().out


def test_worker_rejects_screen_mode_not_recorded_in_launch_state(monkeypatch):
    monkeypatch.setattr(
        monitor_daemon,
        "_worker_state",
        lambda _token: {
            "status": "running",
            "screen_monitor_enabled": False,
        },
    )

    assert monitor_daemon.run_monitor_worker("private-token", True) == 2


def test_reasoner_supervisor_retries_setup_after_local_runtime_returns(monkeypatch):
    current_time = [100.0]
    monkeypatch.setattr(
        monitor_daemon,
        "time",
        SimpleNamespace(monotonic=lambda: current_time[0]),
    )
    config = SimpleNamespace(
        proactive=SimpleNamespace(local_monitor_reasoning_enabled=True),
        intelligence=SimpleNamespace(provider="local"),
    )
    reasoner = object()
    resource = object()
    outcomes = [(None, []), (reasoner, [resource])]
    attempts = []

    def build(actual_config, *, monitor_store):
        attempts.append((actual_config, monitor_store))
        return outcomes.pop(0)

    monkeypatch.setattr(monitor_daemon, "_build_local_reasoner", build)
    store = object()
    supervisor = monitor_daemon._LocalReasonerSupervisor(
        config,
        store,
        retry_interval=30,
    )

    assert supervisor.initialize() is None
    assert supervisor.retry_if_due() is None
    assert len(attempts) == 1

    current_time[0] = 131.0
    assert supervisor.retry_if_due() is reasoner
    assert supervisor.reasoner is reasoner
    assert supervisor.resources == [resource]
    assert len(attempts) == 2
    assert all(call[0] is config and call[1] is store for call in attempts)


def test_pending_incident_gate_retries_with_cooldown_and_releases_success():
    gate = monitor_daemon._PendingIncidentGate(retry_interval=60)

    assert gate.try_queue(7, now=10)
    assert not gate.try_queue(7, now=10)
    gate.release(7, retry=True, now=11)
    assert not gate.try_queue(7, now=70)
    assert gate.try_queue(7, force=True, now=70)
    gate.release(7)
    assert gate.try_queue(7, now=71)
    gate.release(7)
    assert gate.try_queue(7, now=72)


def test_pending_incident_gate_discards_retry_after_incident_resolves():
    gate = monitor_daemon._PendingIncidentGate(retry_interval=60)
    assert gate.try_queue(9, now=10)
    gate.release(9, retry=True, now=11)

    gate.retain_active(set())

    assert gate.try_queue(9, now=12)


@pytest.mark.parametrize(
    ("metric", "details", "measured", "check"),
    [
        (
            "cpu_high",
            {"cpu_percent": 98.5, "limit_percent": 90, "process_name": "secret.exe"},
            "98.5%",
            "Task Manager",
        ),
        (
            "memory_low",
            {"available_percent": 7.5, "available_gb": 1.25},
            "1.2 GiB",
            "Save work",
        ),
        (
            "disk_low",
            {"free_percent": 4, "free_gb": 8.3},
            "8.3 GiB",
            "did not delete anything",
        ),
    ],
)
def test_rule_based_fallback_uses_measurements_and_read_only_checks(
    metric, details, measured, check
):
    result = measured_incident_fallback({"metric": metric, "details": details})

    assert "Rule-based local fallback (no language model; cause not inferred)" in result
    assert measured in result
    assert check in result
    assert "secret.exe" not in result
    assert "No host action or configuration change was made." in result
    assert "remains pending for local-model analysis" in result


def test_rule_based_fallback_ignores_unknown_metrics_and_non_finite_values():
    assert measured_incident_fallback({"metric": "unknown"}) == ""
    result = measured_incident_fallback(
        {"metric": "cpu_high", "details": {"cpu_percent": float("nan")}}
    )
    assert "nan" not in result
    assert "configured threshold" in result


def test_incident_reasoner_refuses_remote_endpoint() -> None:
    engine = MagicMock()
    engine.engine_id = "custom-http"
    engine.is_cloud = False
    engine.is_local = True
    engine._host = "https://inference.example/v1"
    config = SimpleNamespace(agent=SimpleNamespace(context_from_memory=False))

    with pytest.raises(ValueError, match="verified local"):
        LocalIncidentReasoner(engine, "model", config)

    engine.generate.assert_not_called()


def test_incident_reasoner_preserves_locality_through_security_wrappers() -> None:
    engine = MagicMock()
    engine.engine_id = "custom-local"
    engine.is_cloud = False
    engine.is_local = True
    engine._host = "http://127.0.0.1:1234"
    engine.generate.return_value = {"content": "Measure confirmed."}
    secured = GuardrailsEngine(engine, scanners=[])
    wrapped = SerializedInferenceEngine(secured)
    config = SimpleNamespace(agent=SimpleNamespace(context_from_memory=False))

    reasoner = LocalIncidentReasoner(wrapped, "model", config)
    result = reasoner.analyze({"metric": "cpu_high", "summary": "CPU high"})

    assert result == "Measure confirmed."
    messages = engine.generate.call_args.args[0]
    assert "dados não confiáveis" in messages[0].content
    assert '"metric": "cpu_high"' in messages[1].content
