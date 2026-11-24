"""Security protections remain available without the optional Rust module."""

from __future__ import annotations

from openjarvis import _rust_bridge
from openjarvis.security import capabilities as capabilities_module
from openjarvis.security import rate_limiter as rate_limiter_module
from openjarvis.security import scanner as scanner_module


def _missing_extension():
    raise ImportError("openjarvis_rust is not installed")


def test_secret_and_pii_scanners_fall_back_to_python(monkeypatch):
    monkeypatch.setattr(scanner_module, "get_rust_module", _missing_extension)

    secret = scanner_module.SecretScanner()
    pii = scanner_module.PIIScanner()

    secret_result = secret.scan("key sk-abc123def456ghi789jkl012")
    assert [finding.pattern_name for finding in secret_result.findings] == [
        "openai_key"
    ]
    assert secret.redact("key sk-abc123def456ghi789jkl012") == (
        "key [REDACTED:openai_key]"
    )

    pii_result = pii.scan("server 8.8.8.8; invalid 999.1.1.1; build 1.2.3.4.5")
    assert [finding.pattern_name for finding in pii_result.findings] == [
        "ipv4_address"
    ]
    assert pii.redact("server 8.8.8.8") == "server [REDACTED:ipv4_address]"


def test_capability_policy_falls_back_to_python(monkeypatch):
    monkeypatch.setattr(_rust_bridge, "get_rust_module", _missing_extension)

    policy = capabilities_module.CapabilityPolicy(default_deny=True)
    policy.grant("_default", "memory:read", "projects/*")

    assert policy.check("agent-1", "memory:read", "projects/ace")
    assert not policy.check("agent-1", "memory:read", "private/ace")
    assert not policy.check("agent-1", "Memory:read", "projects/ace")
    policy.deny("_default", "memory:read")
    assert not policy.check("agent-1", "memory:read", "projects/ace")


def test_rate_limiter_falls_back_to_python_and_resets(monkeypatch):
    monkeypatch.setattr(_rust_bridge, "get_rust_module", _missing_extension)
    limiter = rate_limiter_module.RateLimiter(
        rate_limiter_module.RateLimitConfig(requests_per_minute=1, burst_size=1)
    )

    assert limiter.check("agent-1") == (True, 0.0)
    allowed, wait = limiter.check("agent-1")
    assert not allowed
    assert wait > 0

    limiter.reset("agent-1")
    assert limiter.check("agent-1") == (True, 0.0)
