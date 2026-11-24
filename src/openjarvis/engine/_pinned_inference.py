"""Release narrowly scoped inference requests to a verified loopback runtime."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx


def _loopback_url(host: str) -> str | None:
    try:
        url = urlsplit(host)
        if (
            url.scheme not in {"http", "https"}
            or url.username is not None
            or url.password is not None
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
        ):
            return None
        port = url.port
        hostname = url.hostname or ""
        address = ipaddress.ip_address(
            "127.0.0.1" if hostname.casefold() == "localhost" else hostname
        )
    except ValueError:
        return None
    if port == 0 or not address.is_loopback:
        return None
    endpoint = f"[{address}]" if address.version == 6 else str(address)
    return f"{url.scheme}://{endpoint}{f':{port}' if port else ''}"


@contextmanager
def pinned_local_inference(
    state: Any,
    *,
    capability: Literal["completion", "vision"],
) -> Iterator[tuple[Any, str] | None]:
    """Use one concrete local model and a client without proxies or redirects."""
    from openjarvis.engine._local_failover import LocalFailoverEngine
    from openjarvis.engine.ollama import OllamaEngine
    from openjarvis.engine.openai_compat_engines import LlamaCppEngine, LMStudioEngine
    from openjarvis.security.guardrails import GuardrailsEngine
    from openjarvis.telemetry.instrumented_engine import InstrumentedEngine

    model = getattr(state, "model", None)
    if not isinstance(model, str) or not (model := model.strip()):
        yield None
        return

    engine = getattr(state, "engine", None)
    guardrails = None
    for _ in range(8):
        if type(engine) is InstrumentedEngine:
            engine = engine._inner
        elif type(engine) is GuardrailsEngine and guardrails is None:
            guardrails = engine
            engine = engine._engine
        elif type(engine) is LocalFailoverEngine:
            engine = engine._primary
        else:
            break

    if (
        type(engine) not in (OllamaEngine, LlamaCppEngine, LMStudioEngine)
        or not (host := _loopback_url(engine._host))
        or model.casefold().removesuffix(":local").endswith((":cloud", "-cloud"))
    ):
        yield None
        return

    if type(engine) is OllamaEngine:
        model = model if model.endswith(":local") else model + ":local"
        direct = OllamaEngine(
            host=host, timeout=180.0, trust_env=False, num_ctx=engine._num_ctx
        )
    elif capability == "completion":
        direct = type(engine)(
            host=host, timeout=180.0, trust_env=False, api_key=engine._api_key
        )
    else:
        yield None
        return

    try:
        try:
            if type(direct) is OllamaEngine:
                available = (
                    direct.supports_local_vision(model)
                    if capability == "vision"
                    else direct.supports_local_model(model, capability=capability)
                )
            else:
                response = direct._client.get(
                    f"{direct._api_prefix}/models", timeout=5.0, follow_redirects=False
                )
                response.raise_for_status()
                models = response.json().get("data", [])
                model_id = direct._resolve_model_id(model)
                available = any(item.get("id") == model_id for item in models)
        except (httpx.HTTPError, AttributeError, TypeError, ValueError):
            available = False
        if not available:
            yield None
            return
        if guardrails is not None:
            # Keep the configured scanners without audit or telemetry hooks.
            local_scanner = copy(guardrails)
            local_scanner._engine = direct
            local_scanner._bus = None
            yield local_scanner, model
        else:
            yield direct, model
    finally:
        direct.close()
