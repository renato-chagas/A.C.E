"""Engine discovery — probe running engines and aggregate available models."""

from __future__ import annotations

import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import EngineRegistry
from openjarvis.engine._base import InferenceEngine
from openjarvis.engine._locality import LOCAL_ENGINE_KEYS, is_local_engine

logger = logging.getLogger(__name__)

# Map registry keys to config host attribute (None = no host arg)
_HOST_MAP: Dict[str, str | None] = {
    "ollama": "ollama_host",
    "vllm": "vllm_host",
    "llamacpp": "llamacpp_host",
    "sglang": "sglang_host",
    "mlx": "mlx_host",
    "lmstudio": "lmstudio_host",
    "exo": "exo_host",
    "nexa": "nexa_host",
    "uzu": "uzu_host",
    "apple_fm": "apple_fm_host",
    "lemonade": "lemonade_host",
    "cloud": None,
    "litellm": None,
    "gemma_cpp": None,
    # In-process: drives the Apple FM SDK directly, so there is no host.
    "afm": None,
}


def engine_host_config_field(key: str) -> str | None:
    """Return the config field used to configure this engine's HTTP endpoint."""
    return _HOST_MAP.get(key)


def _make_engine(key: str, config: JarvisConfig) -> InferenceEngine:
    """Instantiate a registered engine with the appropriate config host."""
    cls = EngineRegistry.get(key)
    local_only = (config.intelligence.provider or "").strip().lower() == "local"

    # LiteLLM cannot enumerate every model supported by every provider.  Its
    # list_models() contract therefore advertises the configured default
    # model, which must be supplied when discovery constructs the engine.
    if key == "litellm":
        return cls(default_model=config.intelligence.default_model or None)

    # gemma_cpp: pass config fields instead of host
    if key == "gemma_cpp":
        cfg = config.engine.gemma_cpp
        return cls(
            model_path=cfg.model_path or None,
            tokenizer_path=cfg.tokenizer_path or None,
            model_type=cfg.model_type or None,
            num_threads=cfg.num_threads,
        )

    # afm: in-process engine, configured by behaviour rather than a host
    if key == "afm":
        cfg = config.engine.afm
        return cls(
            instructions=cfg.instructions,
            use_case=cfg.use_case,
            guardrails=cfg.guardrails,
            sampling=cfg.sampling,
        )

    host_attr = engine_host_config_field(key)
    if host_attr is not None:
        host = getattr(config.engine, host_attr, None)
        options: dict[str, Any] = {}
        if key == "ollama" and config.engine.ollama.num_ctx > 0:
            options["num_ctx"] = config.engine.ollama.num_ctx
        if local_only and key in LOCAL_ENGINE_KEYS:
            # Do not let system proxy variables route a loopback inference
            # request (including prompt contents) through a remote proxy.
            return cls(host=host or None, trust_env=False, **options)
        if host or options:
            return cls(host=host or None, **options)
    return cls()


def _start_local_ollama_if_enabled(
    config: JarvisConfig,
    key: str,
    host: str,
    is_usable: Any,
    *,
    timeout: float = 12.0,
) -> InferenceEngine | None:
    """Start and wait for the configured Ollama service in local-only mode.

    The helper is called only after the configured engine failed its health
    check. It does not download models, run a shell, or start a non-loopback
    endpoint. Generic profiles leave this disabled by default.
    """
    if (
        not config.engine.auto_start_local
        or (config.intelligence.provider or "").strip().lower() != "local"
        or key != "ollama"
        or not host
    ):
        return None

    from openjarvis.engine._local_runtime import start_local_ollama

    if not start_local_ollama(host):
        return None

    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        engine: InferenceEngine | None = None
        try:
            engine = _make_engine(key, config)
            if not is_local_engine(engine, key):
                logger.warning("Refusing auto-start retry for non-local Ollama host")
                engine.close()
                return None
            healthy = bool(engine.health())
            if healthy:
                if is_usable(engine, key, healthy=healthy):
                    logger.info("Local Ollama runtime is ready")
                    return engine
                # The endpoint is up, but the requested model is not usable.
                # Never pull a model implicitly or wait pointlessly for it.
                engine.close()
                return None
        except Exception as exc:
            logger.debug("Waiting for local Ollama startup: %s", exc)
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                logger.debug("Could not close Ollama startup probe: %s", exc)
        if time.monotonic() >= deadline:
            logger.warning("Local Ollama did not become ready after startup")
            return None
        time.sleep(0.25)


def _maybe_register_mining_sidecar_engine() -> None:
    """If a mining sidecar exists with a ``vllm_endpoint``, register a derived
    vLLM engine class pointing at it.  Idempotent.  Quiet on error.

    The trigger is the *shape* of the sidecar (presence of ``vllm_endpoint``),
    not the value of its ``provider`` field — this leaves room for future
    non-engine-replacing providers (e.g., a hypothetical cpu-pearl) whose
    sidecars don't include ``vllm_endpoint``.
    """
    try:
        from openjarvis.mining import Sidecar
        from openjarvis.mining._constants import SIDECAR_PATH
    except ImportError:
        return

    if EngineRegistry.contains("vllm-pearl-mining"):
        return  # idempotent

    payload = Sidecar.read(SIDECAR_PATH)
    if payload is None:
        return

    endpoint = payload.get("vllm_endpoint")
    model = payload.get("model")
    if not endpoint or not model:
        return  # data-driven gate: no vllm_endpoint → don't register

    from openjarvis.engine._openai_compat import _OpenAICompatibleEngine

    # Strip a trailing "/v1" path segment so _default_host is the bare
    # base URL and _api_prefix="/v1" combines correctly in request paths.
    api_prefix = "/v1"
    base_url = endpoint.rstrip("/")
    if base_url.endswith(api_prefix):
        base_url = base_url[: -len(api_prefix)]

    _cls = type(
        "VllmPearlMiningEngine",
        (_OpenAICompatibleEngine,),
        {
            "engine_id": "vllm-pearl-mining",
            "_default_host": base_url,
            "_api_prefix": api_prefix,
        },
    )
    EngineRegistry.register_value("vllm-pearl-mining", _cls)


def discover_engines(config: JarvisConfig) -> List[Tuple[str, InferenceEngine]]:
    """Probe registered engines and return ``[(key, instance)]`` for healthy ones.

    Results are sorted with the config default engine first.
    """
    provider = (config.intelligence.provider or "").strip().lower()
    if provider != "cloudflare":
        _maybe_register_mining_sidecar_engine()

    # Probe engines concurrently: each health() does a blocking network
    # check with its own timeout, so a serial loop costs the SUM of all
    # probe timeouts (dead localhost ports especially). Running them in
    # threads collapses that to roughly the slowest single probe. The
    # healthy.sort() below normalizes order, so completion order is
    # irrelevant and the result is identical to the serial version (#263).
    local_only = provider == "local"
    if provider == "cloudflare":
        keys = ["cloudflare"] if EngineRegistry.contains("cloudflare") else []
    else:
        keys = [key for key in EngineRegistry.keys() if key != "cloudflare"]
    if local_only:
        # A local-only profile must not even probe cloud engines. Besides
        # preventing accidental inference fallback, this avoids outbound
        # provider discovery/health checks when local services are down.
        keys = [
            key
            for key in keys
            if not bool(getattr(EngineRegistry.get(key), "is_cloud", False))
        ]

    def _probe(key: str) -> Tuple[str, InferenceEngine] | None:
        engine: InferenceEngine | None = None
        try:
            engine = _make_engine(key, config)
            if local_only and not is_local_engine(engine, key):
                logger.debug("Skipping non-local engine %r in local-only mode", key)
                try:
                    engine.close()
                except Exception as exc:
                    logger.debug("Could not close non-local engine %r: %s", key, exc)
                engine = None
                return None
            if engine.health():
                return (key, engine)
        except Exception as exc:
            logger.debug("Engine %r failed during discovery: %s", key, exc)
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                logger.debug("Could not close unavailable engine %r: %s", key, exc)
        return None

    healthy: List[Tuple[str, InferenceEngine]] = []
    if keys:
        with ThreadPoolExecutor(max_workers=len(keys)) as pool:
            for result in pool.map(_probe, keys):
                if result is not None:
                    healthy.append(result)

    default_key = config.engine.default

    def sort_key(item: Tuple[str, Any]) -> Tuple[int, str]:
        return (0 if item[0] == default_key else 1, item[0])

    healthy.sort(key=sort_key)
    return healthy


def discover_models(
    engines: List[Tuple[str, InferenceEngine]],
) -> Dict[str, List[str]]:
    """Call ``list_models()`` on each engine and return a dict."""
    result: Dict[str, List[str]] = {}
    for key, engine in engines:
        try:
            result[key] = engine.list_models()
        except Exception as exc:
            logger.debug("Failed to list models for engine %r: %s", key, exc)
            result[key] = []
    return result


def get_engine(
    config: JarvisConfig,
    engine_key: str | None = None,
    model: str | None = None,
    fallback_models: Sequence[str] | None = None,
) -> Tuple[str, InferenceEngine] | None:
    """Get a specific engine by key, or the default with a named fallback.

    An explicit ``engine_key`` is authoritative and is never silently
    substituted. Default-engine fallback prefers the same local/cloud class
    and logs the selected replacement.

    When *model* is given, an engine is selected only if it can actually
    serve that model (``engine.can_serve(model)``). Additional *fallback_models*
    are considered in order, allowing local profiles to select a healthy
    backend capable of serving a configured local fallback. This stops the cloud
    fallback from being chosen — when the local engine is down — for a model
    whose provider client is missing, which otherwise surfaces as a confusing
    "OpenAI client not available" instead of a helpful "start your local
    engine" message (see #532). When *model* is ``None`` selection stays
    model-agnostic (unchanged behaviour). In local-only mode, HTTP engine
    endpoints must resolve to loopback; only known in-process engines may
    operate without a host. Remote OpenAI-compatible services are not probed.

    Returns ``(key, engine_instance)`` or ``None`` if no engine is available.
    """

    provider = (config.intelligence.provider or "").strip().lower()
    local_only = provider == "local"
    if engine_key is None and provider == "cloudflare":
        engine_key = "cloudflare"
    model_candidates = list(
        dict.fromkeys(
            candidate for candidate in [model, *(fallback_models or ())] if candidate
        )
    )

    def _usable(
        engine: InferenceEngine, key: str, *, healthy: bool | None = None
    ) -> bool:
        if local_only and not is_local_engine(engine, key):
            return False
        is_healthy = engine.health() if healthy is None else healthy
        if key == "cloudflare":
            return is_healthy and (not model or engine.can_serve(model))
        return is_healthy and (
            not model_candidates
            or any(engine.can_serve(candidate) for candidate in model_candidates)
        )

    def _with_local_failover(
        key: str, engine: InferenceEngine
    ) -> Tuple[str, InferenceEngine]:
        if not local_only:
            return key, engine

        recover_primary = None
        if config.engine.auto_start_local and key == "ollama":
            engine_host = str(getattr(engine, "_host", "") or "")
            if engine_host:

                def _recover_primary() -> bool:
                    recovered = _start_local_ollama_if_enabled(
                        config, key, engine_host, _usable
                    )
                    if recovered is None:
                        return False
                    try:
                        return True
                    finally:
                        recovered.close()

                recover_primary = _recover_primary

        if engine_key:
            if recover_primary is None:
                return key, engine
            from openjarvis.engine._local_failover import LocalFailoverEngine

            return key, LocalFailoverEngine(
                engine,
                [],
                fallback_model=config.intelligence.fallback_model,
                primary_key=key,
                recover_primary=recover_primary,
            )

        alternatives = []
        for other_key, other_engine in discover_engines(config):
            if other_key != key and is_local_engine(other_engine, other_key):
                alternatives.append((other_key, other_engine))
            else:
                try:
                    other_engine.close()
                except Exception as exc:
                    logger.debug(
                        "Could not close unused discovered engine %r: %s",
                        other_key,
                        exc,
                    )
        from openjarvis.engine._local_failover import LocalFailoverEngine

        return key, LocalFailoverEngine(
            engine,
            alternatives,
            fallback_model=config.intelligence.fallback_model,
            primary_key=key,
            refresh_alternatives=lambda: discover_engines(config),
            recover_primary=recover_primary,
        )

    if engine_key:
        if not EngineRegistry.contains(engine_key):
            logger.warning("Requested engine %r is not registered", engine_key)
            return None
        engine_cls = EngineRegistry.get(engine_key)
        if local_only and bool(getattr(engine_cls, "is_cloud", False)):
            logger.warning(
                "Requested engine %r is cloud-based and blocked by "
                "local-only provider configuration",
                engine_key,
            )
            return None
        engine: InferenceEngine | None = None
        engine_host = ""
        engine_healthy = False
        try:
            engine = _make_engine(engine_key, config)
            engine_host = str(getattr(engine, "_host", "") or "")
            if not local_only or is_local_engine(engine, engine_key):
                engine_healthy = bool(engine.health())
            if _usable(engine, engine_key, healthy=engine_healthy):
                return _with_local_failover(engine_key, engine)
        except Exception as exc:
            logger.debug("Engine %r health check failed: %s", engine_key, exc)
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                logger.debug(
                    "Could not close unavailable engine %r: %s",
                    engine_key,
                    exc,
                )
        if not engine_healthy:
            started = _start_local_ollama_if_enabled(
                config, engine_key, engine_host, _usable
            )
            if started is not None:
                return _with_local_failover(engine_key, started)
        logger.warning(
            "Requested engine %r is unavailable%s; no substitute was selected",
            engine_key,
            f" for model {model!r}" if model else "",
        )
        return None

    default_key = config.engine.default
    default_is_cloud: bool | None = None
    if (
        default_key
        and default_key != "cloudflare"
        and EngineRegistry.contains(default_key)
    ):
        default_cls = EngineRegistry.get(default_key)
        default_is_cloud = bool(getattr(default_cls, "is_cloud", False))
        if local_only and default_is_cloud:
            logger.debug(
                "Skipping cloud default engine %r in local-only mode", default_key
            )
        else:
            engine = None
            engine_host = ""
            engine_healthy = False
            try:
                engine = _make_engine(default_key, config)
                default_is_cloud = bool(getattr(engine, "is_cloud", default_is_cloud))
                engine_host = str(getattr(engine, "_host", "") or "")
                if not local_only or is_local_engine(engine, default_key):
                    engine_healthy = bool(engine.health())
                if _usable(engine, default_key, healthy=engine_healthy):
                    return _with_local_failover(default_key, engine)
            except Exception as exc:
                logger.debug("Engine %r health check failed: %s", default_key, exc)
            if engine is not None:
                try:
                    engine.close()
                except Exception as exc:
                    logger.debug(
                        "Could not close unavailable engine %r: %s",
                        default_key,
                        exc,
                    )
            if not engine_healthy:
                started = _start_local_ollama_if_enabled(
                    config, default_key, engine_host, _usable
                )
                if started is not None:
                    return _with_local_failover(default_key, started)

    candidates = [
        (key, engine)
        for key, engine in discover_engines(config)
        if key != default_key
        and (
            not model_candidates
            or any(engine.can_serve(candidate) for candidate in model_candidates)
        )
    ]
    if not candidates:
        return None

    chosen: Tuple[str, InferenceEngine] | None = None
    if default_is_cloud is not None:
        chosen = next(
            (
                candidate
                for candidate in candidates
                if bool(getattr(candidate[1], "is_cloud", False)) == default_is_cloud
            ),
            None,
        )
    chosen = chosen or candidates[0]
    chosen_is_cloud = bool(getattr(chosen[1], "is_cloud", False))
    boundary = (
        " across the local/cloud boundary"
        if (default_is_cloud is not None and chosen_is_cloud != default_is_cloud)
        else ""
    )
    logger.warning(
        "Default engine %r is unavailable; using %r%s",
        default_key,
        chosen[0],
        boundary,
    )
    return _with_local_failover(*chosen)


def get_configured_engine(
    config: JarvisConfig,
    engine_key: str | None = None,
    model: str | None = None,
    fallback_models: Sequence[str] | None = None,
    *,
    pin_default: bool = False,
) -> Tuple[str, InferenceEngine] | None:
    """Resolve an explicit engine or configured preference with local safety.

    An explicit *engine_key* is authoritative. Otherwise, a saved preferred
    engine is the primary for local-only profiles, with the shared resolver
    free to attach other verified local runtimes for retry. Non-local profiles
    retain the preferred-engine pinning behavior. *pin_default* preserves
    callers that historically treated ``engine.default`` as an exact choice.
    """
    provider = (config.intelligence.provider or "").strip().lower()
    local_only = provider == "local"
    preferred = config.intelligence.preferred_engine

    if engine_key is not None:
        selected_config = config
        selected_key = engine_key
    elif provider == "cloudflare":
        selected_config = config
        selected_key = "cloudflare"
    elif local_only:
        selected_config = config
        if preferred:
            selected_config = copy.deepcopy(config)
            selected_config.engine.default = preferred
        selected_key = None
    else:
        selected_config = config
        selected_key = (
            preferred or (config.engine.default if pin_default else None) or None
        )
        if selected_key == "cloudflare":
            selected_key = None

    return get_engine(
        selected_config,
        selected_key,
        model=model,
        fallback_models=fallback_models,
    )


__all__ = [
    "discover_engines",
    "discover_models",
    "engine_host_config_field",
    "get_configured_engine",
    "get_engine",
]
