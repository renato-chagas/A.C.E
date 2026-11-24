"""Local-only inference failover across already-discovered local engines."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, Dict, List

from openjarvis.core.types import Message
from openjarvis.engine._base import InferenceEngine
from openjarvis.engine._locality import is_local_engine
from openjarvis.engine._stubs import StreamChunk, closing_stream

logger = logging.getLogger(__name__)


class LocalFailoverEngine(InferenceEngine):
    """Retry failed inference on another healthy local engine/model.

    Alternatives are discovered at startup and refreshed after failed local
    inference. Each alternative must confirm it can serve the requested model;
    only an explicitly configured fallback model may be tried in its place.
    This class never chooses an arbitrary installed model or discovers cloud
    engines.
    """

    engine_id = "local-failover"

    def __init__(
        self,
        primary: InferenceEngine,
        alternatives: Sequence[tuple[str, InferenceEngine]],
        fallback_model: str = "",
        *,
        primary_key: str = "",
        refresh_alternatives: (
            Callable[[], Sequence[tuple[str, InferenceEngine]]] | None
        ) = None,
        recover_primary: Callable[[], bool] | None = None,
        recovery_cooldown_seconds: float = 5.0,
    ) -> None:
        if not is_local_engine(primary, primary_key):
            raise ValueError("LocalFailoverEngine requires a verified local primary")
        self._primary = primary
        self._alternatives = []
        for key, engine in alternatives:
            if is_local_engine(engine, key):
                self._alternatives.append((key, engine))
            else:
                _close_engine(engine)
        self._fallback_model = fallback_model
        self._primary_key = primary_key
        self._refresh_alternatives = refresh_alternatives
        self._recover_primary = recover_primary
        self._recovery_cooldown_seconds = max(0.0, recovery_cooldown_seconds)
        self._recovery_retry_after = 0.0
        self._refresh_lock = threading.Lock()
        self.engine_id = getattr(primary, "engine_id", self.engine_id)

    def __getattr__(self, name: str) -> Any:
        # Keep backend-specific controls (for example, Ollama runtime options)
        # available to callers while inference methods use the failover path.
        return getattr(self._primary, name)

    @property
    def is_cloud(self) -> bool:
        """A local failover wrapper never routes inference to cloud engines."""
        return False

    @property
    def is_local(self) -> bool:
        """Preserve locality for explicitly local custom in-process engines."""
        return is_local_engine(self._primary, self._primary_key)

    @property
    def _host(self) -> str:
        """Expose the primary endpoint so outer local-only checks can verify it."""
        return getattr(self._primary, "_host", "") or ""

    def is_local_for_model(self, model: str) -> bool:
        """Failover routes stay within the verified local engine set."""
        del model  # Every eligible route is local; model mapping is backend-owned.
        return self.is_local

    def _try_recover_primary(self) -> bool:
        """Recover an opted-in primary only when its health check is down."""
        if self._recover_primary is None:
            return False
        now = time.monotonic()
        if now < self._recovery_retry_after:
            return False
        try:
            if self._primary.health():
                return False
        except Exception as exc:
            logger.debug("Primary local health check failed: %s", exc)

        try:
            recovered = bool(self._recover_primary())
        except Exception as exc:
            self._recovery_retry_after = (
                time.monotonic() + self._recovery_cooldown_seconds
            )
            logger.warning("Could not recover local inference runtime: %s", exc)
            return False
        if recovered:
            self._recovery_retry_after = 0.0
            logger.info("Recovered primary local inference runtime")
        else:
            self._recovery_retry_after = (
                time.monotonic() + self._recovery_cooldown_seconds
            )
        return recovered

    async def _try_recover_primary_async(self) -> bool:
        """Run synchronous health/startup checks outside the stream event loop."""
        return await asyncio.to_thread(self._try_recover_primary)

    def _routes(self, requested_model: str) -> list[tuple[str, InferenceEngine, str]]:
        routes: list[tuple[str, InferenceEngine, str]] = []
        self._refresh_local_alternatives()
        if self._fallback_model and self._fallback_model != requested_model:
            try:
                available = list(self._primary.list_models())
                has_fallback = not available or self._fallback_model in available
                if (
                    has_fallback
                    and self._primary.health()
                    and self._primary.can_serve(self._fallback_model)
                ):
                    routes.append(("primary", self._primary, self._fallback_model))
            except Exception as exc:
                logger.debug("Primary local fallback model is unavailable: %s", exc)

        for key, engine in self._alternatives:
            if not is_local_engine(engine, key) or not _engine_is_healthy(engine):
                continue
            try:
                available = list(engine.list_models())
            except Exception as exc:
                logger.debug(
                    "Could not list models for local fallback %s: %s", key, exc
                )
                available = []

            choices = [
                candidate
                for candidate in dict.fromkeys([requested_model, self._fallback_model])
                if candidate
                and (
                    candidate == requested_model
                    or not available
                    or candidate in available
                )
            ]
            for candidate in choices:
                try:
                    if engine.can_serve(candidate):
                        routes.append((key, engine, candidate))
                except Exception as exc:
                    logger.debug(
                        "Local fallback %s cannot serve %s: %s", key, candidate, exc
                    )
        return routes

    def _refresh_local_alternatives(self) -> None:
        """Discover local engines again after a failure, without probing cloud."""
        if self._refresh_alternatives is None:
            return
        with self._refresh_lock:
            try:
                discovered = self._refresh_alternatives()
            except Exception as exc:
                logger.debug("Could not refresh local inference engines: %s", exc)
                return

            # Repair any duplicates left by an older wrapper state before
            # incorporating newly discovered instances. Keep the healthy
            # instance; otherwise prefer the latest healthy discovery.
            by_key: dict[str, InferenceEngine] = {}
            for key, engine in self._alternatives:
                current = by_key.get(key)
                if current is None:
                    by_key[key] = engine
                elif _engine_is_healthy(current):
                    _close_engine(engine, skip=current)
                elif _engine_is_healthy(engine):
                    _close_engine(current, skip=self._primary)
                    by_key[key] = engine
                else:
                    _close_engine(engine, skip=current)
            self._alternatives = list(by_key.items())

            for key, engine in discovered:
                if (
                    key == self._primary_key
                    or engine is self._primary
                    or not is_local_engine(engine, key)
                ):
                    _close_engine(engine, skip=self._primary)
                    continue
                existing = by_key.get(key)
                if existing is not None and _engine_is_healthy(existing):
                    _close_engine(engine, skip=existing)
                    continue
                if existing is not None:
                    _close_engine(existing, skip=engine)
                by_key[key] = engine
            self._alternatives = list(by_key.items())

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            return self._primary.generate(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as primary_error:
            last_error = primary_error
            if self._try_recover_primary():
                try:
                    return self._primary.generate(
                        messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                except Exception as exc:
                    last_error = exc
                    logger.debug("Recovered local primary retry failed: %s", exc)
            for key, engine, target in self._routes(model):
                logger.warning(
                    "Local inference on %s failed; retrying with %s / %s",
                    getattr(self._primary, "engine_id", "primary"),
                    key,
                    target,
                )
                try:
                    return engine.generate(
                        messages,
                        model=target,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                except Exception as exc:
                    last_error = exc
                    logger.debug("Local fallback %s failed: %s", key, exc)
            raise last_error

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        emitted = False
        try:
            async for token in self._primary.stream(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            ):
                emitted = emitted or bool(token)
                yield token
            return
        except Exception as primary_error:
            if emitted:
                raise
            last_error = primary_error

        if await self._try_recover_primary_async():
            retry_emitted = False
            try:
                async for token in self._primary.stream(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ):
                    retry_emitted = retry_emitted or bool(token)
                    yield token
                return
            except Exception as exc:
                if retry_emitted:
                    raise
                last_error = exc
                logger.debug("Recovered local stream retry failed: %s", exc)

        for key, engine, target in self._routes(model):
            logger.warning("Local stream failed; retrying with %s / %s", key, target)
            fallback_emitted = False
            try:
                async for token in engine.stream(
                    messages,
                    model=target,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ):
                    fallback_emitted = fallback_emitted or bool(token)
                    yield token
                return
            except Exception as exc:
                if fallback_emitted:
                    raise
                last_error = exc
                logger.debug("Local stream fallback %s failed: %s", key, exc)
        raise last_error

    async def stream_full(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        emitted = False
        try:
            async with closing_stream(
                self._primary.stream_full(messages, model=model, **kwargs)
            ) as chunks:
                async for chunk in chunks:
                    emitted = True
                    yield chunk
            return
        except Exception as primary_error:
            if emitted:
                raise
            last_error = primary_error

        if await self._try_recover_primary_async():
            retry_emitted = False
            try:
                async with closing_stream(
                    self._primary.stream_full(messages, model=model, **kwargs)
                ) as chunks:
                    async for chunk in chunks:
                        retry_emitted = True
                        yield chunk
                return
            except Exception as exc:
                if retry_emitted:
                    raise
                last_error = exc
                logger.debug("Recovered local full-stream retry failed: %s", exc)

        for key, engine, target in self._routes(model):
            logger.warning("Local stream failed; retrying with %s / %s", key, target)
            fallback_emitted = False
            try:
                async with closing_stream(
                    engine.stream_full(messages, model=target, **kwargs)
                ) as chunks:
                    async for chunk in chunks:
                        fallback_emitted = True
                        yield chunk
                return
            except Exception as exc:
                if fallback_emitted:
                    raise
                last_error = exc
                logger.debug("Local stream fallback %s failed: %s", key, exc)
        raise last_error

    def list_models(self) -> List[str]:
        models: list[str] = []
        engines = [
            (self._primary_key or "primary", self._primary),
            *self._alternatives,
        ]
        for key, engine in engines:
            if not is_local_engine(engine, key):
                continue
            try:
                models.extend(engine.list_models())
            except Exception as exc:
                logger.debug("Could not list models from local engine: %s", exc)
        return list(dict.fromkeys(models))

    def can_serve(self, model: str) -> bool:
        engines = [
            (self._primary_key or "primary", self._primary),
            *self._alternatives,
        ]
        for key, engine in engines:
            if not is_local_engine(engine, key):
                continue
            try:
                if engine.can_serve(model):
                    return True
            except Exception as exc:
                logger.debug("Local engine cannot serve %s: %s", model, exc)
        return False

    def health(self) -> bool:
        engines = [
            (self._primary_key or "primary", self._primary),
            *self._alternatives,
        ]
        for key, engine in engines:
            if not is_local_engine(engine, key):
                continue
            try:
                if engine.health():
                    return True
            except Exception as exc:
                logger.debug("Local engine health check failed: %s", exc)
        return False

    def prepare(self, model: str) -> None:
        self._primary.prepare(model)

    def close(self) -> None:
        for engine in [self._primary, *(item for _key, item in self._alternatives)]:
            _close_engine(engine)


def _engine_is_healthy(engine: InferenceEngine) -> bool:
    try:
        return bool(engine.health())
    except Exception:
        return False


def _close_engine(
    engine: InferenceEngine, *, skip: InferenceEngine | None = None
) -> None:
    if engine is skip:
        return
    try:
        engine.close()
    except Exception as exc:
        logger.debug("Could not close discovered local engine: %s", exc)


__all__ = ["LocalFailoverEngine"]
