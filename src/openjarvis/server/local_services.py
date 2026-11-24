"""Loopback API for ACE local storage and explicitly selected cloud chat."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from openjarvis.core.config import JarvisConfig, ServerConfig
from openjarvis.proactive.desktop_monitor import DesktopMonitorController
from openjarvis.server.api_routes import _allows_private_memory_access, memory_router
from openjarvis.server.auth_middleware import AuthMiddleware
from openjarvis.server.code_improvement_routes import router as code_improvement_router
from openjarvis.server.local_monitor_routes import router as local_monitor_router
from openjarvis.server.models import ChatCompletionRequest, ChatMessage
from openjarvis.server.screen_routes import router as screen_router
from openjarvis.server.upload_router import router as upload_router
from openjarvis.server.web_library_routes import router as web_library_router
from openjarvis.server.working_chat_routes import router as working_chat_router

if TYPE_CHECKING:
    from openjarvis.engine import InferenceEngine

# Admit individual storage endpoints so newly added routes remain opt-in here.
_MEMORY_METHODS = {
    "/v1/memory/config": {"GET"},
    "/v1/memory/stats": {"GET"},
    "/v1/memory/search": {"POST"},
    "/v1/memory/store": {"POST"},
    "/v1/memory/facts": {"GET"},
    "/v1/memory/facts/revise": {"POST"},
    "/v1/memory/episodes/{episode_id}": {"GET"},
    "/v1/memory/graph": {"GET"},
    "/v1/memory/graph/events": {"GET"},
    "/v1/memory/graph/entities": {"POST"},
    "/v1/memory/graph/entities/{entity_id}": {"DELETE"},
    "/v1/memory/graph/relations": {"POST"},
    "/v1/memory/graph/relations/{relation_id}": {"PUT", "DELETE"},
}
_UPLOAD_METHODS = {
    "/v1/connectors/upload/ingest": {"POST"},
    "/v1/connectors/upload/ingest/files": {"POST"},
}
_LOCAL_ORIGINS = tuple(ServerConfig().cors_origins)
_INFERENCE_UNAVAILABLE = "Inference is unavailable in local-services-only mode."


def create_local_services_app(
    config: JarvisConfig,
    *,
    api_key: str = "",
    cloudflare_engine: InferenceEngine | None = None,
    cloudflare_model: str | None = None,
) -> FastAPI:
    """Expose local services, with text chat only when Cloudflare is selected."""
    model = ""
    if cloudflare_engine is not None:
        from openjarvis.engine.cloudflare import (
            CLOUDFLARE_MODEL_ID,
            is_cloudflare_model_id,
        )

        model = CLOUDFLARE_MODEL_ID if cloudflare_model is None else cloudflare_model
        if not is_cloudflare_model_id(model):
            raise ValueError("Invalid Cloudflare Workers AI model ID.")
        config = deepcopy(config)
        config.intelligence.provider = "cloudflare"
        config.agent.context_from_memory = False

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            backend = getattr(app.state, "memory_backend", None)
            if cloudflare_engine is not None:
                cloudflare_engine.close()
            if backend is not None:
                backend.close()

    app = FastAPI(
        title="OpenJarvis Local Services",
        lifespan=lifespan,
        openapi_url=None,
    )
    app.state.config = config
    if cloudflare_engine is not None:
        app.state.engine = cloudflare_engine
        app.state.engine_name = "cloudflare"
        app.state.model = model
    app.state.local_monitor_controller = DesktopMonitorController(config)

    local_routes = APIRouter()
    for router, methods_by_path in (
        (memory_router, _MEMORY_METHODS),
        (upload_router, _UPLOAD_METHODS),
    ):
        local_routes.routes.extend(
            route
            for route in router.routes
            if isinstance(route, APIRoute)
            and route.methods <= methods_by_path.get(route.path, set())
        )
    app.include_router(local_routes)
    app.include_router(working_chat_router)
    app.include_router(code_improvement_router)
    app.include_router(local_monitor_router)
    app.include_router(screen_router)
    app.include_router(web_library_router)

    if api_key:
        app.add_middleware(AuthMiddleware, api_key=api_key)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_LOCAL_ORIGINS,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def local_client_only(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        origin = request.headers.get("origin")
        if not _allows_private_memory_access(request) or (
            origin is not None and origin not in _LOCAL_ORIGINS
        ):
            return JSONResponse(
                {"detail": "Direct local access is required."},
                status_code=403,
                headers={"Cache-Control": "no-store"},
            )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    async def health(response: Response) -> dict[str, str | bool]:
        if cloudflare_engine is not None:
            response.headers["X-ACE-Inference-Source"] = "cloudflare"
            return {"status": "ok"}
        return {
            "status": "ok",
            "mode": "local_services_only",
            "inference_available": False,
        }

    @app.get("/v1/info")
    async def info() -> dict[str, str | bool | None]:
        return {
            "model": model,
            "agent": None,
            "engine": "cloudflare" if model else "",
            "inference_available": bool(model),
        }

    @app.get("/v1/models")
    async def models() -> dict[str, str | list]:
        return {
            "object": "list",
            "data": [{"id": model, "object": "model", "owned_by": "cloudflare"}]
            if model
            else [],
        }

    @app.get("/v1/connectors")
    async def connectors() -> dict[str, list]:
        return {"connectors": []}

    @app.post("/v1/working-chats/turn")
    @app.post("/v1/speech/transcribe")
    @app.post("/v1/speech/synthesize")
    @app.post("/api/research")
    async def inference_unavailable() -> None:
        raise HTTPException(status_code=503, detail=_INFERENCE_UNAVAILABLE)

    if cloudflare_engine is None:
        app.add_api_route(
            "/v1/chat/completions", inference_unavailable, methods=["POST"]
        )
    else:
        _mount_cloudflare_chat(app, model)

    @app.post("/v1/memory/index")
    async def path_indexing_unavailable() -> None:
        raise HTTPException(
            status_code=503,
            detail="Upload a local document to index it in local-services-only mode.",
        )

    return app


def _mount_cloudflare_chat(app: FastAPI, model: str) -> None:
    from openjarvis.engine.cloudflare import CloudflareInferenceError
    from openjarvis.server.routes import chat_completions

    @app.post("/v1/chat/completions")
    async def cloudflare_chat(body: ChatCompletionRequest, request: Request):
        if body.model != model:
            raise HTTPException(400, "Use o modelo Cloudflare selecionado no A.C.E.")
        if body.tools:
            raise HTTPException(
                400, "O modo Cloudflare do ACE oferece conversa em texto."
            )
        # A caller-supplied system message prevents the shared handler from
        # loading the local SOUL, MEMORY or USER files for an online request.
        if not any(message.role == "system" for message in body.messages):
            body.messages.insert(
                0,
                ChatMessage(
                    role="system",
                    content=(
                        "Você é o A.C.E., um assistente pessoal. "
                        "Responda no idioma do usuário."
                    ),
                ),
            )
        try:
            return await chat_completions(body, request)
        except CloudflareInferenceError as exc:
            raise HTTPException(exc.status_code, str(exc)) from None
