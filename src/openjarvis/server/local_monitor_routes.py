"""Private loopback routes for opt-in resource monitoring."""

from __future__ import annotations

import sqlite3
from collections.abc import Awaitable, Callable
from copy import deepcopy
from datetime import datetime
from threading import BoundedSemaphore
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Path, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict
from starlette.types import Message

from openjarvis.core.paths import ConfigurationError, get_config_dir
from openjarvis.engine._pinned_inference import pinned_local_inference
from openjarvis.memory.service import _build_scanner, _build_secret_scanner
from openjarvis.proactive.desktop_monitor import DesktopMonitorConflict
from openjarvis.proactive.local_monitor import _DEFAULT_DB_NAME, LocalMonitorStore
from openjarvis.proactive.reasoning import (
    LocalIncidentReasoner,
    measured_incident_fallback,
)
from openjarvis.server.api_routes import _allows_private_memory_access

_NO_STORE = {"Cache-Control": "no-store"}
_UNAVAILABLE = "Local resource monitoring is unavailable."
_INVALID_REQUEST = "Invalid local resource monitor request."
_BODY_LIMIT = 1024
_ANALYSIS_SLOT = BoundedSemaphore(1)
_MEASUREMENTS = {
    "cpu_high": ("CPU usage", ("cpu_percent", "limit_percent")),
    "memory_low": (
        "Available memory",
        ("available_percent", "available_gb", "limit_percent"),
    ),
    "disk_low": ("Free disk space", ("free_percent", "free_gb", "limit_percent")),
}


class _PrivateMonitorRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        handler = super().get_route_handler()

        async def guarded(request: Request) -> Response:
            if not _allows_private_memory_access(request):
                raise HTTPException(
                    403, "Direct local access is required.", headers=_NO_STORE
                )
            if request.method in {"PATCH", "POST"}:
                media_type = request.headers.get("content-type", "").split(";", 1)[0]
                if media_type.strip().lower() != "application/json":
                    raise HTTPException(
                        415, "Use an application/json request.", headers=_NO_STORE
                    )
                request = _bounded_body(request)
            try:
                response = await handler(request)
            except RequestValidationError:
                raise HTTPException(422, _INVALID_REQUEST, headers=_NO_STORE) from None
            response.headers.update(_NO_STORE)
            return response

        return guarded


def _bounded_body(request: Request) -> Request:
    received = 0

    async def receive() -> Message:
        nonlocal received
        message = await request.receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > _BODY_LIMIT:
                raise HTTPException(413, _INVALID_REQUEST, headers=_NO_STORE)
        return message

    return Request(request.scope, receive)


class MonitorOptInRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool
    notifications_enabled: bool = False


class AcknowledgeIncidentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AnalyzeIncidentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    use_local_model: bool = False


class MonitorAssessment(BaseModel):
    id: int
    analysis: str
    created_at: str


class IncidentAnalysisResult(BaseModel):
    incident_id: int
    assessment: MonitorAssessment
    mode: Literal["measurements", "local_model"]


class MonitorIncident(BaseModel):
    id: int
    metric: str
    severity: str
    status: str
    summary: str
    first_seen: str
    last_seen: str
    resolved_at: str | None
    acknowledged_at: str | None
    observations: int
    episodes_last_30d: int
    assessments: list[MonitorAssessment]


class LocalMonitorStatus(BaseModel):
    enabled: bool
    running: bool
    externally_managed: bool
    notifications_enabled: bool
    history_available: bool
    incidents: list[MonitorIncident]


def _read_status(
    request: Request, *, change: MonitorOptInRequest | None = None
) -> LocalMonitorStatus:
    controller = getattr(request.app.state, "local_monitor_controller", None)
    if controller is None:
        raise HTTPException(503, _UNAVAILABLE, headers=_NO_STORE)
    try:
        if change is not None:
            controller.set_enabled(
                change.enabled, notifications_enabled=change.notifications_enabled
            )
        enabled, running, external, notifications_enabled = controller.status()
    except DesktopMonitorConflict:
        raise HTTPException(
            409,
            "The local monitor is busy or must be stopped before changing alerts.",
            headers=_NO_STORE,
        ) from None
    except (OSError, RuntimeError, sqlite3.Error, ValueError):
        raise HTTPException(503, _UNAVAILABLE, headers=_NO_STORE) from None

    history_available = True
    try:
        incidents = [
            MonitorIncident.model_validate(item)
            for item in LocalMonitorStore.read_recent(limit=30)
        ]
    except (ConfigurationError, OSError, sqlite3.Error, ValueError):
        # The stop control must remain available when history cannot be read.
        history_available = False
        incidents = []
    return LocalMonitorStatus(
        enabled=enabled,
        running=running,
        externally_managed=external,
        notifications_enabled=notifications_enabled,
        history_available=history_available,
        incidents=incidents,
    )


router = APIRouter(
    prefix="/v1/local-monitor",
    tags=["local resource monitor"],
    route_class=_PrivateMonitorRoute,
)


@router.get("")
def get_local_monitor(request: Request) -> LocalMonitorStatus:
    """Read the monitor state and recent local incidents."""
    return _read_status(request)


@router.patch("")
def update_local_monitor(
    body: MonitorOptInRequest, request: Request
) -> LocalMonitorStatus:
    """Apply the requested monitor opt-in and return its current state."""
    if body.notifications_enabled and not body.enabled:
        raise HTTPException(422, _INVALID_REQUEST, headers=_NO_STORE)
    return _read_status(request, change=body)


@router.post("/incidents/{incident_id}/acknowledge")
def acknowledge_local_monitor_incident(
    incident_id: Annotated[int, Path(gt=0, le=2**63 - 1)],
    body: AcknowledgeIncidentRequest,
    request: Request,
) -> LocalMonitorStatus:
    """Record that the user has reviewed a saved local incident."""
    try:
        path = get_config_dir() / _DEFAULT_DB_NAME
        acknowledged = path.is_file() and LocalMonitorStore(path).acknowledge(
            incident_id
        )
    except (ConfigurationError, OSError, sqlite3.Error, ValueError):
        raise HTTPException(503, _UNAVAILABLE, headers=_NO_STORE) from None
    if not acknowledged:
        raise HTTPException(404, "Resource incident not found.", headers=_NO_STORE)
    return _read_status(request)


def _aggregate_measurements(snapshot: dict[str, Any]) -> dict[str, Any]:
    name, allowed = _MEASUREMENTS[snapshot["metric"]]
    details = snapshot.get("details")
    details = details if isinstance(details, dict) else {}
    numbers = {}
    for key in allowed:
        value = details.get(key)
        limit = 100 if key.endswith("percent") else 1_000_000
        if type(value) in (int, float) and 0 <= value <= limit:
            numbers[key] = value
    event = {
        "id": snapshot["id"],
        "metric": snapshot["metric"],
        "summary": name,
        "severity": snapshot.get("severity")
        if snapshot.get("severity")
        in (
            "warning",
            "critical",
        )
        else None,
        "status": snapshot.get("status")
        if snapshot.get("status")
        in (
            "active",
            "resolved",
        )
        else None,
        "details": numbers,
    }
    for key in ("first_seen", "last_seen"):
        try:
            event[key] = datetime.fromisoformat(snapshot[key]).isoformat()
        except (KeyError, TypeError, ValueError):
            pass
    for key in ("observations", "episodes_last_30d"):
        value = snapshot.get(key)
        if type(value) is int and 0 <= value <= 1_000_000:
            event[key] = value
    return event


def _local_model_assessment(request: Request, event: dict[str, Any]) -> str:
    try:
        with pinned_local_inference(
            request.app.state, capability="completion"
        ) as runtime:
            if runtime is None:
                return ""
            config = deepcopy(request.app.state.config)
            config.agent.context_from_memory = False
            engine, model = runtime
            response = LocalIncidentReasoner(engine, model, config).analyze(event)
        if not response:
            return ""
        scanners = (_build_secret_scanner(), _build_scanner())
        if all(
            scanner is not None and scanner.scan(response).is_clean is True
            for scanner in scanners
        ):
            return f"Local model suggestion:\n{response}"
    except Exception:
        # A failing model or scanner leaves the stored measurement assessment usable.
        return ""
    return ""


def _record_panel_assessment(
    request: Request, incident_id: int, *, use_local_model: bool
) -> IncidentAnalysisResult:
    try:
        snapshot = LocalMonitorStore.read_assessment_incident(incident_id)
        if snapshot is None:
            raise HTTPException(404, "Resource incident not found.", headers=_NO_STORE)
        if snapshot.get("metric") not in _MEASUREMENTS:
            raise HTTPException(
                422, "This incident has no supported measurements.", headers=_NO_STORE
            )

        event = _aggregate_measurements(snapshot)
        baseline = measured_incident_fallback(event, pending_for_local_model=False)
        store = LocalMonitorStore()
        measured = store.save_panel_assessment(snapshot, baseline)
        if measured is None:
            raise HTTPException(
                409, "The incident changed. Refresh and try again.", headers=_NO_STORE
            )
        if use_local_model and (analysis := _local_model_assessment(request, event)):
            saved = store.save_panel_assessment(snapshot, analysis)
            if saved is None:
                raise HTTPException(
                    409,
                    "The incident changed. Refresh and try again.",
                    headers=_NO_STORE,
                )
            return IncidentAnalysisResult(
                incident_id=incident_id,
                assessment=MonitorAssessment.model_validate(saved),
                mode="local_model",
            )
        return IncidentAnalysisResult(
            incident_id=incident_id,
            assessment=MonitorAssessment.model_validate(measured),
            mode="measurements",
        )
    except (ConfigurationError, OSError, sqlite3.Error, ValueError):
        raise HTTPException(503, _UNAVAILABLE, headers=_NO_STORE) from None


@router.post("/incidents/{incident_id}/analyze")
def analyze_local_monitor_incident(
    incident_id: Annotated[int, Path(gt=0, le=2**63 - 1)],
    body: AnalyzeIncidentRequest,
    request: Request,
) -> IncidentAnalysisResult:
    """Assess this saved snapshot without contacting a cloud model or using tools."""
    if not body.use_local_model:
        return _record_panel_assessment(request, incident_id, use_local_model=False)
    if not _ANALYSIS_SLOT.acquire(blocking=False):
        raise HTTPException(
            429, "A local model assessment is already running.", headers=_NO_STORE
        )
    try:
        return _record_panel_assessment(request, incident_id, use_local_model=True)
    finally:
        _ANALYSIS_SLOT.release()
