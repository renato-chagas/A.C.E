"""REST endpoints for the proactive-agent approval queue."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from openjarvis.tools.approval_store import (
    DECISION_ALWAYS_DENY,
    STATUS_DENIED,
    STATUS_PENDING,
    ApprovalStore,
    PendingAction,
)
from openjarvis.tools.proactive_tools import approve_and_execute_action

try:
    from fastapi import APIRouter, HTTPException
except ImportError:
    raise ImportError("fastapi is required for approval routes")

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton that shares the same DB file as ProactiveAgent (WAL mode is safe)
_store: Optional[ApprovalStore] = None


def _get_store() -> ApprovalStore:
    global _store
    if _store is None:
        _store = ApprovalStore()
    return _store


def _serialize(action: PendingAction) -> Dict[str, Any]:
    return {
        "id": action.id,
        "action_type": action.action_type,
        "description": action.description,
        "payload": action.payload,
        "permission_key": action.permission_key,
        "tier": action.tier,
        "status": action.status,
        "created_at": action.created_at,
        "expires_at": action.expires_at,
    }


@router.get("/v1/approvals/pending")
async def list_pending_approvals() -> Dict[str, Any]:
    store = _get_store()
    store.expire_stale()
    actions = store.list_pending()
    in_progress = [*store.list_approved(), *store.list_executing()]
    return {
        "actions": [_serialize(a) for a in actions],
        "count": len(actions),
        "in_progress": [_serialize(a) for a in in_progress],
        "in_progress_count": len(in_progress),
    }


@router.post("/v1/approvals/{action_id}/approve")
def approve_action(
    action_id: str,
    remember_future: bool = False,
) -> Dict[str, Any]:
    store = _get_store()
    try:
        outcome = approve_and_execute_action(
            action_id,
            store,
            remember_future=remember_future,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if outcome["decision"] == "not_found":
        raise HTTPException(status_code=404, detail="Action not found")
    if outcome["decision"] == "conflict":
        raise HTTPException(
            status_code=409,
            detail=(f"Action cannot be approved from status {outcome['status']!r}"),
        )
    if outcome.get("execution") == "succeeded":
        logger.info("Action %s approved and executed via UI", action_id)
        return {
            "status": outcome["status"],
            "id": action_id,
            "execution": "succeeded",
            "remembered": outcome.get("remembered", False),
        }
    if outcome.get("execution") == "failed":
        logger.warning("Approved action %s could not be completed", action_id)
        message = (
            "The connector could not complete this action. It remains pending; "
            "review it before trying again."
            if outcome["status"] == STATUS_PENDING
            else "The approval expired before the connector action could run."
        )
        return {
            "status": outcome["status"],
            "id": action_id,
            "execution": "failed",
            "remembered": outcome.get("remembered", False),
            "message": message,
        }
    if outcome.get("execution") == "running":
        return {
            "status": outcome["status"],
            "id": action_id,
            "execution": "running",
            "remembered": outcome.get("remembered", False),
        }
    logger.error("Action %s approval execution has an uncertain outcome", action_id)
    return {
        "status": outcome["status"],
        "id": action_id,
        "execution": "uncertain",
        "remembered": outcome.get("remembered", False),
        "message": (
            "ACE could not confirm whether the connector action completed. "
            "Check the connected service before retrying."
        ),
    }


@router.post("/v1/approvals/{action_id}/deny")
async def deny_action(
    action_id: str,
    remember_future: bool = False,
) -> Dict[str, Any]:
    store = _get_store()
    store.expire_stale()
    action = store.get_action(action_id)
    if action is None:
        raise HTTPException(status_code=404, detail="Action not found")
    try:
        transitioned = (
            store.transition_and_remember_permission(
                action_id,
                STATUS_PENDING,
                STATUS_DENIED,
                DECISION_ALWAYS_DENY,
            )
            if remember_future
            else store.transition_status(action_id, STATUS_PENDING, STATUS_DENIED)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if action.status != STATUS_PENDING or not transitioned:
        raise HTTPException(
            status_code=409,
            detail=f"Action cannot be denied from status {action.status!r}",
        )
    logger.info("Action %s denied via UI", action_id)
    return {"status": "denied", "id": action_id, "remembered": remember_future}


__all__ = ["router"]
