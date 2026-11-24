"""Regression tests for proactive pending-action execution."""

from __future__ import annotations

import json

import pytest

from openjarvis.tools.approval_store import (
    DECISION_ALWAYS_APPROVE,
    DECISION_ALWAYS_DENY,
    STATUS_APPROVED,
    STATUS_DENIED,
    STATUS_EXECUTED,
    STATUS_EXECUTING,
    STATUS_EXPIRED,
    STATUS_PENDING,
    TIER_HIGH,
    TIER_LOW,
    TIER_MEDIUM,
    ApprovalStore,
)
from openjarvis.tools.proactive_tools import (
    ExecutePendingActionsTool,
    QueueActionTool,
    approve_and_execute_action,
    parse_approval_response,
)


def _approved_sms_draft(store: ApprovalStore, payload: dict[str, str]):
    action = store.queue_action(
        action_type="sms_draft_reply",
        description="Reply to the message",
        payload=payload,
        permission_key="sms_draft_reply:contact:+15551234567",
        tier=TIER_LOW,
    )
    store.update_status(action.id, STATUS_APPROVED)
    return action


def test_atomic_remembered_approval_uses_queued_exact_pattern(tmp_path) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))
    key = "email_archive:domain:example.test"
    action = store.queue_action(
        action_type="email_archive",
        description="Archive matched email",
        payload={"message_id": "mail-1"},
        permission_key=key,
        tier=TIER_MEDIUM,
    )

    assert store.transition_and_remember_permission(
        action.id,
        STATUS_PENDING,
        STATUS_APPROVED,
        DECISION_ALWAYS_APPROVE,
    )

    assert store.get_action(action.id).status == STATUS_APPROVED
    rule = store.get_permission(key)
    assert rule is not None
    assert rule.decision == DECISION_ALWAYS_APPROVE
    assert rule.times_approved == 1
    assert rule.times_denied == 0
    store.close()


def test_atomic_remembered_approval_rejects_high_tier(tmp_path) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))
    key = "sms_send:contact:+15551234567"
    action = store.queue_action(
        action_type="sms_send",
        description="Send a message",
        payload={"to": "+15551234567", "body": "Hello"},
        permission_key=key,
        tier=TIER_HIGH,
    )

    with pytest.raises(ValueError, match="High-tier"):
        store.transition_and_remember_permission(
            action.id,
            STATUS_PENDING,
            STATUS_APPROVED,
            DECISION_ALWAYS_APPROVE,
        )

    assert store.get_action(action.id).status == STATUS_PENDING
    assert store.get_permission(key) is None
    store.close()


def test_atomic_remembered_denial_allows_high_tier(tmp_path) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))
    key = "sms_send:contact:+15551234567"
    action = store.queue_action(
        action_type="sms_send",
        description="Send a message",
        payload={"to": "+15551234567", "body": "Hello"},
        permission_key=key,
        tier=TIER_HIGH,
    )

    assert store.transition_and_remember_permission(
        action.id,
        STATUS_PENDING,
        STATUS_DENIED,
        DECISION_ALWAYS_DENY,
    )

    assert store.get_action(action.id).status == STATUS_DENIED
    rule = store.get_permission(key)
    assert rule is not None
    assert rule.decision == DECISION_ALWAYS_DENY
    assert rule.times_denied == 1
    store.close()


def test_sms_draft_uses_documented_body_and_marks_success(tmp_path) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))
    action = _approved_sms_draft(
        store,
        {
            "contact": "+15551234567",
            "body": "Sounds good, see you tomorrow.",
        },
    )

    result = ExecutePendingActionsTool(store=store).execute()
    entry = json.loads(result.content)[0]

    assert entry["success"] is True
    assert entry["message"] == (
        "Draft for +15551234567: Sounds good, see you tomorrow."
    )
    persisted = store.get_action(action.id)
    assert persisted is not None
    assert persisted.status == STATUS_EXECUTED
    assert persisted.payload["body"] == "Sounds good, see you tomorrow."
    assert result.metadata == {"attempted": 1, "executed": 1, "failed": 0}
    store.close()


def test_action_outcome_memory_excludes_connector_payload_and_response(
    monkeypatch,
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from openjarvis.memory import service
    from openjarvis.memory.episodes import EpisodeStore

    class CleanScanner:
        def scan(self, _text):
            return SimpleNamespace(is_clean=True)

    episodes_path = tmp_path / "episodes.db"
    config = SimpleNamespace(
        proactive=SimpleNamespace(proactive_action_memory_capture_enabled=True),
        memory=SimpleNamespace(
            episodes_enabled=True,
            episodes_path=str(episodes_path),
            episodes_max_count=100,
            episodes_retention_days=365,
            episodes_embedding_model="nomic-embed-text",
        ),
    )
    monkeypatch.setattr("openjarvis.core.config.load_config", lambda: config)
    monkeypatch.setattr(service, "_build_scanner", lambda: CleanScanner())
    monkeypatch.setattr(service, "_build_secret_scanner", lambda: CleanScanner())

    store = ApprovalStore(str(tmp_path / "approvals.db"))
    action = _approved_sms_draft(
        store,
        {
            "contact": "+15551234567",
            "body": "Private message text must not enter memory.",
        },
    )
    result = ExecutePendingActionsTool(
        store=store,
        executor_fn=lambda _action: (True, "Private connector response."),
    ).execute()

    episodes = EpisodeStore(episodes_path, semantic_enabled=False)
    try:
        saved = episodes.list(limit=10)
    finally:
        episodes.close()
    assert result.success
    assert len(saved) == 1
    assert saved[0].source == "proactive_action"
    combined = saved[0].user_text + saved[0].assistant_text
    assert "sms_draft_reply" in combined
    assert "successfully" in combined
    assert action.id not in combined
    assert "+15551234567" not in combined
    assert "Private message text" not in combined
    assert "Private connector response" not in combined
    store.close()


def test_invalid_sms_draft_does_not_report_or_mark_success(tmp_path) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))
    action = _approved_sms_draft(
        store,
        {"contact": "+15551234567", "body": " \n"},
    )

    result = ExecutePendingActionsTool(store=store).execute()
    entry = json.loads(result.content)[0]

    assert entry["success"] is False
    assert entry["message"] == "Missing body in payload"
    assert result.success is False
    assert result.metadata == {"attempted": 1, "executed": 0, "failed": 1}
    persisted = store.get_action(action.id)
    assert persisted is not None
    assert persisted.status == STATUS_PENDING
    store.close()


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        ({"contact": None, "body": "hello"}, "Missing contact in payload"),
        ({"contact": 15551234567, "body": "hello"}, "Missing contact in payload"),
        ({"contact": "+15551234567", "body": None}, "Missing body in payload"),
        ({"contact": "+15551234567", "body": ["hello"]}, "Missing body in payload"),
    ],
)
def test_non_string_sms_fields_are_not_executed(
    tmp_path, payload, expected_message
) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))
    action = _approved_sms_draft(store, payload)

    result = ExecutePendingActionsTool(store=store).execute()
    entry = json.loads(result.content)[0]

    assert result.success is False
    assert entry["success"] is False
    assert entry["message"] == expected_message
    assert result.metadata == {"attempted": 1, "executed": 0, "failed": 1}
    persisted = store.get_action(action.id)
    assert persisted is not None
    assert persisted.status == STATUS_PENDING
    store.close()


def test_approval_status_transition_is_atomic(tmp_path) -> None:
    db_path = str(tmp_path / "approvals.db")
    store = ApprovalStore(db_path)
    peer_store = ApprovalStore(db_path)
    action = _approved_sms_draft(
        store,
        {"contact": "+15551234567", "body": "Hello"},
    )

    assert store.transition_status(action.id, STATUS_APPROVED, STATUS_EXECUTING)
    assert not peer_store.transition_status(
        action.id, STATUS_APPROVED, STATUS_EXECUTING
    )
    current = peer_store.get_action(action.id)
    assert current is not None
    assert current.status == STATUS_EXECUTING
    peer_store.close()
    store.close()


def test_explicit_approval_executes_one_action_and_records_completion(tmp_path) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))
    action = store.queue_action(
        action_type="test_action",
        description="Run the approved test action",
        payload={"target": "one"},
        permission_key="test_action:one",
        tier=TIER_LOW,
    )
    executed = []

    outcome = approve_and_execute_action(
        action.id,
        store,
        executor_fn=lambda pending: (executed.append(pending.id) is None, "done"),
    )

    assert outcome == {
        "decision": "approved",
        "status": STATUS_EXECUTED,
        "execution": "succeeded",
    }
    assert executed == [action.id]
    assert store.get_action(action.id).status == STATUS_EXECUTED
    store.close()


def test_failed_approved_action_returns_to_pending_without_leaking_error(
    tmp_path,
) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))
    action = store.queue_action(
        action_type="test_action",
        description="Run the approved test action",
        payload={},
        permission_key="test_action:one",
        tier=TIER_HIGH,
    )

    outcome = approve_and_execute_action(
        action.id,
        store,
        executor_fn=lambda _pending: (False, "private connector response"),
    )

    assert outcome == {
        "decision": "approved",
        "status": STATUS_PENDING,
        "execution": "failed",
    }
    assert "private connector response" not in str(outcome)
    assert store.get_action(action.id).status == STATUS_PENDING
    store.close()


def test_denied_or_expired_action_cannot_be_approved_for_execution(tmp_path) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))
    denied = store.queue_action(
        action_type="test_action",
        description="Denied action",
        payload={},
        permission_key="test_action:denied",
        tier=TIER_HIGH,
    )
    store.update_status(denied.id, STATUS_DENIED)
    expired = store.queue_action(
        action_type="test_action",
        description="Expired action",
        payload={},
        permission_key="test_action:expired",
        tier=TIER_HIGH,
    )
    store._conn.execute(
        "UPDATE pending_actions SET expires_at = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", expired.id),
    )
    store._conn.commit()
    calls = []

    def executor(action):
        calls.append(action.id)
        return True, "done"

    denied_outcome = approve_and_execute_action(denied.id, store, executor_fn=executor)
    expired_outcome = approve_and_execute_action(
        expired.id, store, executor_fn=executor
    )

    assert denied_outcome == {"decision": "conflict", "status": STATUS_DENIED}
    assert expired_outcome == {"decision": "conflict", "status": STATUS_EXPIRED}
    assert calls == []
    assert store.list_approved() == []
    store.close()


def test_expired_approved_action_is_not_returned_for_execution(tmp_path) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))
    action = _approved_sms_draft(
        store,
        {"contact": "+15551234567", "body": "Hello"},
    )
    store._conn.execute(
        "UPDATE pending_actions SET expires_at = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", action.id),
    )
    store._conn.commit()

    assert store.list_approved() == []
    assert store.get_action(action.id).status == STATUS_EXPIRED
    assert not store.transition_status(action.id, STATUS_APPROVED, STATUS_EXECUTING)
    store.close()


def test_channel_approval_parser_ignores_expired_actions(tmp_path) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))
    action = store.queue_action(
        action_type="test_action",
        description="Expired action",
        payload={},
        permission_key="test_action:expired",
        tier=TIER_LOW,
    )
    store._conn.execute(
        "UPDATE pending_actions SET expires_at = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", action.id),
    )
    store._conn.commit()

    outcome = parse_approval_response(f"{action.id} yes", store)
    assert outcome == [
        {
            "id": action.id,
            "decision": "conflict",
            "approved": True,
            "remembered": False,
            "status": STATUS_EXPIRED,
            "execution": "not_run",
        }
    ]
    assert store.get_action(action.id).status == STATUS_EXPIRED
    store.close()


def test_channel_approval_parser_does_not_overwrite_decided_actions(tmp_path) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))
    action = store.queue_action(
        action_type="test_action",
        description="Already denied action",
        payload={},
        permission_key="test_action:denied",
        tier=TIER_LOW,
    )
    store.update_status(action.id, STATUS_DENIED)

    outcome = parse_approval_response(f"always yes {action.id}", store)
    assert outcome == [
        {
            "id": action.id,
            "decision": "conflict",
            "approved": True,
            "remembered": False,
            "status": STATUS_DENIED,
            "execution": "not_run",
        }
    ]
    assert store.get_action(action.id).status == STATUS_DENIED
    assert store.get_permission(action.permission_key) is None
    store.close()


def test_channel_approval_parser_executes_only_explicitly_approved_action(
    tmp_path,
) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))
    approved = store.queue_action(
        action_type="test_action",
        description="Approved action",
        payload={"target": "approved"},
        permission_key="test_action:approved",
        tier=TIER_LOW,
    )
    untouched = store.queue_action(
        action_type="test_action",
        description="Untouched action",
        payload={"target": "untouched"},
        permission_key="test_action:untouched",
        tier=TIER_LOW,
    )
    calls = []

    def executor(action):
        calls.append(action.id)
        return True, "done"

    outcome = parse_approval_response(
        f"{approved.id} yes",
        store,
        executor_fn=executor,
    )

    assert outcome == [
        {
            "id": approved.id,
            "decision": "approved",
            "approved": True,
            "remembered": False,
            "status": STATUS_EXECUTED,
            "execution": "succeeded",
        }
    ]
    assert calls == [approved.id]
    assert store.get_action(approved.id).status == STATUS_EXECUTED
    assert store.get_action(untouched.id).status == STATUS_PENDING
    store.close()


def test_channel_approval_parser_remembers_only_supported_tiers(tmp_path) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))
    action = store.queue_action(
        action_type="test_action",
        description="High impact action",
        payload={},
        permission_key="test_action:high",
        tier=TIER_HIGH,
    )

    outcome = parse_approval_response(
        f"always yes {action.id}",
        store,
        executor_fn=lambda _action: (True, "done"),
    )

    assert outcome[0]["decision"] == "approved"
    assert outcome[0]["remembered"] is False
    assert store.get_permission(action.permission_key) is None
    store.close()


def test_channel_approval_parser_rejects_embedded_approval_text(tmp_path) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))
    action = store.queue_action(
        action_type="test_action",
        description="Approval-looking text quoted in a message",
        payload={},
        permission_key="test_action:quoted",
        tier=TIER_LOW,
    )
    calls = []

    def executor(pending):
        calls.append(pending.id)
        return True, "done"

    result = parse_approval_response(
        f"Please review this notification: {action.id} yes",
        store,
        executor_fn=executor,
    )

    assert result == []
    assert calls == []
    assert store.get_action(action.id).status == STATUS_PENDING
    store.close()


@pytest.mark.parametrize(
    "payload",
    [
        {"contact": None, "body": "hello"},
        {"contact": 15551234567, "body": "hello"},
        {"contact": " \n", "body": "hello"},
        {"contact": "+15551234567", "body": None},
        {"contact": "+15551234567", "body": ["hello"]},
        {"contact": "+15551234567", "body": " \n"},
    ],
)
def test_invalid_sms_draft_is_rejected_before_queueing(tmp_path, payload) -> None:
    store = ApprovalStore(str(tmp_path / "approvals.db"))

    result = QueueActionTool(store=store).execute(
        action_type="sms_draft_reply",
        description="Reply to the message",
        payload=payload,
        permission_key="sms_draft_reply:contact:+15551234567",
        tier=TIER_LOW,
    )

    assert result.success is False
    assert result.metadata == {"status": "rejected"}
    assert store.list_pending() == []
    assert store.list_approved() == []
    store.close()
