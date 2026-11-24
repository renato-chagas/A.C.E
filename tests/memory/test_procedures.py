"""Tests for conservative, review-only procedural memory proposals."""

from __future__ import annotations

import time

from openjarvis.core.events import EventBus
from openjarvis.memory.episodes import EpisodeStore
from openjarvis.memory.procedures import (
    build_procedure_candidate,
    is_explicit_failure,
    is_explicit_success,
    procedure_title_similarity,
    skill_markdown,
)
from openjarvis.memory.service import MemoryService, publish_completed_exchange
from openjarvis.memory.store import LocalFactStore

_USER_TASK = "How do I fix the database connection problem?"
_ASSISTANT_STEPS = """Try these steps:
1. Check that the database service is running.
2. Verify the host and port in the configuration.
3. Restart the client and retry the connection.
"""


class _Extractor:
    def extract(self, user_text: str, assistant_text: str = "") -> list[str]:
        return []


class _CleanScanner:
    def scan(self, text: str):
        from types import SimpleNamespace

        return SimpleNamespace(is_clean=True, threat_level="low", findings=[])


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_explicit_success_rejects_negative_uncertain_and_questions():
    assert is_explicit_success("Funcionou, obrigado!")
    assert is_explicit_success("That worked.")
    assert not is_explicit_success("Não funcionou")
    assert not is_explicit_success("Ainda não deu certo")
    assert not is_explicit_success("Acho que funcionou")
    assert not is_explicit_success("Será que funcionou?")


def test_explicit_failure_requires_a_direct_unambiguous_report():
    assert is_explicit_failure("Não funcionou.")
    assert is_explicit_failure("That did not work.")
    assert is_explicit_failure("Still broken after restarting.")
    assert not is_explicit_failure("Maybe it did not work.")
    assert not is_explicit_failure("Did it fail?")
    assert not is_explicit_failure("It did not fail; it works now.")
    assert not is_explicit_failure("Não funcionou no início, mas funcionou depois.")
    assert not is_explicit_failure("It did not work before, but it works now.")


def test_candidate_requires_multiple_structured_steps():
    candidate = build_procedure_candidate(
        episode_id="ab12cd34ef56ab78",
        user_text=_USER_TASK,
        assistant_text=_ASSISTANT_STEPS,
        success_signal="Funcionou!",
    )
    assert candidate is not None
    assert candidate.skill_name == "ace-procedure-ab12cd34ef56ab78"
    assert "Check that the database service is running" in candidate.draft_markdown
    assert "Try these steps" not in candidate.draft_markdown

    assert (
        build_procedure_candidate(
            episode_id="ab12cd34ef56ab78",
            user_text=_USER_TASK,
            assistant_text="Check the configuration and restart the app.",
            success_signal="Funcionou!",
        )
        is None
    )
    assert (
        build_procedure_candidate(
            episode_id="ab12cd34ef56ab78",
            user_text=_USER_TASK,
            assistant_text=_ASSISTANT_STEPS,
            success_signal="Não funcionou.",
        )
        is None
    )


def test_procedure_title_similarity_requires_shared_specific_terms():
    assert procedure_title_similarity(
        "Fix database connection problem",
        "Repair database connection problem",
    ) == 1.0
    assert procedure_title_similarity(
        "Fix database connection problem",
        "Fix display resolution",
    ) == 0.0
    assert procedure_title_similarity("Fix it", "Fix this") == 0.0


def test_episode_session_migration_and_proposal_review_states(tmp_path):
    import sqlite3

    db_path = tmp_path / "episodes.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE episodes (row_id INTEGER PRIMARY KEY, episode_id TEXT NOT NULL "
        "UNIQUE, user_text TEXT NOT NULL, assistant_text TEXT NOT NULL DEFAULT '', "
        "source TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL)"
    )
    connection.commit()
    connection.close()

    store = EpisodeStore(db_path)
    try:
        assert store.record(
            _USER_TASK,
            _ASSISTANT_STEPS,
            source="test",
            session_id="session-one",
        )
        [episode] = store.list_for_session("session-one")
        assert episode.session_id == "session-one"
        assert store.list_for_session("session-two") == []

        candidate = build_procedure_candidate(
            episode_id=episode.episode_id,
            user_text=episode.user_text,
            assistant_text=episode.assistant_text,
            success_signal="Deu certo!",
        )
        assert candidate is not None
        proposal = store.create_procedure_proposal(
            candidate,
            evidence_episode_id=episode.episode_id,
            session_id="session-one",
        )
        assert proposal is not None
        duplicate = store.create_procedure_proposal(
            candidate,
            evidence_episode_id=episode.episode_id,
            session_id="session-one",
        )
        assert duplicate is not None
        assert duplicate.proposal_id == proposal.proposal_id
        assert len(store.list_procedure_proposals()) == 1

        assert store.set_procedure_proposal_status(
            proposal.proposal_id,
            status="approved",
            skill_path="/skills/ace-procedure-test/SKILL.md",
        )
        approved = store.get_procedure_proposal(proposal.proposal_id[:8])
        assert approved is not None
        assert approved.status == "approved"
        assert approved.skill_path.endswith("SKILL.md")
        assert not store.set_procedure_proposal_status(
            proposal.proposal_id,
            status="rejected",
        )
    finally:
        store.close()


def test_skill_markdown_invocation_policy_is_loaded(tmp_path):
    from openjarvis.skills.loader import load_skill_markdown

    candidate = build_procedure_candidate(
        episode_id="ab12cd34ef56ab78",
        user_text=_USER_TASK,
        assistant_text=_ASSISTANT_STEPS,
        success_signal="Worked!",
    )
    assert candidate is not None
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text(
        skill_markdown(candidate, "ab12cd34ef56ab78"),
        encoding="utf-8",
    )
    manifest = load_skill_markdown(skill_path)
    assert manifest.name == candidate.skill_name
    assert manifest.user_invocable is True
    assert manifest.disable_model_invocation is False
    assert "explicitly asks to use this skill by name" in manifest.markdown_content


def test_memory_service_records_success_and_failure_in_same_recent_session(tmp_path):
    episodes = EpisodeStore(tmp_path / "episodes.db")
    service = MemoryService(
        LocalFactStore(tmp_path / "facts.jsonl"),
        _Extractor(),
        scanner=_CleanScanner(),
        secret_scanner=_CleanScanner(),
        episode_store=episodes,
        procedure_proposals_enabled=True,
    )
    try:
        service._process(
            (_USER_TASK, _ASSISTANT_STEPS, "test", "exchange-one", "session-one")
        )
        service._process(
            ("Funcionou, obrigado!", "Que bom.", "test", "exchange-two", "session-one")
        )
        proposals = episodes.list_procedure_proposals()
        assert len(proposals) == 1
        assert (
            proposals[0].evidence_episode_id
            == episodes.list_for_session("session-one", limit=10)[1].episode_id
        )

        service._process(
            (
                "Actually, it did not work after all.",
                "I will review the steps.",
                "test",
                "exchange-three",
                "session-one",
            )
        )
        [failure] = episodes.similar_procedure_outcomes(
            _USER_TASK,
            session_id="another-session",
        )
        assert failure.outcome.outcome == "failure"

        service._process(
            ("Funcionou!", "Ótimo.", "test", "exchange-four", "other-session")
        )
        assert len(episodes.list_procedure_proposals()) == 1
    finally:
        episodes.close()


def test_similar_proposals_are_comparable_across_sessions_only(tmp_path):
    store = EpisodeStore(tmp_path / "related-procedures.db")

    def add_proposal(title: str, session_id: str):
        request = (
            f"{title}. "
            "I need a repeatable fix for this recurring issue and want to compare "
            "it with previous results in future conversations. " * 2
        )
        assert store.record(request, _ASSISTANT_STEPS, session_id=session_id)
        [episode] = store.list_for_session(session_id, limit=1)
        candidate = build_procedure_candidate(
            episode_id=episode.episode_id,
            user_text=title,
            assistant_text=episode.assistant_text,
            success_signal="Funcionou!",
        )
        assert candidate is not None
        proposal = store.create_procedure_proposal(
            candidate,
            evidence_episode_id=episode.episode_id,
            session_id=session_id,
        )
        assert proposal is not None
        return proposal

    try:
        earlier = add_proposal(
            "Fix database connection problem",
            "session-one",
        )
        current = add_proposal(
            "Repair database connection problem",
            "session-two",
        )
        add_proposal("Fix display resolution", "session-three")
        add_proposal("Fix database connection problem", "session-two")

        matches = store.similar_procedure_proposals(current.proposal_id)

        assert [item.proposal.proposal_id for item in matches] == [
            earlier.proposal_id
        ]
        assert matches[0].similarity == 1.0
        assert matches[0].proposal.success_signal == "Funcionou!"
    finally:
        store.close()


def test_explicit_failure_outcome_persists_and_cascades_with_its_episode(tmp_path):
    path = tmp_path / "procedure-outcomes.db"
    store = EpisodeStore(path)
    try:
        assert store.record(_USER_TASK, _ASSISTANT_STEPS, session_id="failure-session")
        assert store.record(
            "Não funcionou.",
            "Vou rever as etapas.",
            session_id="failure-session",
        )
        [failure_episode] = store.list_for_session("failure-session", limit=1)
        outcome = store.record_procedure_outcome(
            evidence_episode_id=failure_episode.episode_id,
            session_id="failure-session",
            task_title=_USER_TASK,
            outcome="failure",
            signal=failure_episode.user_text,
        )
        assert outcome is not None
        assert store.record_procedure_outcome(
            evidence_episode_id=failure_episode.episode_id,
            session_id="other-session",
            task_title=_USER_TASK,
            outcome="failure",
            signal="Not the same conversation",
        ) is None
        [related] = store.similar_procedure_outcomes(
            _USER_TASK,
            session_id="review-session",
        )
        assert related.outcome.outcome == "failure"
        assert related.outcome.signal == "Não funcionou."
    finally:
        store.close()

    reopened = EpisodeStore(path)
    try:
        [related] = reopened.similar_procedure_outcomes(
            "How can I repair the database connection problem?",
            session_id="review-session",
        )
        assert related.outcome.evidence_episode_id == failure_episode.episode_id
        assert reopened.delete(failure_episode.episode_id)
        assert reopened.similar_procedure_outcomes(
            _USER_TASK,
            session_id="review-session",
        ) == []
    finally:
        reopened.close()


def test_completed_exchange_event_carries_cli_session_into_proposals(tmp_path):
    bus = EventBus(record_history=True)
    episodes = EpisodeStore(tmp_path / "events.db")
    service = MemoryService(
        LocalFactStore(tmp_path / "event-facts.jsonl"),
        _Extractor(),
        event_bus=bus,
        scanner=_CleanScanner(),
        secret_scanner=_CleanScanner(),
        episode_store=episodes,
        procedure_proposals_enabled=True,
    )
    service.start()
    try:
        assert publish_completed_exchange(
            bus,
            _USER_TASK,
            _ASSISTANT_STEPS,
            source="cli.chat",
            session_id="chat-session",
        )
        assert publish_completed_exchange(
            bus,
            "Funcionou, obrigado!",
            "Que bom.",
            source="cli.chat",
            session_id="chat-session",
        )
        assert _wait_until(lambda: len(episodes.list_procedure_proposals()) == 1)
    finally:
        service.stop()
        episodes.close()
