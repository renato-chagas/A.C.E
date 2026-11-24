from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from click.testing import CliRunner

from openjarvis.cli import memory_cmd
from openjarvis.memory.procedures import (
    ProcedureOutcome,
    ProcedureProposal,
    RelatedProcedureOutcome,
    RelatedProcedureProposal,
    skill_markdown,
)


def _proposal(
    proposal_id: str,
    *,
    session_id: str,
    title: str,
    status: str = "proposed",
) -> ProcedureProposal:
    return ProcedureProposal(
        proposal_id=proposal_id,
        evidence_episode_id=f"episode-{proposal_id}",
        session_id=session_id,
        skill_name=f"ace-procedure-{proposal_id[:16]}",
        title=title,
        description="User-reviewed workflow.",
        draft_markdown="1. Check the setting.\n2. Retry the task.",
        success_signal="Funcionou!",
        status=status,
        skill_path="",
        created_at=1.0,
        updated_at=1.0,
    )


def test_show_procedure_includes_related_cross_session_drafts(monkeypatch) -> None:
    current = _proposal(
        "ab1234567890",
        session_id="session-current",
        title="Fix database connection problem",
    )
    earlier = _proposal(
        "cd1234567890",
        session_id="session-earlier",
        title="Repair database connection problem",
        status="approved",
    )

    class Store:
        closed = False

        def similar_procedure_proposals(self, proposal_id: str):
            assert proposal_id == current.proposal_id
            return [RelatedProcedureProposal(earlier, 1.0)]

        def similar_procedure_outcomes(self, task_title: str, *, session_id: str):
            assert task_title == current.title
            assert session_id == current.session_id
            outcome = ProcedureOutcome(
                outcome_id="outcome123456",
                evidence_episode_id="episode-failure123456",
                session_id="session-failure",
                task_title="Repair database connection problem",
                outcome="failure",
                signal="It still fails after the restart.",
                created_at=1.0,
            )
            return [RelatedProcedureOutcome(outcome, 1.0)]

        def close(self):
            self.closed = True

    store = Store()
    monkeypatch.setattr(
        memory_cmd,
        "_load_procedure_proposal",
        lambda _proposal_id: (store, current),
    )

    result = CliRunner().invoke(
        memory_cmd.show_procedure_proposal,
        [current.proposal_id],
    )

    assert result.exit_code == 0, result.output
    assert "Possible matches from other conversations" in result.output
    assert "approved" in result.output
    assert "Repair database connection problem" in result.output
    assert "Earlier success signal: Funcionou!" in result.output
    assert "Reported failures on similar tasks" in result.output
    assert "It still fails after the restart." in result.output
    assert store.closed


def test_outcomes_command_lists_source_episode_and_failure_report(monkeypatch) -> None:
    class Store:
        closed = False

        def list_procedure_outcomes(self, *, limit: int):
            assert limit == 2
            return [
                ProcedureOutcome(
                    outcome_id="outcome123456",
                    evidence_episode_id="episode-failure123456",
                    session_id="session-failure",
                    task_title="Fix database connection problem",
                    outcome="failure",
                    signal="Não funcionou depois da reinicialização.",
                    created_at=1.0,
                )
            ]

        def close(self):
            self.closed = True

    store = Store()
    monkeypatch.setattr(memory_cmd, "_get_episode_store", lambda: store)

    result = CliRunner().invoke(
        memory_cmd.procedures_group.commands["outcomes"],
        ["--limit", "2"],
    )

    assert result.exit_code == 0, result.output
    assert "episode-failure123456" in result.output
    assert "Fix database connection problem" in result.output
    assert "Não funcionou depois da reinicialização." in result.output
    assert store.closed



def test_approve_can_create_a_new_version_of_an_existing_procedure(
    monkeypatch,
    tmp_path,
) -> None:
    proposal = _proposal(
        "ab1234567890",
        session_id="session-current",
        title="Improve database recovery",
    )
    skill_name = "ace-procedure-1122334455667788"
    active_dir = tmp_path / "skills" / skill_name
    active_dir.mkdir(parents=True)
    active_file = active_dir / "SKILL.md"
    old_proposal = replace(
        proposal,
        skill_name=skill_name,
        draft_markdown="1. Old step.\n2. Retry once.",
    )
    old_content = skill_markdown(
        old_proposal,
        proposal.evidence_episode_id,
        version="0.1.0",
    )
    active_file.write_text(old_content, encoding="utf-8")
    config = SimpleNamespace(
        skills=SimpleNamespace(skills_dir=str(tmp_path / "skills"))
    )

    class Store:
        closed = False
        status_call = None

        def set_procedure_proposal_status(
            self,
            proposal_id: str,
            *,
            status: str,
            skill_path: str,
        ) -> bool:
            self.status_call = (proposal_id, status, skill_path)
            return True

        def close(self) -> None:
            self.closed = True

    store = Store()
    monkeypatch.setattr(
        memory_cmd,
        "_load_procedure_proposal",
        lambda _proposal_id: (store, proposal),
    )
    monkeypatch.setattr(memory_cmd, "load_config", lambda: config)
    monkeypatch.setattr(memory_cmd, "get_config_dir", lambda: tmp_path / "config")

    result = CliRunner().invoke(
        memory_cmd.procedures_group.commands["approve"],
        [proposal.proposal_id, "--update-skill", skill_name],
        input="y\n",
    )

    active_content = active_file.read_text(encoding="utf-8")
    history_dir = tmp_path / "config" / "skill_versions" / skill_name
    assert result.exit_code == 0, result.output
    assert "Version: 0.1.0 → 0.1.1" in result.output
    assert "version: 0.1.1" in active_content
    assert "1. Old step." not in active_content
    assert (history_dir / "0.1.0.md").read_text(encoding="utf-8") == old_content
    assert (history_dir / "0.1.1.md").read_text(encoding="utf-8") == active_content
    assert store.status_call == (
        proposal.proposal_id,
        "approved",
        str(active_file),
    )
    assert store.closed

    versions = CliRunner().invoke(
        memory_cmd.procedures_group.commands["versions"],
        [skill_name],
    )
    assert versions.exit_code == 0, versions.output
    assert "0.1.0" in versions.output
    assert "0.1.1" in versions.output
    assert "active" in versions.output


def test_rollback_restores_archived_body_as_a_new_version(
    monkeypatch,
    tmp_path,
) -> None:
    from openjarvis.memory.skill_versions import (
        list_skill_versions,
        record_skill_version,
    )

    skill_name = "ace-procedure-1122334455667788"
    skills_root = tmp_path / "skills"
    active_dir = skills_root / skill_name
    active_dir.mkdir(parents=True)
    active_file = active_dir / "SKILL.md"
    old_proposal = _proposal(
        "cd1234567890",
        session_id="session-old",
        title="Old database recovery",
    )
    old_proposal = replace(
        old_proposal,
        skill_name=skill_name,
        draft_markdown="1. Old database recovery.\n2. Retry once.",
    )
    old_content = skill_markdown(
        old_proposal,
        old_proposal.evidence_episode_id,
        version="0.1.0",
    )
    current_proposal = replace(
        old_proposal,
        title="New database recovery",
        draft_markdown="1. New check.\n2. Retry twice.",
    )
    current_content = skill_markdown(
        current_proposal,
        current_proposal.evidence_episode_id,
        version="0.1.1",
    )
    active_file.write_text(current_content, encoding="utf-8")
    history_root = tmp_path / "config" / "skill_versions"
    record_skill_version(history_root, skill_name, old_content)
    record_skill_version(history_root, skill_name, current_content)
    config = SimpleNamespace(skills=SimpleNamespace(skills_dir=str(skills_root)))
    monkeypatch.setattr(memory_cmd, "load_config", lambda: config)
    monkeypatch.setattr(memory_cmd, "get_config_dir", lambda: tmp_path / "config")

    result = CliRunner().invoke(
        memory_cmd.procedures_group.commands["rollback"],
        [skill_name, "0.1.0"],
        input="y\n",
    )

    active_after = active_file.read_text(encoding="utf-8")
    assert result.exit_code == 0, result.output
    assert "0.1.0 → 0.1.2" in result.output
    assert "version: 0.1.2" in active_after
    assert "1. Old database recovery." in active_after
    snapshots = list_skill_versions(history_root, skill_name)
    assert [item.version for item in snapshots] == ["0.1.0", "0.1.1", "0.1.2"]



def test_failed_proposal_state_restores_previous_active_skill(
    monkeypatch,
    tmp_path,
) -> None:
    proposal = _proposal(
        "ef1234567890",
        session_id="session-current",
        title="Update database recovery",
    )
    skill_name = "ace-procedure-1122334455667788"
    skills_root = tmp_path / "skills"
    active_dir = skills_root / skill_name
    active_dir.mkdir(parents=True)
    active_file = active_dir / "SKILL.md"
    existing_proposal = replace(proposal, skill_name=skill_name)
    old_content = skill_markdown(
        existing_proposal,
        proposal.evidence_episode_id,
        version="0.1.0",
    )
    active_file.write_text(old_content, encoding="utf-8")
    config = SimpleNamespace(skills=SimpleNamespace(skills_dir=str(skills_root)))

    class Store:
        def set_procedure_proposal_status(self, *args, **kwargs) -> bool:
            return False

        def close(self) -> None:
            pass

    store = Store()
    monkeypatch.setattr(
        memory_cmd,
        "_load_procedure_proposal",
        lambda _proposal_id: (store, proposal),
    )
    monkeypatch.setattr(memory_cmd, "load_config", lambda: config)
    monkeypatch.setattr(memory_cmd, "get_config_dir", lambda: tmp_path / "config")

    result = CliRunner().invoke(
        memory_cmd.procedures_group.commands["approve"],
        [proposal.proposal_id, "--update-skill", skill_name],
        input="y\n",
    )

    assert result.exit_code != 0
    assert "previous active skill was restored" in result.output
    assert active_file.read_text(encoding="utf-8") == old_content



def test_first_approval_archives_initial_skill_version(monkeypatch, tmp_path) -> None:
    proposal = _proposal(
        "ab1234567890",
        session_id="session-current",
        title="Recover a local database",
    )
    skills_root = tmp_path / "skills"
    config = SimpleNamespace(skills=SimpleNamespace(skills_dir=str(skills_root)))

    class Store:
        status_call = None
        closed = False

        def set_procedure_proposal_status(
            self,
            proposal_id: str,
            *,
            status: str,
            skill_path: str,
        ) -> bool:
            self.status_call = (proposal_id, status, skill_path)
            return True

        def close(self) -> None:
            self.closed = True

    store = Store()
    monkeypatch.setattr(
        memory_cmd,
        "_load_procedure_proposal",
        lambda _proposal_id: (store, proposal),
    )
    monkeypatch.setattr(memory_cmd, "load_config", lambda: config)
    monkeypatch.setattr(memory_cmd, "get_config_dir", lambda: tmp_path / "config")

    result = CliRunner().invoke(
        memory_cmd.procedures_group.commands["approve"],
        [proposal.proposal_id],
        input="y\n",
    )

    active_file = skills_root / proposal.skill_name / "SKILL.md"
    active_content = active_file.read_text(encoding="utf-8")
    archived_file = (
        tmp_path
        / "config"
        / "skill_versions"
        / proposal.skill_name
        / "0.1.0.md"
    )
    assert result.exit_code == 0, result.output
    assert "version: 0.1.0" in active_content
    assert archived_file.read_text(encoding="utf-8") == active_content
    assert store.status_call == (
        proposal.proposal_id,
        "approved",
        str(active_file),
    )
    assert store.closed
