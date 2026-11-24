"""Tests for local, named, bounded working history."""

import pytest

from openjarvis.sessions.working_chat import (
    SavedChatChanged,
    WorkingChatStore,
    validate_chat_name,
)


def test_named_chat_restores_only_its_completed_turns(tmp_path):
    path = tmp_path / "working-chats.db"
    store = WorkingChatStore(path)
    work = store.append(store.open("Planejamento"), "Task A", "Draft A")
    store.append(store.open("Other"), "Task B", "Draft B")

    reopened = WorkingChatStore(path)
    assert reopened.open("Planejamento") == work
    assert reopened.open("Other").turns == (("Task B", "Draft B"),)
    assert set(reopened.list_names()) == {"Planejamento", "Other"}


def test_stale_chat_cannot_overwrite_new_turns_or_recreate_deleted_chat(tmp_path):
    path = tmp_path / "working-chats.db"
    first = WorkingChatStore(path)
    second = WorkingChatStore(path)
    original = first.open("Continue")
    updated = second.append(second.open("Continue"), "Question", "Answer")

    with pytest.raises(SavedChatChanged):
        first.clear(original)
    with pytest.raises(SavedChatChanged):
        first.append(original, "Stale question", "Stale answer")

    assert second.open("Continue") == updated
    assert second.forget("Continue")
    fresh = first.open("Continue")
    assert fresh.session_id != updated.session_id
    with pytest.raises(SavedChatChanged):
        first.append(updated, "Stale question", "Stale answer")
    assert not second.open("Continue").turns


def test_history_bounds_keep_complete_pairs(tmp_path):
    store = WorkingChatStore(tmp_path / "working-chats.db")
    chat = store.open("Bounded")
    for i in range(30):
        chat = store.append(chat, f"Question {i}", f"Answer {i}")
    assert len(chat.turns) == 12
    assert chat.turns[0] == ("Question 18", "Answer 18")

    chat = store.append(chat, "p" * 4_000, "a" * 5_000)
    chat = store.append(chat, "q" * 4_000, "b" * 5_000)
    [final_turn] = chat.turns
    assert all(len(text) <= 3_000 for text in final_turn)
    assert all(text.endswith("[Saved text shortened.]") for text in final_turn)
    cleared = store.clear(chat)
    assert cleared.turns == ()
    assert cleared.session_id != chat.session_id


@pytest.mark.parametrize("name", ["", " " * 3, "with\nnewline", "x" * 81])
def test_invalid_chat_name(name):
    with pytest.raises(ValueError, match="80 printable"):
        validate_chat_name(name)
