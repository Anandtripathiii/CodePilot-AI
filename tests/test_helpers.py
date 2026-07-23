"""Tests for prompt building, file rules and history."""

import tempfile
from pathlib import Path

from utils import History, build_prompt, is_allowed_file, new_id


def test_allowed_extensions() -> None:
    assert is_allowed_file("main.py")
    assert is_allowed_file("App.JSX")
    assert not is_allowed_file("photo.png")
    assert not is_allowed_file("archive.zip")


def test_prompt_contains_question_and_task() -> None:
    prompt = build_prompt("debug", "my loop crashes")
    assert "my loop crashes" in prompt
    assert "USER REQUEST" in prompt


def test_context_is_included_when_given() -> None:
    prompt = build_prompt("ask", "what does this do", "def add(a, b): return a + b")
    assert "def add" in prompt


def test_ids_sort_chronologically() -> None:
    """uuid7 is time-ordered, so ids generated in sequence sort in sequence."""
    ids = [new_id() for _ in range(20)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 20


def test_history_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as folder:
        history = History(Path(folder) / "history.json")
        assert history.all() == []

        history.add(
            {"id": new_id(), "question": "hi", "answer": "hello",
             "mode": "ask", "provider": "gemini"}
        )
        assert len(history.all()) == 1

        history.clear()
        assert history.all() == []


def test_history_respects_limit() -> None:
    with tempfile.TemporaryDirectory() as folder:
        history = History(Path(folder) / "history.json", limit=5)
        for n in range(12):
            history.add({"id": new_id(), "question": str(n)})
        items = history.all()
        assert len(items) == 5
        assert items[-1]["question"] == "11"
