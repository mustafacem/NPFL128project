"""Unit tests for conversation history management."""

import json
from pathlib import Path

from talker.dialogue import (
    DEFAULT_SYSTEM_PROMPT,
    DialogueManager,
    is_farewell,
    load_history_json,
)


def test_history_starts_with_system_prompt() -> None:
    """A fresh DialogueManager seeds history with the system prompt."""
    manager = DialogueManager()

    assert manager.history == [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    ]


def test_history_records_turns_in_order() -> None:
    """User and assistant turns are appended in the order they occur."""
    manager = DialogueManager(system_prompt="persona")

    manager.add_user_turn("hi there")
    manager.add_assistant_turn("hello!")

    assert manager.history == [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello!"},
    ]


def test_history_property_returns_a_copy() -> None:
    """Mutating the returned history does not affect internal state."""
    manager = DialogueManager()

    snapshot = manager.history
    snapshot.append({"role": "user", "content": "ignored"})

    assert manager.history == [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    ]


def test_is_farewell_matches_known_keywords() -> None:
    """Farewell keywords are matched case-insensitively as substrings."""
    assert is_farewell("Okay, Goodbye!")
    assert is_farewell("gotta run, BYE")
    assert not is_farewell("what's the weather like")


def test_is_farewell_empty_text_is_false() -> None:
    """Empty input is never treated as a farewell."""
    assert not is_farewell("")


def test_save_json_round_trips_history(tmp_path: Path) -> None:
    """Saving then loading a history yields the same messages."""
    manager = DialogueManager(system_prompt="persona")
    manager.add_user_turn("hi")
    manager.add_assistant_turn("hello")
    out_path = tmp_path / "history.json"

    manager.save_json(out_path)
    loaded = load_history_json(out_path)

    assert loaded == manager.history


def test_save_json_writes_trailing_newline(tmp_path: Path) -> None:
    """The written file ends with a newline, per project style rules."""
    manager = DialogueManager()
    out_path = tmp_path / "history.json"

    manager.save_json(out_path)

    contents = out_path.read_text(encoding="utf-8")
    assert contents.endswith("\n")
    json.loads(contents)  # still valid JSON despite the trailing newline
