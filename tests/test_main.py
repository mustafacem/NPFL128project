"""Unit tests for argument parsing and the conversation turn helper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openai import APIError

from talker.__main__ import (
    DEFAULT_HISTORY_PATH,
    exchange,
    load_system_prompt,
    parse_args,
    reply_to,
    require_env,
)
from talker.dialogue import DEFAULT_SYSTEM_PROMPT, DialogueManager


def test_parse_args_defaults() -> None:
    """Defaults select the hosted backends and voice mode."""
    args = parse_args([])

    assert args.stt == "openai"
    assert args.tts == "openai"
    assert args.text_mode is False
    assert args.history_out == DEFAULT_HISTORY_PATH


def test_parse_args_selects_offline_backends() -> None:
    """Backends can be switched to the offline implementations."""
    args = parse_args(["--stt", "local", "--tts", "gtts", "--text-mode"])

    assert args.stt == "local"
    assert args.tts == "gtts"
    assert args.text_mode is True


def test_parse_args_history_out_is_a_path() -> None:
    """Paths are parsed into pathlib objects, not left as strings."""
    args = parse_args(["--history-out", "out/session.json"])

    assert args.history_out == Path("out/session.json")


def test_parse_args_rejects_unknown_backend() -> None:
    """An unsupported backend name is rejected by argparse."""
    with pytest.raises(SystemExit):
        parse_args(["--stt", "vosk"])


def test_load_system_prompt_defaults_without_file() -> None:
    """With no file given, the built-in persona is used."""
    assert load_system_prompt(None) == DEFAULT_SYSTEM_PROMPT


def test_load_system_prompt_reads_file(tmp_path: Path) -> None:
    """A prompt file's contents are read and stripped."""
    prompt_file = tmp_path / "persona.txt"
    with prompt_file.open("w", encoding="utf-8") as handle:
        handle.write("  You are a pirate.\n")

    assert load_system_prompt(prompt_file) == "You are a pirate."


def test_load_system_prompt_missing_file_exits(tmp_path: Path) -> None:
    """An unreadable prompt file stops the program with a message."""
    with pytest.raises(SystemExit):
        load_system_prompt(tmp_path / "does-not-exist.txt")


def test_require_env_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A set environment variable is returned unchanged."""
    monkeypatch.setenv("TALKER_TEST_KEY", "secret")

    assert require_env("TALKER_TEST_KEY") == "secret"


def test_require_env_exits_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing environment variable stops the program."""
    monkeypatch.delenv("TALKER_TEST_KEY", raising=False)

    with pytest.raises(SystemExit):
        require_env("TALKER_TEST_KEY")


def test_reply_to_reports_api_failure_without_crashing() -> None:
    """An unreachable model ends the turn with None, not an exception."""
    manager = DialogueManager(system_prompt="persona")
    failure = APIError("no credits", request=MagicMock(), body=None)

    with patch("talker.llm.generate_reply", side_effect=failure):
        reply = reply_to(manager, "hello", MagicMock(), "some/model")

    assert reply is None


def test_failed_turn_leaves_history_unchanged() -> None:
    """A failed request records neither the question nor a reply."""
    manager = DialogueManager(system_prompt="persona")
    failure = APIError("no credits", request=MagicMock(), body=None)

    with patch("talker.llm.generate_reply", side_effect=failure):
        reply_to(manager, "hello", MagicMock(), "some/model")

    assert manager.history == [{"role": "system", "content": "persona"}]


def test_exchange_records_both_turns() -> None:
    """A turn appends the user utterance and the generated reply."""
    manager = DialogueManager(system_prompt="persona")
    client = MagicMock()

    with patch("talker.llm.generate_reply", return_value="Hi!") as generate:
        reply = exchange(manager, "hello", client, "some/model")

    assert reply == "Hi!"
    assert manager.history == [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    # The user turn must be in the history before the model is queried.
    sent_messages, _ = generate.call_args
    assert sent_messages[0][-1] == {"role": "user", "content": "hello"}
