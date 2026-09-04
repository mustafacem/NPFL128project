"""Conversation state: message history, farewell detection, JSON storage.

Kept free of any LLM/audio I/O so it can be unit tested in isolation.
"""

import json
from pathlib import Path
from typing import Dict, List, Sequence

DEFAULT_SYSTEM_PROMPT: str = (
    "You are a helpful, friendly voice assistant. Your input comes from a "
    "speech-to-text system, so it may contain transcription errors or odd "
    "phrasing; use context to infer the user's intent. Keep replies short, "
    "clear, and conversational, since they will be read aloud."
)
"""Default persona used when no system prompt is supplied."""

DEFAULT_FAREWELL_KEYWORDS: Sequence[str] = (
    "bye",
    "goodbye",
    "exit",
    "quit",
    "stop",
)
"""Words that, if present in a user utterance, end the conversation."""


class DialogueManager:
    """Tracks a single conversation's system prompt and turn history."""

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        """Start a new conversation with the given system prompt.

        Args:
            system_prompt: Instructions that set the assistant's persona and
                behavior. Stored as the first message in the history.
        """
        self._history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    @property
    def history(self) -> List[Dict[str, str]]:
        """A copy of the conversation history in OpenAI chat message format.

        Returns:
            A new list of ``{"role": ..., "content": ...}`` dictionaries; the
            returned list can be freely mutated by the caller.
        """
        return list(self._history)

    def add_user_turn(self, text: str) -> None:
        """Append a user utterance to the history.

        Args:
            text: The transcribed user utterance.

        Returns:
            None.
        """
        self._history.append({"role": "user", "content": text})

    def add_assistant_turn(self, text: str) -> None:
        """Append an assistant response to the history.

        Args:
            text: The assistant's reply text.

        Returns:
            None.
        """
        self._history.append({"role": "assistant", "content": text})

    def save_json(self, path: Path) -> None:
        """Write the conversation history to a JSON file.

        Args:
            path: Destination file path. Parent directories are not created.

        Returns:
            None. Side effect: overwrites ``path`` with the history as JSON.
        """
        with path.open("w", encoding="utf-8") as history_file:
            json.dump(
                self._history, history_file, ensure_ascii=False, indent=2
            )
            history_file.write("\n")


def is_farewell(
    text: str,
    keywords: Sequence[str] = DEFAULT_FAREWELL_KEYWORDS,
) -> bool:
    """Check whether an utterance signals the user wants to end the session.

    Args:
        text: The user utterance to check.
        keywords: Farewell words to match, matched case-insensitively as
            substrings of ``text``.

    Returns:
        True if any keyword occurs in ``text``, False otherwise (including
        when ``text`` is empty).
    """
    if not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def load_history_json(path: Path) -> List[Dict[str, str]]:
    """Read a conversation history previously written by ``save_json``.

    Args:
        path: Path to a JSON file containing a list of chat messages.

    Returns:
        The parsed list of ``{"role": ..., "content": ...}`` dictionaries.
    """
    with path.open("r", encoding="utf-8") as history_file:
        history: List[Dict[str, str]] = json.load(history_file)
    return history
