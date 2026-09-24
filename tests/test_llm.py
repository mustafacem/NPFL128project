"""Unit tests for the LLM module using a mocked OpenRouter client."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from talker.llm import DEFAULT_LLM_MODEL, OPENROUTER_BASE_URL, generate_reply


def _make_mock_client(content: object) -> MagicMock:
    """Build a mock client whose completion returns ``content``.

    Args:
        content: The message content the mocked completion should return.

    Returns:
        A MagicMock standing in for an OpenRouter-backed client.
    """
    client = MagicMock()
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[choice]
    )
    return client


def test_generate_reply_returns_stripped_content() -> None:
    """The reply text is returned without surrounding whitespace."""
    client = _make_mock_client("  Hello there.\n")

    reply = generate_reply([{"role": "user", "content": "hi"}], client)

    assert reply == "Hello there."


def test_generate_reply_handles_empty_content() -> None:
    """A model returning no content yields an empty string, not None."""
    client = _make_mock_client(None)

    reply = generate_reply([{"role": "user", "content": "hi"}], client)

    assert reply == ""


def test_generate_reply_forwards_model_and_history() -> None:
    """The full conversation and chosen model reach the API call."""
    client = _make_mock_client("ok")
    messages = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hi"},
    ]

    generate_reply(messages, client, model="meta-llama/llama-3-8b-instruct")

    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["model"] == "meta-llama/llama-3-8b-instruct"
    assert kwargs["messages"] == messages


def test_generate_reply_uses_default_model() -> None:
    """Omitting the model falls back to the documented default."""
    client = _make_mock_client("ok")

    generate_reply([{"role": "user", "content": "hi"}], client)

    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["model"] == DEFAULT_LLM_MODEL


def test_openrouter_base_url_is_the_api_endpoint() -> None:
    """The base URL points at OpenRouter's OpenAI-compatible API."""
    assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
