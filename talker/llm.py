"""LLM response generation via the OpenRouter chat completions API.

OpenRouter exposes an OpenAI-compatible endpoint, so the same ``openai``
client library is reused with a different base URL.
"""

from typing import Dict, List, Sequence

from openai import OpenAI

OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
"""Base URL of OpenRouter's OpenAI-compatible chat completions API."""

DEFAULT_LLM_MODEL: str = "qwen/qwen3-8b"
"""Default OpenRouter model used for dialogue responses."""


def build_client(api_key: str) -> OpenAI:
    """Create an OpenAI-compatible client pointed at OpenRouter.

    Args:
        api_key: An OpenRouter API key.

    Returns:
        A client that sends chat completion requests to OpenRouter.
    """
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def generate_reply(
    messages: Sequence[Dict[str, str]],
    client: OpenAI,
    model: str = DEFAULT_LLM_MODEL,
) -> str:
    """Generate the assistant's next reply for a conversation.

    Args:
        messages: The conversation so far, as chat messages with ``role`` and
            ``content`` keys, ordered oldest first.
        client: An OpenRouter-backed client, e.g. from :func:`build_client`.
        model: The OpenRouter model identifier to query.

    Returns:
        The assistant's reply text, stripped of surrounding whitespace. An
        empty string is returned if the model produced no content.
    """
    history: List[Dict[str, str]] = list(messages)
    completion = client.chat.completions.create(
        model=model,
        messages=history,
    )
    content = completion.choices[0].message.content
    return content.strip() if content else ""
