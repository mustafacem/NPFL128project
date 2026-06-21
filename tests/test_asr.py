"""Unit tests for the ASR module using a mocked OpenAI client."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from talker.asr import DEFAULT_ASR_MODEL, transcribe


def _make_mock_client(text: str) -> MagicMock:
    """Build a mock OpenAI client whose transcription returns ``text``.

    Args:
        text: The transcript text the mock should return.

    Returns:
        A MagicMock standing in for an OpenAI client.
    """
    client = MagicMock()
    create = client.audio.transcriptions.create
    create.return_value = SimpleNamespace(text=text)
    return client


def test_transcribe_returns_stripped_text() -> None:
    """The transcript is returned with surrounding whitespace removed."""
    client = _make_mock_client("  hello world  \n")

    result = transcribe(b"fake-wav-bytes", client)

    assert result == "hello world"


def test_transcribe_passes_model_and_language() -> None:
    """The configured model and language are forwarded to the API call."""
    client = _make_mock_client("ahoj")

    transcribe(b"fake-wav-bytes", client, language="cs")

    _, kwargs = client.audio.transcriptions.create.call_args
    assert kwargs["model"] == DEFAULT_ASR_MODEL
    assert kwargs["language"] == "cs"
    # The uploaded file must carry a .wav name so Whisper infers the format.
    assert kwargs["file"].name == "audio.wav"
