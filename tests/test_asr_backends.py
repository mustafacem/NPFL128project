"""Unit tests for ASR backend selection."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from talker.asr import build_transcriber


def test_openai_backend_calls_the_api() -> None:
    """The openai backend forwards audio and language to the Whisper API."""
    client = MagicMock()
    client.audio.transcriptions.create.return_value = SimpleNamespace(
        text=" hello "
    )

    transcriber = build_transcriber("openai", client=client, language="en")
    result = transcriber(b"fake-wav-bytes")

    assert result == "hello"
    _, kwargs = client.audio.transcriptions.create.call_args
    assert kwargs["language"] == "en"


def test_openai_backend_requires_client() -> None:
    """Selecting the openai backend without a client is an error."""
    with pytest.raises(ValueError, match="requires an OpenAI client"):
        build_transcriber("openai")


def test_local_backend_delegates_to_local_whisper() -> None:
    """The local backend routes audio through a LocalWhisper instance."""
    recognizer = MagicMock()
    recognizer.transcribe.return_value = "ahoj"

    with patch("talker.asr_local.LocalWhisper", return_value=recognizer):
        transcriber = build_transcriber("local", language="cs")
        result = transcriber(b"fake-wav-bytes")

    assert result == "ahoj"
    recognizer.transcribe.assert_called_once_with(
        b"fake-wav-bytes", language="cs"
    )


def test_unknown_backend_raises() -> None:
    """An unsupported backend name is rejected with a helpful message."""
    with pytest.raises(ValueError, match="Unknown ASR backend"):
        build_transcriber("vosk")
