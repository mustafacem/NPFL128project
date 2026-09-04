"""Unit tests for speech synthesis, with network and playback mocked."""

from unittest.mock import MagicMock, patch

import pytest

from talker.tts import (
    DEFAULT_TTS_MODEL,
    DEFAULT_VOICE,
    speak,
    synthesize_openai,
)


def _make_mock_client(audio_bytes: bytes) -> MagicMock:
    """Build a mock OpenAI client whose speech call returns ``audio_bytes``.

    Args:
        audio_bytes: The audio payload the mocked response should return.

    Returns:
        A MagicMock standing in for an OpenAI client.
    """
    client = MagicMock()
    client.audio.speech.create.return_value.read.return_value = audio_bytes
    return client


def test_synthesize_openai_returns_audio_bytes() -> None:
    """The response body is returned verbatim as audio bytes."""
    client = _make_mock_client(b"RIFF-fake-wav")

    result = synthesize_openai("hello", client)

    assert result == b"RIFF-fake-wav"


def test_synthesize_openai_requests_wav_and_defaults() -> None:
    """WAV is requested so libsndfile-based playback can decode it."""
    client = _make_mock_client(b"audio")

    synthesize_openai("hello", client)

    _, kwargs = client.audio.speech.create.call_args
    assert kwargs["response_format"] == "wav"
    assert kwargs["model"] == DEFAULT_TTS_MODEL
    assert kwargs["voice"] == DEFAULT_VOICE
    assert kwargs["input"] == "hello"


def test_speak_openai_backend_plays_synthesized_audio() -> None:
    """The openai backend hands its WAV bytes to the audio player."""
    client = _make_mock_client(b"RIFF-fake-wav")

    with patch("talker.audio.play_audio") as play_audio:
        speak("hello", backend="openai", client=client)

    play_audio.assert_called_once_with(b"RIFF-fake-wav")


def test_speak_gtts_backend_plays_mp3() -> None:
    """The gtts backend synthesizes MP3 and plays it with the MP3 player."""
    with patch("talker.tts.synthesize_gtts", return_value=b"ID3-fake-mp3"), \
            patch("talker.tts.play_mp3") as play_mp3:
        speak("hello", backend="gtts", language="cs")

    play_mp3.assert_called_once_with(b"ID3-fake-mp3")


def test_speak_empty_text_is_a_no_op() -> None:
    """Empty text never reaches a backend or the speakers."""
    client = _make_mock_client(b"audio")

    with patch("talker.audio.play_audio") as play_audio:
        speak("", backend="openai", client=client)

    play_audio.assert_not_called()
    client.audio.speech.create.assert_not_called()


def test_speak_unknown_backend_raises() -> None:
    """An unsupported backend name is rejected with a helpful message."""
    with pytest.raises(ValueError, match="Unknown TTS backend"):
        speak("hello", backend="espeak")


def test_speak_openai_backend_requires_client() -> None:
    """Selecting the openai backend without a client is an error."""
    with pytest.raises(ValueError, match="requires an OpenAI client"):
        speak("hello", backend="openai", client=None)
