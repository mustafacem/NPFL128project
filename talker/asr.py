"""Automatic speech recognition via the OpenAI Whisper API."""

import io
from typing import Optional

from openai import OpenAI, omit

DEFAULT_ASR_MODEL: str = "whisper-1"
"""Default Whisper model used for transcription."""


def transcribe(
    audio_bytes: bytes,
    client: OpenAI,
    model: str = DEFAULT_ASR_MODEL,
    language: Optional[str] = None,
) -> str:
    """Transcribe encoded audio to text using the OpenAI Whisper API.

    Args:
        audio_bytes: WAV-encoded audio data to transcribe.
        client: An initialized OpenAI client used to make the request.
        model: The Whisper model name to use for transcription.
        language: Optional ISO-639-1 language code (e.g. ``"en"``) to guide
            recognition. If ``None``, the language is auto-detected.

    Returns:
        The transcribed text with leading and trailing whitespace stripped.
    """
    # Whisper infers the audio format from the file name, so the BytesIO
    # buffer must carry a recognizable ".wav" name.
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"

    # Passing NOT_GIVEN lets Whisper auto-detect the language.
    response = client.audio.transcriptions.create(
        model=model,
        file=audio_file,
        language=language if language is not None else omit,
    )
    return response.text.strip()
