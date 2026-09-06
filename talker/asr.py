"""Automatic speech recognition, via the OpenAI Whisper API or a local model.

The ``openai`` backend calls the hosted Whisper API; the ``local`` backend
runs a Whisper checkpoint on this machine (see :mod:`talker.asr_local`).
:func:`build_transcriber` hides that choice behind one callable.
"""

import io
from typing import Callable, Optional

from openai import OpenAI, omit

DEFAULT_ASR_MODEL: str = "whisper-1"
"""Default Whisper model used for transcription."""

OPENAI_BACKEND: str = "openai"
"""Name of the hosted OpenAI Whisper API backend."""

LOCAL_BACKEND: str = "local"
"""Name of the offline, locally hosted Whisper backend."""

AVAILABLE_BACKENDS: tuple[str, ...] = (OPENAI_BACKEND, LOCAL_BACKEND)
"""All ASR backend names accepted by :func:`build_transcriber`."""

Transcriber = Callable[[bytes], str]
"""A callable that turns WAV-encoded audio into text."""


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


def build_transcriber(
    backend: str,
    client: Optional[OpenAI] = None,
    language: Optional[str] = None,
    local_model: Optional[str] = None,
) -> Transcriber:
    """Build a transcription function for the chosen backend.

    Args:
        backend: Either ``"openai"`` or ``"local"``.
        client: An initialized OpenAI client. Required by the ``openai``
            backend and unused by ``local``.
        language: Optional ISO-639-1 language code passed to the recognizer.
        local_model: Hugging Face model id for the ``local`` backend. Defaults
            to :data:`talker.asr_local.DEFAULT_LOCAL_ASR_MODEL`.

    Raises:
        ValueError: If ``backend`` is unknown, or if the ``openai`` backend
            was selected without a client.

    Returns:
        A callable taking WAV-encoded audio bytes and returning the
        transcript.
    """
    if backend == OPENAI_BACKEND:
        if client is None:
            raise ValueError(
                "The 'openai' ASR backend requires an OpenAI client."
            )

        def transcribe_with_api(audio_bytes: bytes) -> str:
            """Transcribe audio using the hosted Whisper API."""
            return transcribe(audio_bytes, client, language=language)

        return transcribe_with_api

    if backend == LOCAL_BACKEND:
        # Imported lazily so the optional ML extras stay optional.
        from talker.asr_local import DEFAULT_LOCAL_ASR_MODEL, LocalWhisper

        recognizer = LocalWhisper(local_model or DEFAULT_LOCAL_ASR_MODEL)

        def transcribe_locally(audio_bytes: bytes) -> str:
            """Transcribe audio using the local Whisper model."""
            return recognizer.transcribe(audio_bytes, language=language)

        return transcribe_locally

    raise ValueError(
        f"Unknown ASR backend {backend!r}; "
        f"choose one of {', '.join(AVAILABLE_BACKENDS)}."
    )
