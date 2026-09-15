"""Text-to-speech synthesis with interchangeable backends.

Two backends are available:

``openai``
    The OpenAI speech API. Needs an API key; returns WAV audio.
``gtts``
    Google Translate's TTS via the :mod:`gtts` package. Needs no API key;
    returns MP3 audio, which libsndfile decodes for playback.

Synthesis is kept separate from playback so the network call can be unit
tested without an audio device.
"""

import io
from typing import Optional

from openai import OpenAI

OPENAI_BACKEND: str = "openai"
"""Name of the OpenAI speech API backend."""

GTTS_BACKEND: str = "gtts"
"""Name of the gTTS (Google Translate) backend."""

AVAILABLE_BACKENDS: tuple[str, ...] = (OPENAI_BACKEND, GTTS_BACKEND)
"""All TTS backend names accepted by :func:`speak`."""

DEFAULT_TTS_MODEL: str = "tts-1"
"""Default OpenAI speech model."""

DEFAULT_VOICE: str = "alloy"
"""Default OpenAI voice name."""


def synthesize_openai(
    text: str,
    client: OpenAI,
    model: str = DEFAULT_TTS_MODEL,
    voice: str = DEFAULT_VOICE,
) -> bytes:
    """Synthesize speech with the OpenAI speech API.

    Args:
        text: The text to read aloud.
        client: An initialized OpenAI client.
        model: The speech model to use.
        voice: The voice name to speak with.

    Returns:
        The synthesized speech as WAV file contents. WAV is requested because
        :func:`talker.audio.play_audio` decodes it via libsndfile.
    """
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format="wav",
    )
    return response.read()


def synthesize_gtts(text: str, language: str = "en") -> bytes:
    """Synthesize speech with Google Translate's TTS service.

    Args:
        text: The text to read aloud.
        language: ISO-639-1 language code for the voice, e.g. ``"en"``.

    Returns:
        The synthesized speech as MP3 file contents.
    """
    # Imported lazily so the package is only required when this backend runs.
    from gtts import gTTS

    buffer = io.BytesIO()
    gTTS(text=text, lang=language).write_to_fp(buffer)
    return buffer.getvalue()


def speak(
    text: str,
    backend: str,
    client: Optional[OpenAI] = None,
    language: str = "en",
    voice: str = DEFAULT_VOICE,
) -> None:
    """Synthesize ``text`` with the chosen backend and play it aloud.

    Args:
        text: The text to read aloud. Empty text is ignored.
        backend: Either ``"openai"`` or ``"gtts"``.
        client: An initialized OpenAI client. Required for the ``openai``
            backend and unused by ``gtts``.
        language: ISO-639-1 language code, used by the ``gtts`` backend.
        voice: Voice name, used by the ``openai`` backend.

    Raises:
        ValueError: If ``backend`` is unknown, or if the ``openai`` backend
            was selected without a client.

    Returns:
        None. Side effect: audio is played on the default output device.
    """
    if not text:
        return

    # Imported here to keep module import free of audio-device side effects.
    from talker import audio

    if backend == OPENAI_BACKEND:
        if client is None:
            raise ValueError(
                "The 'openai' TTS backend requires an OpenAI client."
            )
        audio.play_audio(synthesize_openai(text, client, voice=voice))
    elif backend == GTTS_BACKEND:
        audio.play_audio(synthesize_gtts(text, language=language))
    else:
        raise ValueError(
            f"Unknown TTS backend {backend!r}; "
            f"choose one of {', '.join(AVAILABLE_BACKENDS)}."
        )
