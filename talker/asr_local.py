"""Offline speech recognition with a locally hosted Whisper model.

This backend needs no API key and sends no audio off the machine, at the cost
of downloading model weights and running inference locally. Its heavy machine
learning dependencies are optional extras (``pip install -e ".[local]"``) and
are imported only when a model is actually loaded.
"""

import io
from typing import Any, Optional

DEFAULT_LOCAL_ASR_MODEL: str = "openai/whisper-small"
"""Default Hugging Face model id used for local transcription."""

WHISPER_SAMPLE_RATE: int = 16_000
"""Sample rate Whisper models expect, in Hz."""


class LocalWhisper:
    """A lazily loaded local Whisper model for offline transcription."""

    def __init__(self, model_id: str = DEFAULT_LOCAL_ASR_MODEL) -> None:
        """Record which model to use without loading it yet.

        Loading is deferred to the first :meth:`transcribe` call so that
        constructing this object stays cheap and import-time free of side
        effects.

        Args:
            model_id: Hugging Face model id of a Whisper checkpoint.
        """
        self._model_id = model_id
        self._pipeline: Optional[Any] = None

    def _load_pipeline(self) -> Any:
        """Build the speech recognition pipeline, reusing it once loaded.

        Returns:
            A Hugging Face ``automatic-speech-recognition`` pipeline placed on
            the GPU when one is available, otherwise on the CPU.

        Raises:
            ImportError: If the optional local-inference extras are missing.
        """
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch
            from transformers import pipeline
        except ImportError as error:
            raise ImportError(
                "The local ASR backend needs extra packages. Install them "
                'with: pip install -e ".[local]"'
            ) from error

        use_cuda = torch.cuda.is_available()
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self._model_id,
            dtype=torch.float16 if use_cuda else torch.float32,
            device=0 if use_cuda else -1,
        )
        return self._pipeline

    def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe encoded audio to text with the local model.

        Args:
            audio_bytes: WAV-encoded audio data to transcribe.
            language: Optional ISO-639-1 language code (e.g. ``"en"``). If
                ``None``, Whisper detects the language itself.

        Returns:
            The transcribed text with surrounding whitespace stripped.
        """
        # Imported here so the module imports without soundfile installed.
        import soundfile as sf

        from talker.audio import resample

        with io.BytesIO(audio_bytes) as buffer:
            samples, sample_rate = sf.read(buffer, dtype="float32")

        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        # Convert here rather than letting the pipeline do it, which would
        # pull in torchaudio purely to resample.
        samples = resample(samples, int(sample_rate), WHISPER_SAMPLE_RATE)

        generate_kwargs = {"task": "transcribe"}
        if language is not None:
            generate_kwargs["language"] = language

        recognizer = self._load_pipeline()
        result = recognizer(
            {"raw": samples, "sampling_rate": WHISPER_SAMPLE_RATE},
            generate_kwargs=generate_kwargs,
        )
        return str(result["text"]).strip()
