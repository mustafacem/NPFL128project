"""Local wake-word detection using the openWakeWord library.

Detection runs entirely on the CPU with no network access, so the microphone
can be monitored continuously without per-request API cost or latency.
"""

from types import TracebackType
from typing import Optional, Type

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

DEFAULT_MODEL: str = "hey_jarvis"
"""Name of the pre-trained openWakeWord model used by default."""

DEFAULT_FRAMEWORK: str = "onnx"
"""Inference backend for openWakeWord. ONNX is used because tflite-runtime has
no wheel for recent Python versions on Linux."""

_SAMPLE_RATE: int = 16_000
"""Sample rate required by openWakeWord models, in Hz."""

_FRAME_SAMPLES: int = 1_280
"""Number of samples per inference frame (80 ms at 16 kHz)."""

_DEFAULT_THRESHOLD: float = 0.5
"""Confidence score above which a wake word is considered detected."""


class WakeWordDetector:
    """Detects a spoken wake word on the default microphone.

    Intended to be used as a context manager so the input stream is always
    released::

        with WakeWordDetector() as detector:
            detector.wait_for_wake_word()
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = _DEFAULT_THRESHOLD,
        inference_framework: str = DEFAULT_FRAMEWORK,
    ) -> None:
        """Load the wake-word model and prepare (but do not open) the stream.

        Args:
            model_name: Name of the pre-trained openWakeWord model to load.
            threshold: Confidence score in [0, 1] above which detection fires.
            inference_framework: openWakeWord backend, ``"onnx"`` or ``"tflite"``.
        """
        self._model_name = model_name
        self._threshold = threshold
        self._model = Model(
            wakeword_models=[model_name],
            inference_framework=inference_framework,
        )
        self._stream: Optional[sd.InputStream] = None

    def __enter__(self) -> "WakeWordDetector":
        """Open the microphone input stream."""
        self._stream = sd.InputStream(
            samplerate=_SAMPLE_RATE,
            channels=1,
            blocksize=_FRAME_SAMPLES,
            dtype="int16",
        )
        self._stream.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Close the microphone input stream."""
        self.close()

    def wait_for_wake_word(self) -> None:
        """Block until the configured wake word is detected.

        Reads audio frames from the open input stream and feeds them to the
        model until any loaded wake word exceeds the confidence threshold.

        Raises:
            RuntimeError: If called outside an open input stream context.

        Returns:
            None. Returns as soon as the wake word is detected.
        """
        if self._stream is None:
            raise RuntimeError(
                "Input stream is not open. Use WakeWordDetector as a context "
                "manager before calling wait_for_wake_word()."
            )

        while True:
            frame, _overflowed = self._stream.read(_FRAME_SAMPLES)
            samples = np.frombuffer(frame, dtype=np.int16)
            scores = self._model.predict(samples)
            if any(score >= self._threshold for score in scores.values()):
                return

    def close(self) -> None:
        """Stop and close the microphone input stream if it is open.

        Returns:
            None. Side effect: releases the audio input device.
        """
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
