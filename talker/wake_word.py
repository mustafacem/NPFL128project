"""Local wake-word detection using the openWakeWord library.

Detection runs entirely on the CPU with no network access, so the microphone
can be monitored continuously without per-request API cost or latency.
"""

from pathlib import Path
from types import TracebackType
from typing import Optional, Type

import numpy as np
import openwakeword
import sounddevice as sd
from openwakeword.model import Model
from openwakeword.utils import download_models

from talker.audio import resample, supported_input_rate

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


def ensure_model_downloaded(model_name: str = DEFAULT_MODEL) -> None:
    """Download the pre-trained wake-word model if it is not installed yet.

    openWakeWord ships without model weights, so a fresh install has nothing
    to load. The download runs once and is skipped on later starts.

    Args:
        model_name: Name of the pre-trained openWakeWord model, without the
            version suffix used in the released file names.

    Returns:
        None. Side effect: model files are written into openWakeWord's
        resource directory on first use.
    """
    model_directory = Path(openwakeword.__file__).parent / "resources/models"
    if any(model_directory.glob(f"{model_name}*.onnx")):
        return

    print(f"[Talker] Downloading wake-word model '{model_name}'...")
    # The release assets carry version suffixes, so the whole set is fetched
    # and openWakeWord picks the matching file by name.
    download_models()


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
            inference_framework: Backend, ``"onnx"`` or ``"tflite"``.
        """
        self._model_name = model_name
        self._threshold = threshold
        ensure_model_downloaded(model_name)
        self._model = Model(
            wakeword_models=[model_name],
            inference_framework=inference_framework,
        )
        self._stream: Optional[sd.InputStream] = None
        # The microphone may not offer 16 kHz, so record at a rate it does
        # support and convert each block before scoring it.
        self._device_rate = supported_input_rate(_SAMPLE_RATE, 1)
        self._device_frames = (
            _FRAME_SAMPLES * self._device_rate // _SAMPLE_RATE
        )

    def __enter__(self) -> "WakeWordDetector":
        """Open the microphone input stream."""
        self._stream = sd.InputStream(
            samplerate=self._device_rate,
            channels=1,
            blocksize=self._device_frames,
            dtype="float32",
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
            block, _overflowed = self._stream.read(self._device_frames)
            if self.detect_in_frame(self._to_model_frame(block[:, 0])):
                return

    def _to_model_frame(self, block: np.ndarray) -> np.ndarray:
        """Convert one recorded block into a frame the model can score.

        Args:
            block: Mono float32 samples captured at the device's own rate.

        Returns:
            Exactly :data:`_FRAME_SAMPLES` int16 samples at 16 kHz.
        """
        converted = resample(block, self._device_rate, _SAMPLE_RATE)
        scaled = np.clip(converted, -1.0, 1.0) * 32767
        return np.asarray(scaled, dtype=np.int16)

    def detect_in_frame(self, samples: np.ndarray) -> bool:
        """Score one audio frame and report whether the wake word fired.

        Kept separate from the recording loop so detection can be exercised
        on pre-recorded audio without a microphone.

        Args:
            samples: One frame of 16 kHz mono audio as int16 samples,
                :data:`_FRAME_SAMPLES` long.

        Returns:
            True if any loaded wake word scored at or above the configured
            threshold, False otherwise.
        """
        scores = self._model.predict(samples)
        return any(score >= self._threshold for score in scores.values())

    def close(self) -> None:
        """Stop and close the microphone input stream if it is open.

        Returns:
            None. Side effect: releases the audio input device.
        """
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
