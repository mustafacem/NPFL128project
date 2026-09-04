"""Unit tests for wake-word detection, with the model and files mocked."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from talker.wake_word import WakeWordDetector, ensure_model_downloaded


def _make_detector(scores: dict) -> WakeWordDetector:
    """Build a detector whose model always reports ``scores``.

    Args:
        scores: Model output mapping wake-word names to confidences.

    Returns:
        A WakeWordDetector wired to a mocked model, with no files downloaded.
    """
    model = MagicMock()
    model.predict.return_value = scores
    with patch("talker.wake_word.Model", return_value=model), \
            patch("talker.wake_word.ensure_model_downloaded"):
        return WakeWordDetector()


def test_detect_in_frame_fires_above_threshold() -> None:
    """A confidence at or above the threshold counts as a detection."""
    detector = _make_detector({"hey_jarvis": 0.9})

    assert detector.detect_in_frame(np.zeros(1280, dtype=np.int16))


def test_detect_in_frame_ignores_low_confidence() -> None:
    """Confidences below the threshold are not detections."""
    detector = _make_detector({"hey_jarvis": 0.1})

    assert not detector.detect_in_frame(np.zeros(1280, dtype=np.int16))


def test_detect_in_frame_fires_on_any_loaded_model() -> None:
    """Any loaded wake word exceeding the threshold triggers detection."""
    detector = _make_detector({"hey_jarvis": 0.1, "alexa": 0.8})

    assert detector.detect_in_frame(np.zeros(1280, dtype=np.int16))


def test_wait_for_wake_word_requires_open_stream() -> None:
    """Listening outside the context manager is a clear error."""
    detector = _make_detector({"hey_jarvis": 0.9})

    with pytest.raises(RuntimeError, match="Input stream is not open"):
        detector.wait_for_wake_word()


def test_ensure_model_downloaded_skips_existing_model() -> None:
    """An already-installed model is not downloaded again."""
    with patch("talker.wake_word.Path") as path_type, \
            patch("talker.wake_word.download_models") as download:
        model_dir = path_type.return_value.parent.__truediv__.return_value
        model_dir.glob.return_value = iter(["hey_jarvis_v0.1.onnx"])
        ensure_model_downloaded("hey_jarvis")

    download.assert_not_called()


def test_ensure_model_downloaded_fetches_missing_model() -> None:
    """A missing model is downloaded before the detector loads it."""
    with patch("talker.wake_word.Path") as path_type, \
            patch("talker.wake_word.download_models") as download:
        model_dir = path_type.return_value.parent.__truediv__.return_value
        model_dir.glob.return_value = iter([])
        ensure_model_downloaded("hey_jarvis")

    download.assert_called_once()
