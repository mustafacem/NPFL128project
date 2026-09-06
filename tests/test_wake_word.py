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


def test_reading_without_an_open_stream_is_an_error() -> None:
    """Reading audio before the stream is opened fails loudly."""
    detector = _make_detector({"hey_jarvis": 0.9})

    with pytest.raises(RuntimeError, match="Input stream is not open"):
        detector._read_block()


def test_microphone_is_released_after_detection() -> None:
    """The device is freed once the wake word fires.

    The recorder opens the microphone for the utterance that follows, and
    sound cards commonly allow only one capture stream at a time.
    """
    detector = _make_detector({"hey_jarvis": 0.9})
    stream = MagicMock()
    stream.read.return_value = (np.zeros((3528, 1), dtype=np.float32), False)

    with patch("talker.wake_word.sd.InputStream", return_value=stream):
        detector.wait_for_wake_word()

    stream.stop.assert_called_once()
    stream.close.assert_called_once()


def test_listening_clears_buffered_audio() -> None:
    """Stale audio is dropped so the last utterance cannot re-trigger."""
    detector = _make_detector({"hey_jarvis": 0.9})
    stream = MagicMock()
    stream.read.return_value = (np.zeros((3528, 1), dtype=np.float32), False)

    with patch("talker.wake_word.sd.InputStream", return_value=stream):
        detector.wait_for_wake_word()

    detector._model.reset.assert_called_once()


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
