"""Unit tests for the audio module's pure helper functions."""

import io
from unittest.mock import patch

import numpy as np
import sounddevice as sd
import soundfile as sf

from talker.audio import (
    audio_to_wav_bytes,
    compute_rms,
    resample,
    supported_output_rate,
)


def test_compute_rms_silence_is_zero() -> None:
    """A buffer of zeros has zero RMS amplitude."""
    samples = np.zeros(1000, dtype=np.float32)
    assert compute_rms(samples) == 0.0


def test_compute_rms_empty_buffer_is_zero() -> None:
    """An empty buffer is treated as silence rather than raising."""
    samples = np.zeros(0, dtype=np.float32)
    assert compute_rms(samples) == 0.0


def test_compute_rms_full_scale_is_one() -> None:
    """A constant full-scale signal has an RMS amplitude of 1.0."""
    samples = np.ones(1000, dtype=np.float32)
    assert compute_rms(samples) == 1.0


def test_audio_to_wav_bytes_roundtrip() -> None:
    """Encoding samples to WAV and reading them back preserves the signal."""
    sample_rate = 16_000
    original = np.sin(np.linspace(0, np.pi, 800)).astype(np.float32)

    wav_bytes = audio_to_wav_bytes(original, sample_rate)

    with io.BytesIO(wav_bytes) as buffer:
        decoded, decoded_rate = sf.read(buffer, dtype="float32")

    assert decoded_rate == sample_rate
    assert decoded.shape == original.shape
    # 16-bit PCM quantization introduces a small rounding error.
    assert np.allclose(decoded, original, atol=1e-3)


def test_resample_leaves_matching_rate_untouched() -> None:
    """Converting to the rate the audio already has is a no-op."""
    samples = np.sin(np.linspace(0, np.pi, 480)).astype(np.float32)

    assert resample(samples, 24_000, 24_000) is samples


def test_resample_scales_length_with_the_rate() -> None:
    """Doubling the sample rate roughly doubles the number of samples."""
    samples = np.sin(np.linspace(0, 4 * np.pi, 2_400)).astype(np.float32)

    converted = resample(samples, 24_000, 48_000)

    assert converted.shape[0] == 4_800


def test_resample_to_an_awkward_ratio() -> None:
    """A 24 kHz to 44.1 kHz conversion keeps the signal's duration."""
    samples = np.sin(np.linspace(0, 4 * np.pi, 24_000)).astype(np.float32)

    converted = resample(samples, 24_000, 44_100)

    # One second of audio in must stay one second of audio out.
    assert converted.shape[0] == 44_100


def test_supported_output_rate_keeps_a_playable_rate() -> None:
    """A rate the device accepts is returned unchanged."""
    with patch("talker.audio.sd.check_output_settings"):
        assert supported_output_rate(24_000, 1) == 24_000


def test_supported_output_rate_falls_back_to_device_default() -> None:
    """A rate the device rejects falls back to the device's own default."""
    error = sd.PortAudioError("Invalid sample rate")
    with patch("talker.audio.sd.check_output_settings", side_effect=error), \
            patch(
                "talker.audio.sd.query_devices",
                return_value={"default_samplerate": 44_100.0},
            ):
        assert supported_output_rate(24_000, 1) == 44_100
