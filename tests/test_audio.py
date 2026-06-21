"""Unit tests for the audio module's pure helper functions."""

import io

import numpy as np
import soundfile as sf

from talker.audio import audio_to_wav_bytes, compute_rms


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
