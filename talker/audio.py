"""Low-level audio I/O: microphone recording with silence detection,
WAV encoding, and audio playback.

All functions operate on in-memory data so they remain testable and free of
hidden file side effects.
"""

import io
import math
import queue
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE: int = 16_000
"""Default sample rate in Hz. Matches the rate expected by the Whisper API."""

CHANNELS: int = 1
"""Default number of audio channels (mono)."""

_CHUNK_DURATION: float = 0.1
"""Length in seconds of each audio chunk read while recording."""


def compute_rms(samples: np.ndarray) -> float:
    """Compute the root-mean-square amplitude of an audio buffer.

    Args:
        samples: A 1-D array of float32 audio samples in the range [-1, 1].

    Returns:
        The RMS amplitude as a non-negative float. Returns 0.0 for an empty
        buffer.
    """
    if samples.size == 0:
        return 0.0
    return float(math.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def audio_to_wav_bytes(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Encode float audio samples as in-memory WAV bytes.

    Args:
        samples: A 1-D array of float32 audio samples in the range [-1, 1].
        sample_rate: The sample rate of the audio in Hz.

    Returns:
        The audio encoded as 16-bit PCM WAV file contents.
    """
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def record_utterance(
    silence_threshold: float = 0.01,
    silence_duration: float = 1.5,
    max_duration: float = 30.0,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Record from the default microphone until the speaker falls silent.

    Recording stops after ``silence_duration`` seconds of consecutive silence
    (RMS amplitude below ``silence_threshold``) or once ``max_duration`` is
    reached, whichever comes first.

    Args:
        silence_threshold: RMS amplitude below which a chunk counts as silent.
        silence_duration: Seconds of consecutive silence that end recording.
        max_duration: Hard cap on total recording length in seconds.
        sample_rate: The sample rate to record at, in Hz.

    Returns:
        The recorded audio encoded as 16-bit PCM WAV bytes. Returns an empty
        WAV file if nothing was captured.
    """
    chunk_frames = int(sample_rate * _CHUNK_DURATION)
    silent_chunks_needed = int(silence_duration / _CHUNK_DURATION)
    max_chunks = int(max_duration / _CHUNK_DURATION)

    chunk_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def _callback(
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Hand one recorded chunk to the consumer loop.

        Called by sounddevice on its own thread for every captured block.

        Args:
            indata: The captured block, shaped (frames, channels).
            frames: Number of frames in this block.
            time_info: Stream timestamps supplied by PortAudio (opaque).
            status: Flags reporting input overflows or underflows.

        Returns:
            None. Side effect: the block is appended to ``chunk_queue``.
        """
        # Copy because sounddevice reuses the input buffer between callbacks.
        chunk_queue.put(indata[:, 0].copy())

    recorded: list[np.ndarray] = []
    consecutive_silent = 0
    captured_speech = False

    with sd.InputStream(
        samplerate=sample_rate,
        channels=CHANNELS,
        blocksize=chunk_frames,
        dtype="float32",
        callback=_callback,
    ):
        for _ in range(max_chunks):
            chunk = chunk_queue.get()
            recorded.append(chunk)

            if compute_rms(chunk) < silence_threshold:
                consecutive_silent += 1
                if (
                    captured_speech
                    and consecutive_silent >= silent_chunks_needed
                ):
                    break
            else:
                captured_speech = True
                consecutive_silent = 0

    if not recorded:
        return audio_to_wav_bytes(np.zeros(0, dtype=np.float32), sample_rate)

    samples = np.concatenate(recorded)
    return audio_to_wav_bytes(samples, sample_rate)


def play_audio(audio_bytes: bytes) -> None:
    """Play encoded audio (e.g. WAV or MP3) through the default output device.

    This function blocks until playback finishes.

    Args:
        audio_bytes: Encoded audio file contents in any format supported by
            libsndfile (WAV, FLAC, OGG) read via :mod:`soundfile`.

    Returns:
        None. Side effect: audio is played on the default output device.
    """
    with io.BytesIO(audio_bytes) as buffer:
        samples, sample_rate = sf.read(buffer, dtype="float32")
    sd.play(samples, sample_rate)
    sd.wait()
