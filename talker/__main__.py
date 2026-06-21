"""Command-line entry point for the Talker spoken dialogue agent.

Currently wires the input half of the pipeline: it waits for a wake word,
records the following utterance, transcribes it, and prints the transcript.
LLM dialogue and text-to-speech responses are added in later stages.
"""

import argparse
import os
import sys
from typing import Optional, Sequence

from openai import OpenAI

from talker import asr, audio
from talker.wake_word import DEFAULT_MODEL, WakeWordDetector


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list to parse. If ``None``, ``sys.argv`` is used.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="talker",
        description="Spoken dialogue agent: wake word -> ASR -> LLM -> TTS.",
    )
    parser.add_argument(
        "--wake-word",
        default=DEFAULT_MODEL,
        help=f"openWakeWord model to listen for (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional ISO-639-1 language code for transcription (e.g. 'en').",
    )
    parser.add_argument(
        "--exit-phrase",
        default="goodbye",
        help="Spoken phrase that ends the conversation (default: 'goodbye').",
    )
    return parser.parse_args(argv)


def build_openai_client() -> OpenAI:
    """Construct an OpenAI client from the ``OPENAI_API_KEY`` env variable.

    Returns:
        An initialized OpenAI client.

    Raises:
        SystemExit: If the ``OPENAI_API_KEY`` environment variable is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key)


def run_input_loop(
    client: OpenAI,
    wake_word: str,
    language: Optional[str],
    exit_phrase: str,
) -> None:
    """Listen for the wake word, transcribe utterances, and print them.

    The loop ends when the transcript contains ``exit_phrase`` or on Ctrl+C.

    Args:
        client: An initialized OpenAI client for transcription.
        wake_word: The openWakeWord model name to listen for.
        language: Optional language code passed to the transcriber.
        exit_phrase: Phrase that, when heard, ends the loop.

    Returns:
        None.
    """
    print(
        f'[Talker] Listening for "{wake_word}" '
        f'(say "{exit_phrase}" to exit)'
    )
    with WakeWordDetector(model_name=wake_word) as detector:
        while True:
            detector.wait_for_wake_word()
            print("[Talker] Wake word detected, listening...")

            wav_bytes = audio.record_utterance()
            transcript = asr.transcribe(wav_bytes, client, language=language)

            if not transcript:
                print("[Talker] Heard nothing, going back to sleep.")
                continue

            print(f"[You] {transcript}")
            if exit_phrase.lower() in transcript.lower():
                print("[Talker] Goodbye!")
                return


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Program entry point.

    Args:
        argv: Optional argument list for testing; defaults to ``sys.argv``.

    Returns:
        None.
    """
    args = parse_args(argv)
    client = build_openai_client()
    try:
        run_input_loop(
            client=client,
            wake_word=args.wake_word,
            language=args.language,
            exit_phrase=args.exit_phrase,
        )
    except KeyboardInterrupt:
        print("\n[Talker] Interrupted. Goodbye!")


if __name__ == "__main__":
    main()
