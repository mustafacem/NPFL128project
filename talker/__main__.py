"""Command-line entry point for the Talker spoken dialogue agent.

Wires the full pipeline: a wake word starts recording, the utterance is
transcribed, an LLM produces a reply, and the reply is spoken back. The
conversation is written to a JSON file when the session ends.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from openai import APIError, OpenAI

from talker import asr, audio, dialogue, llm, tts
from talker.dialogue import DialogueManager
from talker.wake_word import DEFAULT_MODEL, WakeWordDetector

DEFAULT_HISTORY_PATH: Path = Path("history.json")
"""Default file the conversation transcript is written to on exit."""


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
        help="Optional ISO-639-1 language code for speech (e.g. 'en').",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=audio.DEFAULT_SILENCE_THRESHOLD,
        help=(
            "loudness below which the microphone counts as silent, ending "
            "a recording (default: "
            f"{audio.DEFAULT_SILENCE_THRESHOLD}). Raise it in a noisy room "
            "if recordings never stop; lower it if they cut you off."
        ),
    )
    parser.add_argument(
        "--exit-phrase",
        default="goodbye",
        help="Spoken phrase that ends the conversation (default: 'goodbye').",
    )
    parser.add_argument(
        "--stt",
        choices=asr.AVAILABLE_BACKENDS,
        default=asr.OPENAI_BACKEND,
        help="Speech recognition backend (default: %(default)s).",
    )
    parser.add_argument(
        "--local-asr-model",
        default=None,
        help="Hugging Face model id used by the local ASR backend.",
    )
    parser.add_argument(
        "--tts",
        choices=tts.AVAILABLE_BACKENDS,
        default=tts.OPENAI_BACKEND,
        help="Speech synthesis backend (default: %(default)s).",
    )
    parser.add_argument(
        "--voice",
        default=tts.DEFAULT_VOICE,
        help="Voice name for the openai TTS backend (default: %(default)s).",
    )
    parser.add_argument(
        "--llm-model",
        default=llm.DEFAULT_LLM_MODEL,
        help="OpenRouter model used for replies (default: %(default)s).",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="File holding the system prompt that sets the agent's persona.",
    )
    parser.add_argument(
        "--history-out",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="Where to save the conversation (default: %(default)s).",
    )
    parser.add_argument(
        "--text-mode",
        action="store_true",
        help="Type instead of speaking; needs no microphone or speakers.",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="In text mode, also read the replies aloud.",
    )
    return parser.parse_args(argv)


def require_env(name: str) -> str:
    """Read a required environment variable or exit with an error.

    Args:
        name: Name of the environment variable to read.

    Returns:
        The variable's value.

    Raises:
        SystemExit: If the variable is unset or empty.
    """
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Error: {name} environment variable is not set.")
    return value


def load_system_prompt(path: Optional[Path]) -> str:
    """Load the agent's system prompt from a file, or use the default.

    Args:
        path: File containing the system prompt, or ``None`` for the default.

    Returns:
        The system prompt text.

    Raises:
        SystemExit: If ``path`` is given but cannot be read.
    """
    if path is None:
        return dialogue.DEFAULT_SYSTEM_PROMPT
    try:
        with path.open("r", encoding="utf-8") as prompt_file:
            return prompt_file.read().strip()
    except OSError as error:
        sys.exit(f"Error: cannot read system prompt file {path}: {error}")


def exchange(
    manager: DialogueManager,
    user_text: str,
    client: OpenAI,
    model: str,
) -> str:
    """Record a user turn, generate a reply, and record the reply.

    Both turns are recorded only once the reply arrives, so a failed request
    does not leave an unanswered user turn in the history.

    Args:
        manager: The conversation whose history is extended in place.
        user_text: What the user just said.
        client: An OpenRouter-backed client used to generate the reply.
        model: The OpenRouter model identifier to query.

    Returns:
        The assistant's reply text.

    Raises:
        openai.APIError: If the model could not be reached.
    """
    pending = manager.history + [{"role": "user", "content": user_text}]
    reply = llm.generate_reply(pending, client, model=model)
    manager.add_user_turn(user_text)
    manager.add_assistant_turn(reply)
    return reply


def reply_to(
    manager: DialogueManager,
    user_text: str,
    client: OpenAI,
    model: str,
) -> Optional[str]:
    """Generate a reply, reporting API failures instead of crashing.

    Args:
        manager: The conversation whose history is extended in place.
        user_text: What the user just said.
        client: An OpenRouter-backed client used to generate the reply.
        model: The OpenRouter model identifier to query.

    Returns:
        The assistant's reply, or ``None`` if the model could not be reached.
    """
    try:
        return exchange(manager, user_text, client, model)
    except APIError as error:
        detail = getattr(error, "message", None) or str(error)
        print(
            f"[Talker] Sorry, I could not reach the language model. {detail}"
        )
        return None


def run_voice_loop(
    manager: DialogueManager,
    args: argparse.Namespace,
    transcribe: asr.Transcriber,
    llm_client: OpenAI,
    openai_client: Optional[OpenAI],
) -> None:
    """Hold a spoken conversation until the user says the exit phrase.

    Args:
        manager: The conversation to extend with each turn.
        args: Parsed command-line arguments.
        transcribe: Callable turning recorded audio into text.
        llm_client: An OpenRouter-backed client for generating replies.
        openai_client: An OpenAI client, needed by the openai TTS backend.

    Returns:
        None.
    """
    print(
        f'[Talker] Listening for "{args.wake_word}" '
        f'(say "{args.exit_phrase}" to exit)'
    )
    with WakeWordDetector(model_name=args.wake_word) as detector:
        while True:
            detector.wait_for_wake_word()
            print("[Talker] Wake word detected, listening...")

            transcript = transcribe(
                audio.record_utterance(
                    silence_threshold=args.silence_threshold
                )
            )
            if not transcript:
                print("[Talker] Heard nothing, going back to sleep.")
                continue

            print(f"[You] {transcript}")
            if dialogue.is_farewell(transcript, (args.exit_phrase,)):
                farewell = "Goodbye!"
                print(f"[Talker] {farewell}")
                speak_reply(farewell, args, openai_client)
                return

            reply = reply_to(
                manager, transcript, llm_client, args.llm_model
            )
            if reply is None:
                continue
            print(f"[Talker] {reply}")
            speak_reply(reply, args, openai_client)


def run_text_loop(
    manager: DialogueManager,
    args: argparse.Namespace,
    llm_client: OpenAI,
    openai_client: Optional[OpenAI],
) -> None:
    """Hold a typed conversation, for use without audio hardware.

    Args:
        manager: The conversation to extend with each turn.
        args: Parsed command-line arguments.
        llm_client: An OpenRouter-backed client for generating replies.
        openai_client: An OpenAI client, needed by the openai TTS backend.

    Returns:
        None. Ends on the exit phrase or end of input.
    """
    print(f'[Talker] Text mode. Type "{args.exit_phrase}" to exit.')
    while True:
        try:
            user_text = input("[You] ").strip()
        except EOFError:
            print()
            return

        if not user_text:
            continue
        if dialogue.is_farewell(user_text, (args.exit_phrase,)):
            print("[Talker] Goodbye!")
            return

        reply = reply_to(manager, user_text, llm_client, args.llm_model)
        if reply is None:
            continue
        print(f"[Talker] {reply}")
        if args.speak:
            speak_reply(reply, args, openai_client)


def speak_reply(
    text: str,
    args: argparse.Namespace,
    openai_client: Optional[OpenAI],
) -> None:
    """Read a reply aloud with the configured TTS backend.

    Args:
        text: The reply to speak.
        args: Parsed command-line arguments.
        openai_client: An OpenAI client, needed by the openai TTS backend.

    Returns:
        None. Side effect: audio is played on the default output device.
    """
    tts.speak(
        text,
        backend=args.tts,
        client=openai_client,
        language=args.language or "en",
        voice=args.voice,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Program entry point.

    Args:
        argv: Optional argument list for testing; defaults to ``sys.argv``.

    Returns:
        None.
    """
    args = parse_args(argv)

    # Text mode never transcribes, and only speaks when asked to, so it can
    # run without an OpenAI key even while the hosted backends are selected.
    transcribing = not args.text_mode
    speaking = args.speak or not args.text_mode
    needs_openai = (transcribing and args.stt == asr.OPENAI_BACKEND) or (
        speaking and args.tts == tts.OPENAI_BACKEND
    )
    openai_client = (
        OpenAI(api_key=require_env("OPENAI_API_KEY")) if needs_openai else None
    )
    llm_client = llm.build_client(require_env("OPENROUTER_API_KEY"))

    manager = DialogueManager(load_system_prompt(args.system_prompt_file))
    try:
        if args.text_mode:
            run_text_loop(manager, args, llm_client, openai_client)
        else:
            transcribe = asr.build_transcriber(
                args.stt,
                client=openai_client,
                language=args.language,
                local_model=args.local_asr_model,
            )
            run_voice_loop(
                manager, args, transcribe, llm_client, openai_client
            )
    except KeyboardInterrupt:
        print("\n[Talker] Interrupted. Goodbye!")
    finally:
        manager.save_json(args.history_out)
        print(f"[Talker] Conversation saved to {args.history_out}")


if __name__ == "__main__":
    main()
