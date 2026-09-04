# Talker — Spoken Dialogue Agent

A voice-driven conversational agent that combines automatic speech recognition (ASR),
large language model (LLM) dialogue management, and text-to-speech (TTS) synthesis
into a single end-to-end pipeline.

## Background

This project implements a spoken dialogue system inspired by the architecture described in:

> Yang, C.-H. H., Stolcke, A., & Heck, L. (2024). *Spoken Conversational Agents with
> Large Language Models*. arXiv:2512.02593. <https://arxiv.org/abs/2512.02593>

The paper provides a tutorial-style treatment of integrating LLMs into spoken dialogue
systems, covering speech input processing, dialogue management, and response generation.
Talker follows that cascaded design: each spoken turn passes through a separate
recognition, dialogue, and synthesis stage rather than a single end-to-end speech model.

## Scope

### Implemented

- Local wake-word detection via [openWakeWord](https://github.com/dscripka/openWakeWord)
  (no API key, runs on CPU)
- Microphone recording with automatic end-of-speech (silence) detection
- Speech-to-text with two interchangeable backends: the OpenAI Whisper API, or a
  Whisper checkpoint running locally for fully offline recognition
- Multi-turn dialogue management: the whole conversation is sent with every request,
  so the agent can resolve context such as "how many people live *there*?"
- LLM response generation via OpenRouter (configurable model)
- Text-to-speech with two interchangeable backends: the OpenAI speech API, or gTTS
- Configurable system prompt (agent persona / instructions)
- Conversation history saved to JSON when the session ends
- Graceful exit on a spoken farewell phrase or Ctrl+C
- Text mode: a keyboard-driven conversation for machines with no audio hardware
- Unit tests for the dialogue, ASR, LLM, TTS, audio, and command-line modules

### Left for Future Work

- Streaming ASR and TTS, so the agent can reply before the user stops speaking
- Barge-in: letting the user interrupt a reply that is already being spoken
- Retrieval-augmented generation (RAG) for domain knowledge
- Emotion / sentiment detection from speech prosody
- Custom wake words trained for this agent rather than a stock model

## Requirements

- Python 3.11+
- An [OpenRouter API key](https://openrouter.ai/) — used for LLM dialogue (always required)
- An [OpenAI API key](https://platform.openai.com/) — only if you use the hosted
  Whisper (`--stt openai`) or speech (`--tts openai`) backends, which are the defaults

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

# Install Talker and its dependencies.
pip install -e .

# openWakeWord declares a hard dependency on tflite-runtime, which has no
# wheel for recent Python versions on Linux. Talker uses the ONNX backend
# instead, so openWakeWord is installed without its own dependencies:
pip install --no-deps openwakeword==0.6.0
```

The offline backends are optional extras, installed only if you want them:

```bash
pip install -e ".[gtts]"   # gTTS speech synthesis  (--tts gtts)
pip install -e ".[local]"  # local Whisper model    (--stt local)
```

Set the API keys you need:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENAI_API_KEY="sk-..."
```

## How to Run

```bash
python -m talker
```

Say the wake word (default: **"hey jarvis"**), then speak. The agent records
until you pause, transcribes your speech, replies, and reads the reply aloud.
Say **"goodbye"** to exit. The conversation is written to `history.json`.
The wake-word model is downloaded automatically the first time you run it.

The default model, like most on OpenRouter, needs credits on your account. To
try the agent without them, pick one of OpenRouter's free models:

```bash
python -m talker --llm-model "liquid/lfm-2.5-2.6b:free"
```

To run without a microphone or speakers, type instead of speaking:

```bash
python -m talker --text-mode
```

To run without sending audio to a hosted service, use the offline backends:

```bash
python -m talker --stt local --tts gtts
```

### Options

```
usage: talker [-h] [--wake-word WAKE_WORD] [--language LANGUAGE]
              [--exit-phrase EXIT_PHRASE] [--stt {openai,local}]
              [--local-asr-model LOCAL_ASR_MODEL] [--tts {openai,gtts}]
              [--voice VOICE] [--llm-model LLM_MODEL]
              [--system-prompt-file SYSTEM_PROMPT_FILE]
              [--history-out HISTORY_OUT] [--text-mode] [--speak]

options:
  -h, --help            show this help message and exit
  --wake-word           openWakeWord model to listen for (default: hey_jarvis)
  --language            ISO-639-1 language code for speech (e.g. 'en')
  --exit-phrase         spoken phrase that ends the conversation (default: 'goodbye')
  --stt                 speech recognition backend (default: openai)
  --local-asr-model     Hugging Face model id used by the local ASR backend
  --tts                 speech synthesis backend (default: openai)
  --voice               voice name for the openai TTS backend (default: alloy)
  --llm-model           OpenRouter model used for replies (default: qwen/qwen3-8b)
  --system-prompt-file  file holding the system prompt that sets the persona
  --history-out         where to save the conversation (default: history.json)
  --text-mode           type instead of speaking; needs no microphone
  --speak               in text mode, also read the replies aloud
```

## Running the Tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite mocks every network call and audio device, so it needs no API
keys and no microphone. The project is also checked with `mypy talker/` and
`pycodestyle --max-line-length=79 talker/ tests/`.

## Sample Output

See [sample_output.txt](sample_output.txt) for full sessions in both text and
voice mode, together with the JSON transcript they produce.
