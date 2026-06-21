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

## Status

> **Work in progress.** The audio input pipeline (wake word → recording → ASR)
> is implemented. LLM dialogue and text-to-speech are not yet built.

## Scope

### Implemented

- Microphone recording with automatic end-of-speech (silence) detection
- Local wake-word detection via [openWakeWord](https://github.com/dscripka/openWakeWord)
  (no API key, runs on CPU)
- Speech-to-text via the OpenAI Whisper API
- Command-line entry point that listens, records, transcribes, and prints

### Planned

- Multi-turn dialogue management with full conversation history
- LLM response generation via OpenRouter (configurable model)
- Text-to-speech synthesis via the OpenAI TTS API
- Configurable system prompt (agent persona / instructions)
- Conversation history saved to JSON on exit
- Graceful exit on farewell keyword or Ctrl+C

### Left for Future Work

- Wake-word detection (always-on listening)
- Streaming TTS for lower latency
- Retrieval-augmented generation (RAG) for domain knowledge
- Emotion / sentiment detection from speech prosody
- Multi-language support

## Requirements

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/) — used for Whisper (ASR) and TTS
- An [OpenRouter API key](https://openrouter.ai/) — used for LLM dialogue (planned)

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

Set the API key used for transcription:

```bash
export OPENAI_API_KEY="sk-..."
```

## How to Run

```bash
python -m talker
```

Say the wake word (default: **"hey jarvis"**), then speak. The agent records
until you pause, transcribes your speech, and prints the transcript. Say
**"goodbye"** to exit.

### Options

```
usage: talker [-h] [--wake-word WAKE_WORD] [--language LANGUAGE]
              [--exit-phrase EXIT_PHRASE]

options:
  -h, --help            show this help message and exit
  --wake-word           openWakeWord model name to listen for (default: hey_jarvis)
  --language            ISO-639-1 language code for transcription (e.g. 'en')
  --exit-phrase         spoken phrase that ends the conversation (default: 'goodbye')
```

## Running the Tests

```bash
pip install -e ".[dev]"
pytest
```

## Sample Output

```
[Talker] Listening for "hey_jarvis" (say "goodbye" to exit)
[Talker] Wake word detected, listening...
[You] What is the capital of France?
[Talker] Wake word detected, listening...
[You] Goodbye.
[Talker] Goodbye!
```
