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

> **Work in progress.** No functionality has been implemented yet.
> This repository contains the initial project scaffold only.

## Scope

### Planned

- Microphone audio capture and playback
- Speech-to-text via the OpenAI Whisper API
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
- An [OpenRouter API key](https://openrouter.ai/) — used for LLM dialogue

## How to Run

> Not yet available. Will be updated once the implementation is complete.

## Sample Output

> Not yet available.
