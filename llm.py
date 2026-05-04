"""llm.py — LLM adapter for the chatbot.

Single interface so the rest of the system doesn't care which LLM backs it.

Backends available:
  - groq    (default for production - free tier, fast, no local install needed)
  - ollama  (original local backend - needs Ollama installed)
  - stub    (tests - no LLM needed)

Set the backend via the LLM_BACKEND environment variable.
Set the Groq API key via the GROQ_API_KEY environment variable.

--- Groq free tier limits (as of 2025) ---
  Requests per minute : 30
  Requests per day    : 14,400
  Tokens per minute   : 6,000 (varies by model)
  Cost                : $0

--- Recommended Groq models ---
  llama-3.3-70b-versatile   best quality, still fast       <- default
  llama-3.1-8b-instant      faster, slightly lower quality
  mixtral-8x7b-32768        good for technical Q&A

--- Ollama models (if switching back to local) ---
  llama3.2:3b   fast, ~4GB RAM
  llama3.1:8b   better quality, ~6GB RAM
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterator


# ── Environment config ────────────────────────────────────────────────────────
LLM_BACKEND   = os.getenv("LLM_BACKEND", "groq").lower()
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11434")


# ── Abstract base ─────────────────────────────────────────────────────────────
class LLM(ABC):
    @abstractmethod
    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Return a single completion string."""

    @abstractmethod
    def stream(self, system: str, user: str, temperature: float = 0.0) -> Iterator[str]:
        """Yield tokens as they arrive."""


# ── Groq backend ──────────────────────────────────────────────────────────────
class GroqLLM(LLM):
    """Calls the Groq hosted API. Free tier, very fast (1-2s responses).

    Requires:
        pip install groq
        GROQ_API_KEY environment variable set to your key (starts with gsk_...)

    Get a free key at: https://console.groq.com
    """

    def __init__(self, model: str = GROQ_MODEL, api_key: str = GROQ_API_KEY):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set.\n"
                "Get a free key at https://console.groq.com then set it:\n"
                "  Windows CMD:        set GROQ_API_KEY=gsk_...\n"
                "  Windows PowerShell: $env:GROQ_API_KEY='gsk_...'\n"
                "  Linux/Mac:          export GROQ_API_KEY=gsk_..."
            )
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError(
                "The 'groq' Python package is required. Install it with:\n"
                "  pip install groq"
            ) from e

        self.model = model
        self._client = Groq(api_key=api_key)

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""

    def stream(self, system: str, user: str, temperature: float = 0.0) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=1024,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── Ollama backend (original local backend, kept for reference) ───────────────
class OllamaLLM(LLM):
    """Calls a locally running Ollama instance.

    Requires Ollama installed and running:
        https://ollama.com
        ollama pull llama3.2:3b
        ollama serve
    """

    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.host = host
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "The 'ollama' Python package is required. Install it with:\n"
                "  pip install ollama\n"
                "Also install Ollama itself from https://ollama.com and run:\n"
                f"  ollama pull {model}"
            ) from e
        self._client = ollama.Client(host=host)

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        resp = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            options={"temperature": temperature},
        )
        return resp["message"]["content"]

    def stream(self, system: str, user: str, temperature: float = 0.0) -> Iterator[str]:
        for chunk in self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            options={"temperature": temperature},
            stream=True,
        ):
            yield chunk["message"]["content"]


# ── Stub backend (tests) ──────────────────────────────────────────────────────
class StubLLM(LLM):
    """Returns a canned response. Used in tests — no LLM needed."""

    def __init__(self, reply: str = "This is a stub response."):
        self.reply = reply

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.reply

    def stream(self, system: str, user: str, temperature: float = 0.0) -> Iterator[str]:
        yield self.reply


# ── Factory ───────────────────────────────────────────────────────────────────
def get_llm() -> LLM:
    """Return the right LLM based on the LLM_BACKEND environment variable.

    groq   → GroqLLM   (default, free, fast, requires GROQ_API_KEY)
    ollama → OllamaLLM (local, requires Ollama installed)
    stub   → StubLLM   (tests only)
    """
    if LLM_BACKEND == "stub":
        return StubLLM()
    if LLM_BACKEND == "ollama":
        return OllamaLLM()
    # Default: groq
    return GroqLLM()
