"""The only module that talks to a language model.

Two backends behind one :func:`complete` call: Gemini through
google-generativeai (the default, free tier) and a local model through ollama
(the offline fallback, so the demo survives a dead API key or no network). Both
go through the same disk cache in `data/processed`, keyed on the prompt and
model, which makes repeated notebook runs and test runs free and deterministic.

Also owns retries, timeouts and JSON-mode requests. Callers get text back and
never see which backend served them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class Completion:
    """One model response, whatever the backend."""

    text: str
    model: str
    backend: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    from_cache: bool


class LLMError(RuntimeError):
    """Raised when every backend and retry has failed."""


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def cache_key(prompt: str, system: str | None, model: str, temperature: float) -> str:
    """Return the stable hash under which a response is cached."""
    ...


def cache_get(key: str) -> Completion | None:
    """Look a response up in the on-disk cache."""
    ...


def cache_put(key: str, completion: Completion) -> None:
    """Write a response to the on-disk cache."""
    ...


def clear_cache() -> int:
    """Empty the LLM cache and return how many entries were dropped."""
    ...


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

def gemini_client(api_key: str | None = None, model: str | None = None) -> Any:
    """Build and cache the Gemini client."""
    ...


def complete_gemini(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.2,
    json_mode: bool = False,
) -> Completion:
    """Send one prompt to Gemini."""
    ...


def complete_ollama(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.2,
    json_mode: bool = False,
) -> Completion:
    """Send one prompt to the local Ollama model."""
    ...


def ollama_available(model: str | None = None) -> bool:
    """Return whether an Ollama server is up and the model is pulled."""
    ...


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def complete(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.2,
    json_mode: bool = False,
    backend: str | None = None,
    use_cache: bool = True,
) -> Completion:
    """Complete a prompt on the configured backend, falling back if it fails."""
    ...


def complete_json(
    prompt: str,
    system: str | None = None,
    schema: dict[str, Any] | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Complete a prompt and return the parsed JSON object."""
    ...


def stream(
    prompt: str,
    system: str | None = None,
    backend: str | None = None,
) -> Iterator[str]:
    """Yield response chunks as they arrive, for the demo UI."""
    ...


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed texts with the provider's embedding model, cache included."""
    ...
