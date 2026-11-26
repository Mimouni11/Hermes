"""The only module that talks to a language model.

Two backends behind one :func:`complete` call: Gemini through `google-genai`
(the default, free tier) and a local model through ollama (the offline
fallback, so the demo survives a dead API key or no network). Both go through
the same disk cache in `data/processed`, keyed on the prompt and model, which
makes repeated notebook runs and test runs free and deterministic.

**The timeout is load-bearing, not defensive.** Measured against this key: the
same prompt to the same model returned in 0.89 s once and 33.4 s another time,
and two of the three Flash models that `models.list` advertises answer 503 on
every attempt. With roughly 1.5 s of the 5-second turn budget left after STT
and TTS, a call that cannot be bounded cannot be used. So every call is given
a hard wall-clock budget and the caller always gets *something* back — for the
FAQ that something is the centre's own stored answer, which needs no model at
all.

Every fallback is counted. A rate that climbs is the free tier degrading, and
that is a number the report should carry rather than a surprise on demo day.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from hermes.config import get_settings


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


class LLMTimeout(LLMError):
    """The call did not finish inside its budget."""


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------

@dataclass
class Usage:
    """How the LLM path has behaved this process.

    Kept because the free tier is the project's least predictable dependency:
    503s on two of three Flash models and a 37x latency spread on the third.
    A fallback rate is the honest way to report that.
    """

    calls: int = 0
    served: int = 0
    from_cache: int = 0
    timed_out: int = 0
    failed: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def fallback_rate(self) -> float:
        """Share of calls that did not come from the model."""
        if not self.calls:
            return 0.0
        return (self.timed_out + self.failed) / self.calls

    def summary(self) -> str:
        """One line for a log or a report."""
        if not self.calls:
            return "LLM: no calls"
        mean = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        return (
            f"LLM: {self.calls} calls, {self.served} served "
            f"({self.from_cache} cached), {self.timed_out} timed out, "
            f"{self.failed} failed, fallback {self.fallback_rate:.0%}, "
            f"mean {mean:.2f}s"
        )


USAGE = Usage()


def reset_usage() -> None:
    """Start the counters again."""
    global USAGE
    USAGE = Usage()


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def cache_key(prompt: str, system: str | None, model: str, temperature: float) -> str:
    """Return the stable hash under which a response is cached."""
    material = "␟".join([prompt or "", system or "", model, f"{temperature:.3f}"])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def _cache_dir() -> Path:
    directory = get_settings().paths.processed / "llm_cache"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def cache_get(key: str) -> Completion | None:
    """Look a response up in the on-disk cache."""
    path = _cache_dir() / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return Completion(
        text=data["text"],
        model=data.get("model", ""),
        backend=data.get("backend", ""),
        prompt_tokens=data.get("prompt_tokens"),
        completion_tokens=data.get("completion_tokens"),
        latency_s=float(data.get("latency_s", 0.0)),
        from_cache=True,
    )


def cache_put(key: str, completion: Completion) -> None:
    """Write a response to the on-disk cache."""
    path = _cache_dir() / f"{key}.json"
    path.write_text(
        json.dumps(
            {
                "text": completion.text,
                "model": completion.model,
                "backend": completion.backend,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "latency_s": completion.latency_s,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def clear_cache() -> int:
    """Empty the LLM cache and return how many entries were dropped."""
    directory = _cache_dir()
    removed = 0
    for path in directory.glob("*.json"):
        path.unlink()
        removed += 1
    return removed


# --------------------------------------------------------------------------
# Bounding a call
# --------------------------------------------------------------------------

def run_within(seconds: float, work: Callable[[], Any]) -> Any:
    """Run ``work`` and raise :class:`LLMTimeout` if it overruns.

    A thread rather than the SDK's own timeout, deliberately. The SDK's
    timeout governs the HTTP exchange; this governs the wall clock the caller
    actually has. The worker is left to finish in the background — there is no
    safe way to kill it, and its result is simply dropped.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(work)
    try:
        return future.result(timeout=seconds)
    except FutureTimeout as exc:
        raise LLMTimeout(f"no answer within {seconds:.1f}s") from exc
    finally:
        pool.shutdown(wait=False)


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

_CLIENT: dict[str, Any] = {}


def gemini_client(api_key: str | None = None, model: str | None = None) -> Any:
    """Build and cache the Gemini client.

    `google-genai`, not `google-generativeai`: the latter is deprecated and
    its repository is now named `deprecated-generative-ai-python`.
    """
    from hermes.config import require_gemini_key

    key = api_key or require_gemini_key()
    if "client" not in _CLIENT:
        from google import genai

        _CLIENT["client"] = genai.Client(api_key=key)
    return _CLIENT["client"]


def complete_gemini(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.2,
    json_mode: bool = False,
    timeout: float | None = None,
) -> Completion:
    """Send one prompt to Gemini, inside a hard wall-clock budget."""
    from google.genai import types

    settings = get_settings()
    budget = settings.llm_timeout_s if timeout is None else timeout

    # Outside the budget on purpose, and warmed by `warm()` at startup. The
    # first construction imports the SDK and costs about 14 s — measured —
    # which would eat any budget whole. Billing that to the first caller's
    # turn would be the same mistake as loading Whisper inside a timed block.
    client = gemini_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system or None,
        response_mime_type="application/json" if json_mode else None,
    )

    started = time.perf_counter()
    response = run_within(
        budget,
        lambda: client.models.generate_content(
            model=settings.gemini_model, contents=prompt, config=config
        ),
    )
    elapsed = time.perf_counter() - started

    usage = getattr(response, "usage_metadata", None)
    return Completion(
        text=(response.text or "").strip(),
        model=settings.gemini_model,
        backend="gemini",
        prompt_tokens=getattr(usage, "prompt_token_count", None),
        completion_tokens=getattr(usage, "candidates_token_count", None),
        latency_s=elapsed,
        from_cache=False,
    )


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
    timeout: float | None = None,
    key: str | None = None,
) -> Completion:
    """Complete a prompt, or raise. Cache first, then the model, inside budget.

    ``key`` overrides the cache key. Callers that know what actually
    determines the answer should supply one: hashing the whole prompt makes
    the cache miss on any difference at all, including ones that cannot change
    the reply.

    Raises rather than inventing a reply: the caller knows what its own
    fallback is, and for the FAQ that fallback is the centre's own text. A
    module that guessed here would be the one putting words in the agent's
    mouth.
    """
    settings = get_settings()
    chosen = backend or ("ollama" if settings.use_local_llm else "gemini")
    if key is None:
        key = cache_key(prompt, system, settings.gemini_model, temperature)

    USAGE.calls += 1

    if use_cache:
        cached = cache_get(key)
        if cached is not None:
            USAGE.served += 1
            USAGE.from_cache += 1
            return cached

    try:
        if chosen == "gemini":
            completion = complete_gemini(
                prompt, system, temperature, json_mode, timeout
            )
        else:
            completion = complete_ollama(prompt, system, temperature, json_mode)
    except LLMTimeout:
        USAGE.timed_out += 1
        raise
    except Exception as exc:  # noqa: BLE001 — one failure mode for the caller
        USAGE.failed += 1
        raise LLMError(f"{type(exc).__name__}: {exc}") from exc

    USAGE.served += 1
    USAGE.latencies.append(completion.latency_s)
    if use_cache:
        cache_put(key, completion)
    return completion


def warm() -> bool:
    """Build the client before the first turn needs it.

    Returns whether it succeeded. Called at startup the way the Whisper model
    is: the SDK import measured about 14 s, and a per-call budget cannot
    absorb that — the first caller would time out no matter how fast the API
    answered.
    """
    settings = get_settings()
    if settings.use_local_llm:
        return ollama_available()
    try:
        gemini_client()
        return True
    except Exception:  # noqa: BLE001 — a cold client is a fallback, not a crash
        return False


def complete_or(
    fallback: str,
    prompt: str,
    system: str | None = None,
    temperature: float = 0.2,
    timeout: float | None = None,
    key: str | None = None,
) -> tuple[str, str]:
    """Complete a prompt, or return ``fallback``. Never raises.

    Returns ``(text, source)`` where source is "llm", "cache" or the reason
    the fallback was used. The caller can log which, and the counters in
    :data:`USAGE` aggregate it.
    """
    try:
        completion = complete(
            prompt, system=system, temperature=temperature, timeout=timeout,
            key=key,
        )
    except LLMTimeout:
        return fallback, "timeout"
    except LLMError as exc:
        return fallback, f"error:{type(exc).__name__}"

    if not completion.text.strip():
        return fallback, "empty"
    return completion.text, "cache" if completion.from_cache else "llm"


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
