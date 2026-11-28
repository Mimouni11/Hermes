"""Tests for the LLM wrapper: caching, JSON mode and backend fallback."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet")


def test_identical_prompts_hit_the_cache(fake_llm: Any) -> None:
    """The second identical call is served from disk, not the backend."""
    ...


def test_different_temperature_is_a_different_key() -> None:
    """Cache keys separate prompts that differ only in their parameters."""
    ...


def test_cache_can_be_bypassed(fake_llm: Any) -> None:
    """``use_cache=False`` reaches the backend even on a repeat prompt."""
    ...


def test_gemini_failure_falls_back_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing Gemini call is retried on the local backend."""
    ...


def test_both_backends_failing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no backend available the wrapper raises its own error."""
    ...


def test_json_mode_returns_a_dict(fake_llm: Any) -> None:
    """A JSON-mode completion comes back parsed."""
    ...


def test_missing_api_key_is_reported_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent key produces a readable error, not an attribute error."""
    ...
