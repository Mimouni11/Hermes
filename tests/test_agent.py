"""Tests for prompt assembly and for parsing the model's JSON answer."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet")


def test_prompt_includes_the_retrieved_faq(sample_faq: Any) -> None:
    """Retrieved FAQ passages appear in the prompt as grounding."""
    ...


def test_prompt_includes_the_exam_candidates(sample_exams: Any) -> None:
    """Exam candidates appear in the prompt when the intent calls for them."""
    ...


def test_prompt_includes_the_dialogue_history() -> None:
    """Earlier turns and filled slots are carried into the prompt."""
    ...


def test_one_llm_call_per_turn(fake_llm: Any) -> None:
    """A turn triggers exactly one completion, never two."""
    ...


def test_valid_json_is_parsed() -> None:
    """A well-formed response becomes an agent decision."""
    ...


def test_json_in_a_code_fence_is_parsed() -> None:
    """A response wrapped in a markdown fence is still parsed."""
    ...


def test_malformed_json_falls_back_safely() -> None:
    """Unparseable output yields the safe fallback, not an exception."""
    ...


def test_unknown_action_falls_back_safely() -> None:
    """An action outside the enum yields the safe fallback."""
    ...


def test_state_update_appends_the_turn() -> None:
    """Folding a decision into the state records the turn and the slots."""
    ...
