"""Tests for FAQ retrieval: the right entry, and silence when there is none."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet")


def test_paraphrased_question_retrieves_its_entry(sample_faq: Any) -> None:
    """A reworded question still retrieves the FAQ entry it came from."""
    ...


def test_results_are_ordered_by_score(sample_faq: Any) -> None:
    """Search returns hits sorted best first."""
    ...


def test_k_limits_the_number_of_hits(sample_faq: Any) -> None:
    """Search returns at most ``k`` entries."""
    ...


def test_unanswerable_question_returns_none(sample_faq: Any) -> None:
    """A question the FAQ does not cover yields no best match."""
    ...


def test_index_round_trip(sample_faq: Any) -> None:
    """An index saved to disk loads back with the same entries and vectors."""
    ...


def test_context_block_contains_the_answers(sample_faq: Any) -> None:
    """The formatted grounding block carries the retrieved answer text."""
    ...
