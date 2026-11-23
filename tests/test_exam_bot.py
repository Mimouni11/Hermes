"""Tests for exam matching: exact names, aliases, typos and ambiguity."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet")


def test_exact_name_matches(sample_exams: Any) -> None:
    """The catalogue's own wording matches with full confidence."""
    ...


def test_alias_matches(sample_exams: Any) -> None:
    """A known synonym resolves to the official exam."""
    ...


def test_typo_matches_fuzzily(sample_exams: Any) -> None:
    """A misspelling still matches above the fuzzy threshold."""
    ...


def test_accents_and_filler_are_normalised(sample_exams: Any) -> None:
    """Accents, punctuation and filler words do not affect the match."""
    ...


def test_unrelated_phrase_does_not_match(sample_exams: Any) -> None:
    """A phrase naming no exam returns no confident best match."""
    ...


def test_two_close_exams_are_ambiguous(sample_exams: Any) -> None:
    """Two similarly scoring exams are flagged ambiguous with both candidates."""
    ...


def test_ambiguity_produces_a_clarifying_question(sample_exams: Any) -> None:
    """An ambiguous result yields a question naming the candidates."""
    ...


def test_gazetteer_covers_every_exam(sample_exams: Any) -> None:
    """Every catalogue exam has at least one surface form in the gazetteer."""
    ...
