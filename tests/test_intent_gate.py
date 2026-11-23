"""Tests for intent routing and, above all, for refusing what is out of scope."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet")


def test_booking_phrase_routes_to_book() -> None:
    """"I'd like an appointment for an MRI" routes to the booking intent."""
    ...


def test_faq_phrase_routes_to_faq() -> None:
    """A question about opening hours routes to the FAQ intent."""
    ...


def test_human_request_routes_to_human() -> None:
    """"Put me through to someone" routes to the human intent."""
    ...


def test_unrelated_question_is_out_of_scope() -> None:
    """A question with nothing to do with imaging is refused as out of scope."""
    ...


def test_low_margin_is_treated_as_unknown() -> None:
    """Two near-tied intents yield unknown rather than a coin flip."""
    ...


def test_empty_utterance_is_unknown() -> None:
    """An empty or whitespace utterance never routes to a real intent."""
    ...


def test_centroids_round_trip(tmp_path: object) -> None:
    """Centroids saved to disk load back unchanged."""
    ...
