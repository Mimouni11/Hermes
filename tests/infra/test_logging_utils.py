"""Tests for anonymisation — the ones that matter most in this project."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet")


def test_phone_numbers_are_removed() -> None:
    """A phone number in any common format is replaced by a placeholder."""
    ...


def test_names_are_removed() -> None:
    """A person's name is replaced by a placeholder."""
    ...


def test_dates_of_birth_are_removed() -> None:
    """A date of birth is replaced, while an appointment date survives."""
    ...


def test_emails_and_addresses_are_removed() -> None:
    """Emails and street addresses are replaced by placeholders."""
    ...


def test_social_security_numbers_are_removed() -> None:
    """A social security number is replaced by a placeholder."""
    ...


def test_caller_ref_is_stable_and_not_reversible() -> None:
    """The same identifier hashes the same way and does not contain the input."""
    ...


def test_logged_turn_contains_no_raw_transcript(tmp_path: Path) -> None:
    """What reaches the log file is the scrubbed text, never the original."""
    ...


def test_log_lines_are_valid_jsonl(tmp_path: Path) -> None:
    """Every written line parses as a standalone JSON object."""
    ...
