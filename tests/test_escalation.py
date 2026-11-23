"""Tests for the handover rules, including the ones that must never be skipped."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet")


def test_caller_asking_for_a_human_escalates() -> None:
    """An explicit request for a person escalates."""
    ...


def test_emergency_wording_escalates_immediately() -> None:
    """Words describing an emergency escalate at immediate priority."""
    ...


def test_emergency_outranks_everything_else() -> None:
    """Emergency escalation wins even mid-booking and overrides the agent."""
    ...


def test_emergency_script_tells_the_caller_to_call_emergency_services() -> None:
    """The emergency script names the emergency number and says it transfers."""
    ...


def test_request_for_diagnosis_escalates() -> None:
    """A request for medical advice is handed over, never answered."""
    ...


def test_results_request_escalates() -> None:
    """Asking for exam results is handed over, never answered."""
    ...


def test_high_frustration_escalates() -> None:
    """A frustration score past the threshold escalates."""
    ...


def test_turn_limit_escalates() -> None:
    """A call that runs past the turn limit escalates."""
    ...


def test_repeated_match_failures_escalate() -> None:
    """Several failed matches in a row escalate instead of looping."""
    ...


def test_normal_turn_does_not_escalate() -> None:
    """An ordinary in-scope turn carries on with the bot."""
    ...


def test_every_reason_has_a_script() -> None:
    """No escalation reason is left without words to say."""
    ...
