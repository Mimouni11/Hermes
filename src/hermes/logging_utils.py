"""Call logs that are useful for the report but hold no personal data.

Writes one JSONL record per turn and one summary per call: timings per stage,
intent, scores, action taken, escalation reason, backend used. Before anything
is written, the text goes through a scrubber that removes names, phone numbers,
dates of birth, social security numbers, emails and addresses, and the caller is
identified only by a salted hash.

This is a school project on real-sounding medical calls, so the rule is simple:
nothing identifiable reaches disk, and the audio clips are gitignored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TurnRecord:
    """One logged turn, already anonymised."""

    call_id: str
    turn_index: int
    timestamp: str
    transcript: str
    reply: str
    intent: str | None
    intent_score: float | None
    frustration_score: float | None
    action: str | None
    escalated: bool
    escalation_reason: str | None
    snr_db: float | None
    stage_timings_ms: dict[str, float]
    backend: str | None


@dataclass(frozen=True)
class CallRecord:
    """The summary written when a call ends."""

    call_id: str
    caller_ref: str
    started_at: str
    ended_at: str
    turns: int
    outcome: str
    escalated: bool
    escalation_reason: str | None
    appointment_id: str | None


def get_logger(name: str = "hermes") -> logging.Logger:
    """Return the configured logger, set up on first call."""
    ...


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure console and file handlers for the whole package."""
    ...


# --------------------------------------------------------------------------
# Anonymisation
# --------------------------------------------------------------------------

def scrub(text: str) -> str:
    """Replace personal data in free text with placeholder tags."""
    ...


def scrub_phone(text: str) -> str:
    """Replace phone numbers with ``[PHONE]``."""
    ...


def scrub_names(text: str) -> str:
    """Replace detected person names with ``[NAME]``."""
    ...


def scrub_dates(text: str) -> str:
    """Replace dates of birth with ``[DOB]``, leaving appointment dates alone."""
    ...


def caller_ref(identifier: str, salt: str | None = None) -> str:
    """Return the salted, truncated hash that stands in for a caller."""
    ...


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def new_call_id() -> str:
    """Return a fresh call id."""
    ...


def log_turn(record: TurnRecord, path: str | Path | None = None) -> None:
    """Append one scrubbed turn to the JSONL call log."""
    ...


def log_call(record: CallRecord, path: str | Path | None = None) -> None:
    """Append one call summary to the JSONL call log."""
    ...


def log_event(call_id: str, event: str, payload: dict[str, Any] | None = None) -> None:
    """Record a one-off event: a backend failure, a cache miss, a retry."""
    ...


# --------------------------------------------------------------------------
# Reading back
# --------------------------------------------------------------------------

def read_turns(path: str | Path | None = None) -> list[TurnRecord]:
    """Load the logged turns, for the evaluation notebooks."""
    ...


def read_calls(path: str | Path | None = None) -> list[CallRecord]:
    """Load the logged call summaries, for the evaluation notebooks."""
    ...
