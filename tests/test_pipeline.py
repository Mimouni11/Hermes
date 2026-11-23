"""End-to-end tests for one turn, with every external model faked."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet")


def test_turn_returns_reply_text_and_audio(sine_wav: Path, fake_stt: Any, fake_llm: Any) -> None:
    """A good clip produces a spoken reply and a filled result."""
    ...


def test_noisy_clip_short_circuits_before_stt(noisy_wav: Path, fake_stt: Any) -> None:
    """A clip failing the SNR gate never reaches transcription."""
    ...


def test_out_of_scope_turn_skips_the_llm(fake_llm: Any) -> None:
    """A refused turn is answered from the script, with no completion spent."""
    ...


def test_emergency_turn_skips_the_llm(fake_llm: Any) -> None:
    """An emergency reads the script and transfers without an LLM call."""
    ...


def test_booking_turn_writes_an_appointment(agenda_db: Path, fake_llm: Any) -> None:
    """A confirmed booking action creates an appointment and returns its id."""
    ...


def test_state_persists_across_turns(fake_llm: Any) -> None:
    """Slots filled on one turn are still there on the next."""
    ...


def test_every_turn_is_logged(tmp_path: Path, fake_llm: Any) -> None:
    """Each turn appends exactly one record to the call log."""
    ...


def test_stage_timings_are_recorded(sine_wav: Path, fake_llm: Any) -> None:
    """The result carries a duration for each stage that ran."""
    ...


def test_backend_failure_degrades_gracefully(fake_llm: Any) -> None:
    """A dead LLM backend yields an apology and an escalation, not a crash."""
    ...
