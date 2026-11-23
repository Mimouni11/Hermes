"""Turn whatever the caller called the exam into an exam the centre offers.

Callers say "a scan of my head", "the MRI with the dye", "radio of the knee";
the catalogue in `data/raw` says something else. This module builds a gazetteer
from the catalogue — official names, abbreviations, synonyms, common
misspellings, body parts — and matches the caller's phrase against it with
rapidfuzz, after a normalisation pass that strips accents and filler words.

Exact and alias hits win outright. Fuzzy hits must clear the score threshold,
and an ambiguous result returns the candidates so the agent can ask the caller
which one they mean rather than picking for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Exam:
    """One exam from the centre's catalogue."""

    exam_id: str
    name: str
    modality: str
    body_part: str | None
    duration_min: int | None
    requires_contrast: bool
    requires_fasting: bool
    requires_prescription: bool
    preparation: str | None
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ExamMatch:
    """One candidate exam for a caller's phrase."""

    exam: Exam
    score: float
    matched_on: str
    method: str


@dataclass(frozen=True)
class MatchResult:
    """The outcome of matching one phrase against the gazetteer."""

    query: str
    best: ExamMatch | None
    candidates: tuple[ExamMatch, ...]
    is_ambiguous: bool
    is_confident: bool


def load_catalogue(path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw exam catalogue CSV into a dataframe."""
    ...


def parse_exams(catalogue: pd.DataFrame) -> list[Exam]:
    """Turn catalogue rows into :class:`Exam` objects."""
    ...


def build_gazetteer(
    exams: list[Exam] | None = None,
    out_path: str | Path | None = None,
) -> dict[str, str]:
    """Map every surface form (name, alias, abbreviation) to an ``exam_id``."""
    ...


def load_gazetteer(path: str | Path | None = None) -> dict[str, str]:
    """Load the cached gazetteer, building it if it is missing."""
    ...


def normalise(phrase: str) -> str:
    """Lowercase, strip accents and punctuation, drop filler words."""
    ...


def match(phrase: str, limit: int = 5, threshold: int | None = None) -> MatchResult:
    """Match a caller's phrase to catalogue exams, exact first then fuzzy."""
    ...


def get_exam(exam_id: str) -> Exam | None:
    """Look up one exam by its catalogue id."""
    ...


def preparation_text(exam: Exam) -> str:
    """Return the spoken preparation instructions for an exam."""
    ...


def clarification_question(result: MatchResult) -> str:
    """Return the question that disambiguates between candidate exams."""
    ...
