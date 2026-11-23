"""Retrieval over the centre's FAQ, so answers come from the centre's own text.

The FAQ lives as question/answer rows in `data/raw`. At build time every
question is embedded and the matrix is cached in `data/processed`; at call time
the caller's question is embedded and the nearest entries are returned. The
retrieved passages are handed to :mod:`hermes.agent` as grounding — the agent
rephrases them, it does not invent around them.

Returns nothing when no entry clears the acceptance threshold, which is the
signal to fall back or escalate rather than to guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FaqEntry:
    """One row of the centre's FAQ."""

    faq_id: str
    question: str
    answer: str
    category: str | None
    source: str | None


@dataclass(frozen=True)
class Retrieved:
    """One retrieved FAQ entry with its similarity to the caller's question."""

    entry: FaqEntry
    score: float


@dataclass(frozen=True)
class FaqIndex:
    """The embedded FAQ, ready to search."""

    entries: tuple[FaqEntry, ...]
    embeddings: np.ndarray
    model_name: str


def load_faq(path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw FAQ CSV into a dataframe."""
    ...


def build_index(
    faq: pd.DataFrame | None = None,
    out_path: str | Path | None = None,
) -> FaqIndex:
    """Embed every FAQ question and cache the index to `data/processed`."""
    ...


def load_index(path: str | Path | None = None) -> FaqIndex:
    """Load the cached FAQ index, building it if it is missing."""
    ...


def search(question: str, k: int = 3) -> list[Retrieved]:
    """Return the ``k`` closest FAQ entries to ``question``, best first."""
    ...


def best_match(question: str, threshold: float | None = None) -> Retrieved | None:
    """Return the single best FAQ entry, or ``None`` if none is close enough."""
    ...


def format_context(hits: list[Retrieved]) -> str:
    """Render retrieved entries as the grounding block for the LLM prompt."""
    ...
