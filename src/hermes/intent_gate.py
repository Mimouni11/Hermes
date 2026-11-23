"""First routing decision of every turn: what does the caller want, if anything.

Embeds the caller's utterance with a sentence-transformers model and compares it
against the centroids of the labelled intent examples in `data/raw`. Routes to
one of the handled intents (FAQ, exam question, booking, cancellation, human) or
rejects the turn as out of scope — the gate that keeps a medical imaging call
line from answering questions it has no business answering.

Accepts only when the best similarity clears the threshold *and* beats the
runner-up by the configured margin; everything else is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class Intent(str, Enum):
    """The intents the agent is allowed to act on, plus the catch-alls."""

    FAQ = "faq"
    EXAM_INFO = "exam_info"
    BOOK = "book"
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"
    PREPARATION = "preparation"
    RESULTS = "results"
    HUMAN = "human"
    SMALL_TALK = "small_talk"
    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntentDecision:
    """The gate's verdict on one utterance."""

    intent: Intent
    score: float
    margin: float
    runner_up: Intent | None
    in_scope: bool
    scores: dict[Intent, float]


def load_encoder(model_name: str | None = None) -> Any:
    """Load and cache the sentence-transformers encoder."""
    ...


def encode(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts as L2-normalised row vectors."""
    ...


def load_examples(path: str | Path | None = None) -> dict[Intent, list[str]]:
    """Read the labelled intent examples used to build the centroids."""
    ...


def build_centroids(
    examples: dict[Intent, list[str]] | None = None,
    out_path: str | Path | None = None,
) -> dict[Intent, np.ndarray]:
    """Average the example embeddings per intent and cache the result."""
    ...


def load_centroids(path: str | Path | None = None) -> dict[Intent, np.ndarray]:
    """Load the precomputed intent centroids from `data/processed`."""
    ...


def classify(text: str) -> IntentDecision:
    """Route one utterance to an intent, or mark it out of scope."""
    ...


def is_out_of_scope(decision: IntentDecision) -> bool:
    """Return whether a decision should be refused rather than handled."""
    ...


def out_of_scope_reply() -> str:
    """Return the line the agent speaks when a request is out of scope."""
    ...
