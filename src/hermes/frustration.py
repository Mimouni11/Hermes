"""Notice when a caller is getting annoyed, before they have to say so.

A small scikit-learn MLP over sentence embeddings of the caller's utterance,
plus a few conversation features (turn index, repeats, failed matches), trained
in `notebooks/` on the labelled transcripts in `data/raw`. The score it returns
feeds :mod:`hermes.escalation`: past the threshold, the call goes to a human.

Weights live in `models/` and are gitignored; the model is loaded once per
process and falls back to a neutral score if the file is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier


@dataclass(frozen=True)
class TurnFeatures:
    """The conversation signals that accompany the text embedding."""

    turn_index: int
    repeated_question: bool
    failed_matches: int
    audio_retries: int
    words: int
    exclamations: int


@dataclass(frozen=True)
class FrustrationScore:
    """How frustrated the caller sounds on this turn."""

    score: float
    label: str
    is_frustrated: bool
    contributing: tuple[str, ...]


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Read the labelled frustration transcripts from `data/raw`."""
    ...


def featurise(text: str, features: TurnFeatures | None = None) -> np.ndarray:
    """Build one feature vector: text embedding concatenated with turn signals."""
    ...


def featurise_batch(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build the training matrix and label vector from a labelled dataframe."""
    ...


def train(
    frame: pd.DataFrame | None = None,
    out_path: str | Path | None = None,
) -> MLPClassifier:
    """Fit the MLP, report held-out metrics and save the weights to `models/`."""
    ...


def evaluate(model: MLPClassifier, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Score a trained model: accuracy, precision, recall, F1, ROC AUC."""
    ...


def load_model(path: str | Path | None = None) -> Any:
    """Load and cache the trained MLP from `models/`."""
    ...


def predict(text: str, features: TurnFeatures | None = None) -> FrustrationScore:
    """Score one caller turn for frustration."""
    ...
