"""Tests for the frustration model's shapes, thresholds and fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet")


def test_feature_vector_has_the_expected_width() -> None:
    """One turn becomes a vector of embedding plus turn-signal features."""
    ...


def test_score_is_a_probability() -> None:
    """The returned score sits between zero and one."""
    ...


def test_calm_utterance_scores_low() -> None:
    """A polite first-turn question is not flagged as frustrated."""
    ...


def test_angry_repeated_utterance_scores_high() -> None:
    """A repeated, late-turn complaint is flagged as frustrated."""
    ...


def test_threshold_decides_the_flag() -> None:
    """The boolean flag follows the configured escalation threshold."""
    ...


def test_missing_weights_return_a_neutral_score(tmp_path: Path) -> None:
    """With no trained model on disk, scoring degrades instead of crashing."""
    ...


def test_training_saves_weights(tmp_path: Path) -> None:
    """Training writes the model file and reports held-out metrics."""
    ...
