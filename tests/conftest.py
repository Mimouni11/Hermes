"""Shared pytest fixtures: temporary paths, sample data, fake backends.

Everything that would hit the network, a GPU or the real `data/` directory gets
replaced here, so the test suite runs offline and leaves no artefacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest


@pytest.fixture
def tmp_paths(tmp_path: Path) -> Iterator[Any]:
    """Yield a :class:`hermes.config.Paths` rooted in a temporary directory."""
    ...


@pytest.fixture
def settings(tmp_paths: Any) -> Iterator[Any]:
    """Yield test settings: temporary paths, local backends, no API key."""
    ...


#: Everything the audio fixtures write is at the pipeline's rate.
AUDIO_SR = 16_000


def _speech_like(seconds: float = 2.0, sr: int = AUDIO_SR, seed: int = 7) -> Any:
    """Build a crude voiced signal: a tone with harmonics, gated into syllables.

    A bare sine wave is a bad stand-in for speech here, because its frame
    energies are constant — the noise floor and the speech level come out equal
    and the SNR estimate reads 0 dB for perfectly clean audio. Gating the tone
    into syllables gives the quiet frames the estimator needs.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    tone = (
        0.6 * np.sin(2 * np.pi * 140 * t)
        + 0.3 * np.sin(2 * np.pi * 280 * t)
        + 0.1 * np.sin(2 * np.pi * 420 * t)
    )

    # Four syllables a second, each with a gap: speech is not continuous.
    envelope = (np.sin(2 * np.pi * 4 * t) > -0.2).astype("float64")
    smoothing = max(int(0.005 * sr), 1)
    envelope = np.convolve(envelope, np.ones(smoothing) / smoothing, mode="same")

    signal = tone * envelope
    signal += 0.001 * rng.standard_normal(signal.size)  # a touch of mic hiss
    peak = float(np.max(np.abs(signal)))
    return (signal / peak * 0.5).astype("float32")


@pytest.fixture
def sine_wav(tmp_path: Path) -> Path:
    """Write a clean tone to a wav file and return its path."""
    import soundfile as sf

    path = tmp_path / "clean.wav"
    sf.write(path, _speech_like(), AUDIO_SR, subtype="PCM_16")
    return path


@pytest.fixture
def noisy_wav(tmp_path: Path) -> Path:
    """Write a tone buried in noise to a wav file and return its path."""
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(11)
    clean = _speech_like()
    noise = rng.standard_normal(clean.size).astype("float32")

    # Mix at roughly 0 dB SNR: well under any sane min_snr_db.
    clean_rms = float(np.sqrt(np.mean(clean.astype("float64") ** 2)))
    noise_rms = float(np.sqrt(np.mean(noise.astype("float64") ** 2)))
    mixed = clean + noise * (clean_rms / noise_rms)
    mixed = mixed / max(float(np.max(np.abs(mixed))), 1.0) * 0.9

    path = tmp_path / "noisy.wav"
    sf.write(path, mixed.astype("float32"), AUDIO_SR, subtype="PCM_16")
    return path


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Replace :mod:`hermes.llm` with a scripted responder, no network."""
    ...


@pytest.fixture
def fake_stt(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Replace transcription with a canned transcript, no model download."""
    ...


@pytest.fixture
def sample_exams() -> Any:
    """Return a small exam catalogue dataframe."""
    ...


@pytest.fixture
def sample_faq() -> Any:
    """Return a small FAQ dataframe."""
    ...


@pytest.fixture
def agenda_db(tmp_path: Path) -> Iterator[Path]:
    """Yield an initialised, seeded agenda database in a temporary directory."""
    ...
