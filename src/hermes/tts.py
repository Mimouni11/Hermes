"""Text to speech for the agent's side of the conversation.

Two free backends behind one function: piper-tts (local, fast, the default for
the demo) and gTTS (network, used as a fallback and for voice comparison in the
report). Synthesised lines are cached on disk by hash, since the agent repeats
the same prompts and safety scripts constantly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Speech:
    """One synthesised utterance."""

    path: Path
    text: str
    duration_s: float
    sample_rate: int
    backend: str
    from_cache: bool


def load_voice(backend: str | None = None, voice: str | None = None) -> Any:
    """Load and cache a TTS voice for the requested backend."""
    ...


def synthesise(
    text: str,
    out_path: str | Path | None = None,
    backend: str | None = None,
    voice: str | None = None,
) -> Speech:
    """Render ``text`` to a wav file and return where it landed."""
    ...


def synthesise_samples(
    text: str,
    backend: str | None = None,
) -> tuple[np.ndarray, int]:
    """Render ``text`` straight to samples, skipping the disk."""
    ...


def prepare_text(text: str) -> str:
    """Make text speakable: expand abbreviations, dates, times and exam codes."""
    ...


def cache_key(text: str, backend: str, voice: str | None) -> str:
    """Return the stable cache filename for one utterance."""
    ...


def clear_cache() -> int:
    """Delete the cached audio and return how many files were removed."""
    ...
