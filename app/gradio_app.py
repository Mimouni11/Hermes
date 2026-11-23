"""Gradio demo for the call line.

A microphone input and an audio output wired to :func:`hermes.pipeline.run_turn`,
with a transcript panel and a debug panel showing what each stage decided
(SNR, intent and score, exam candidates, frustration, action, escalation).
There is a text box too, so the pipeline can be driven without a microphone
during the presentation.

    python app/gradio_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from hermes.agent import DialogueState
from hermes.pipeline import TurnResult


def on_audio(
    audio_path: str | None,
    state: DialogueState | None,
) -> tuple[str, str | None, str, DialogueState]:
    """Handle a recorded turn: run the pipeline, return transcript, audio, debug."""
    ...


def on_text(
    text: str,
    state: DialogueState | None,
) -> tuple[str, str | None, str, DialogueState]:
    """Handle a typed turn, for demoing without a microphone."""
    ...


def on_reset() -> tuple[str, None, str, DialogueState]:
    """Start a fresh call and clear the panels."""
    ...


def format_transcript(state: DialogueState) -> str:
    """Render the call history as markdown for the transcript panel."""
    ...


def format_debug(result: TurnResult) -> str:
    """Render one turn's stage-by-stage decisions for the debug panel."""
    ...


def build_ui() -> gr.Blocks:
    """Assemble the Gradio interface."""
    ...


def main() -> int:
    """Warm the models up and launch the demo."""
    ...


if __name__ == "__main__":
    raise SystemExit(main())
