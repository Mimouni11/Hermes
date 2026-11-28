"""One turn of the call, start to finish: audio in, audio out.

The order is fixed, and each stage can short-circuit the ones after it:

1. :mod:`hermes.listen.audio_quality`  too noisy? ask the caller to repeat, stop here
2. :mod:`hermes.listen.stt`            transcribe
3. :mod:`hermes.guard.escalation`     emergency words? read the script, transfer, stop
4. :mod:`hermes.understand.intent_gate`    route, or refuse as out of scope
5. :mod:`hermes.understand.faq_rag` / :mod:`hermes.understand.exam_bot`  gather grounding for the intent
6. :mod:`hermes.understand.frustration`    score the caller's patience — recorded only,
   never acted on here; WP6's orchestrator owns what a level means
7. :mod:`hermes.agent`          one LLM call, JSON decision
8. :mod:`hermes.guard.escalation`     re-check the rules against the decision
9. :mod:`hermes.book.agenda`         carry out a booking, reschedule or cancellation
10. :mod:`hermes.speak.tts`           speak the reply
11. :mod:`hermes.infra.logging_utils` log the turn, anonymised

:func:`run_turn` is what the Gradio demo and the tests call; :func:`run_call`
loops it over several clips for the end-to-end evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes.agent import AgentDecision, DialogueState
from hermes.listen.audio_quality import AudioReport
from hermes.guard.escalation import Escalation
from hermes.understand.frustration import FrustrationScore
from hermes.understand.intent_gate import IntentDecision
from hermes.listen.stt import Transcript
from hermes.speak.tts import Speech


@dataclass
class TurnResult:
    """Everything one turn produced, for the UI, the logs and the tests."""

    call_id: str
    turn_index: int
    audio_report: AudioReport | None
    transcript: Transcript | None
    intent: IntentDecision | None
    frustration: FrustrationScore | None
    decision: AgentDecision | None
    escalation: Escalation | None
    reply_text: str
    reply_audio: Speech | None
    appointment_id: str | None
    end_call: bool
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None


def score_frustration(text: str, lang: str = "fr") -> FrustrationScore:
    """Stage 6: score the caller's patience. **Observation only.**

    The result goes into `TurnResult.frustration` and the turn log, and
    nothing in this module branches on it. Deciding what a `soft` or `hard`
    level *does* — apologise and simplify, hand over, or hand over after two
    consecutive soft hits — belongs to the WP6 orchestrator, which is the only
    place with the dialogue state those rules need.

    Kept deliberately thin so that when `run_turn` is implemented it has one
    call to make and no policy to re-derive.
    """
    from hermes.understand import frustration

    return frustration.detect(text, lang=lang)


def start_call(caller_id: str | None = None) -> DialogueState:
    """Open a call: new id, empty state, greeting logged."""
    ...


def greeting() -> str:
    """Return the line the agent opens the call with."""
    ...


def run_turn(
    audio_path: str | Path,
    state: DialogueState,
    speak: bool = True,
) -> TurnResult:
    """Run one full turn: caller audio in, agent audio out."""
    ...


def run_text_turn(
    text: str,
    state: DialogueState,
    speak: bool = False,
) -> TurnResult:
    """Run a turn from typed text, skipping the audio stages."""
    ...


def run_call(
    audio_paths: list[str | Path],
    caller_id: str | None = None,
) -> list[TurnResult]:
    """Run a whole scripted call, one clip per turn."""
    ...


def apply_action(decision: AgentDecision, state: DialogueState) -> str | None:
    """Carry out a booking action against the agenda, returning any new id."""
    ...


def end_call(state: DialogueState, outcome: str) -> None:
    """Close a call: write the summary record and release any resources."""
    ...


def warm_up() -> dict[str, Any]:
    """Preload the models and indexes so the first real turn is not slow."""
    ...

