"""The conversational brain: exactly one LLM call per turn, answering in JSON.

Assembles the prompt from everything the earlier stages produced — transcript,
intent, FAQ passages, exam candidates, frustration score, dialogue state — and
asks the model for a single structured object: the reply to speak, the slots it
filled, the action to take (answer, ask, book, cancel, escalate) and whether the
turn is done.

One call per turn is the design constraint: it keeps the call line responsive
and the cost predictable. Nothing here talks to the network directly, that is
:mod:`hermes.infra.llm`; nothing here acts on the decision, that is
:mod:`hermes.pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hermes.understand.exam_bot import MatchResult
from hermes.understand.faq_rag import Retrieved
from hermes.understand.frustration import FrustrationScore
from hermes.understand.intent_gate import IntentDecision


class Action(str, Enum):
    """What the pipeline should do once the agent has spoken."""

    ANSWER = "answer"
    ASK = "ask"
    CONFIRM = "confirm"
    BOOK = "book"
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"
    OFFER_SLOTS = "offer_slots"
    ESCALATE = "escalate"
    REFUSE = "refuse"
    END = "end"


@dataclass
class Turn:
    """One exchange in the call history."""

    role: str
    text: str
    intent: str | None = None
    action: str | None = None


@dataclass
class DialogueState:
    """What the agent has gathered so far across the call."""

    call_id: str
    turns: list[Turn] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)
    pending_action: Action | None = None
    failed_matches: int = 0
    audio_retries: int = 0
    escalated: bool = False


@dataclass(frozen=True)
class AgentDecision:
    """The parsed JSON the model returned for one turn."""

    reply: str
    action: Action
    intent: str
    slots: dict[str, Any]
    needs: tuple[str, ...]
    escalate: bool
    end_call: bool
    confidence: float
    raw: dict[str, Any]


def system_prompt() -> str:
    """Return the standing instructions: role, scope, tone, JSON contract."""
    ...


def build_prompt(
    text: str,
    state: DialogueState,
    intent: IntentDecision | None = None,
    faq_hits: list[Retrieved] | None = None,
    exam_result: MatchResult | None = None,
    frustration: FrustrationScore | None = None,
) -> str:
    """Assemble the single user prompt for this turn from the stage outputs."""
    ...


def response_schema() -> dict[str, Any]:
    """Return the JSON schema the model's answer must satisfy."""
    ...


def parse_response(raw: str) -> AgentDecision:
    """Parse and validate the model's JSON, repairing minor formatting slips."""
    ...


def fallback_decision(reason: str) -> AgentDecision:
    """Return a safe decision for when the model fails or returns junk."""
    ...


def decide(
    text: str,
    state: DialogueState,
    intent: IntentDecision | None = None,
    faq_hits: list[Retrieved] | None = None,
    exam_result: MatchResult | None = None,
    frustration: FrustrationScore | None = None,
) -> AgentDecision:
    """Run the one LLM call for this turn and return the parsed decision."""
    ...


def update_state(state: DialogueState, text: str, decision: AgentDecision) -> DialogueState:
    """Fold a decision back into the dialogue state for the next turn."""
    ...
