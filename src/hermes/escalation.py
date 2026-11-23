"""When to stop being a bot, and what to say while handing over.

Collects every reason this call should reach a person — the caller asked, the
frustration score crossed the line, the turn count ran out, the intent gate kept
failing, the request is out of scope, or the utterance mentions something urgent
— and returns one decision with a priority and the exact words to speak.

The safety script is deliberate: a medical imaging call line must never triage,
interpret results or give medical advice. If a caller describes a medical
emergency, the agent reads the emergency script and transfers immediately,
whatever else was happening in the dialogue. These rules outrank the agent's own
judgement and are applied after the LLM call, not by it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hermes.agent import AgentDecision, DialogueState
from hermes.frustration import FrustrationScore
from hermes.intent_gate import IntentDecision


class Reason(str, Enum):
    """Why a call is being escalated."""

    CALLER_REQUEST = "caller_request"
    FRUSTRATION = "frustration"
    MEDICAL_URGENCY = "medical_urgency"
    MEDICAL_ADVICE = "medical_advice"
    RESULTS_REQUEST = "results_request"
    REPEATED_FAILURE = "repeated_failure"
    TURN_LIMIT = "turn_limit"
    OUT_OF_SCOPE = "out_of_scope"
    AUDIO_QUALITY = "audio_quality"
    SYSTEM_ERROR = "system_error"
    AGENT_REQUEST = "agent_request"


class Priority(str, Enum):
    """How fast the handover has to happen."""

    IMMEDIATE = "immediate"
    HIGH = "high"
    NORMAL = "normal"


@dataclass(frozen=True)
class Escalation:
    """The decision to hand a call to a human."""

    escalate: bool
    reason: Reason | None
    priority: Priority
    script: str
    transfer_to: str | None
    end_call: bool


def urgency_keywords() -> frozenset[str]:
    """Return the phrases that mean a caller may be having an emergency."""
    ...


def advice_keywords() -> frozenset[str]:
    """Return the phrases that mean the caller is asking for medical advice."""
    ...


def detect_urgency(text: str) -> bool:
    """Return whether an utterance describes a possible medical emergency."""
    ...


def detect_advice_request(text: str) -> bool:
    """Return whether an utterance asks for a diagnosis or medical advice."""
    ...


def emergency_script() -> str:
    """Return the fixed words for a suspected emergency: call 15, transferring now."""
    ...


def safety_script(reason: Reason) -> str:
    """Return the handover wording for a given escalation reason."""
    ...


def check(
    text: str,
    state: DialogueState,
    decision: AgentDecision | None = None,
    intent: IntentDecision | None = None,
    frustration: FrustrationScore | None = None,
) -> Escalation:
    """Apply every escalation rule to this turn and return the verdict."""
    ...


def route(reason: Reason) -> str:
    """Return the queue or extension a reason should be transferred to."""
    ...


def no_escalation() -> Escalation:
    """Return the neutral decision: carry on with the bot."""
    ...
