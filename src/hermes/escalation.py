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

import re
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


# --------------------------------------------------------------------------
# The deterministic medical rules
# --------------------------------------------------------------------------
#
# These run *before* routing and again after the LLM. Measured justification:
# a class-balanced classifier over 209 French examples reaches 92% recall on
# medical questions, which means roughly one in twelve reaches the booking
# branch instead of a person. "J'ai un pacemaker, je peux faire l'IRM ?" was
# one of them. A rule that costs nothing catches it every time, and a false
# alarm costs one transfer.

#: Phrases that mean the caller may be having an emergency right now. These
#: trigger the fixed script and an immediate transfer.
URGENCY_PATTERNS: tuple[str, ...] = (
    r"douleur\s+(?:à|a|dans)\s+la\s+poitrine",
    r"mal\s+(?:à|a)\s+la\s+poitrine",
    r"j'?ai\s+du\s+mal\s+(?:à|a)\s+respirer",
    r"je\s+(?:n'?)?arrive\s+pas\s+(?:à|a)\s+respirer",
    r"je\s+saigne",
    r"perte\s+de\s+connaissance",
    r"(?:je\s+me\s+suis\s+)?évanoui",
    r"\bavc\b",
    r"\binfarctus\b",
    r"\burgence[s]?\b",
    r"c'?est\s+(?:une\s+)?urgence",
    # Darija
    r"نزيف",
    r"ما\s*نجمش\s*نتنفس",
    r"وجع\s*في\s*صدري",
)

#: Phrases that mean the caller is asking the agent to interpret something
#: medical, or to judge whether a procedure is safe for them. Not an emergency,
#: but never the agent's to answer.
#:
#: **Strong**: medical whatever else the sentence says. An implant, a
#: pregnancy or a request for a judgement outranks any logistics phrasing
#: around it — "je suis enceinte, que dois-je faire avant l'examen ?" is a
#: safety question wearing a logistics coat.
STRONG_ADVICE_PATTERNS: tuple[str, ...] = (
    # Implants and contraindications
    r"pace[\s-]?maker",
    r"\bpacemaker\b",
    r"\bimplant",
    r"\bprothèse",
    r"\bstimulateur\s+cardiaque",
    # Pregnancy and breastfeeding
    r"\benceinte\b",
    r"\bgrossesse\b",
    r"j'?allaite",
    r"حامل",
    r"\b7amla?\b",
    # Asking for a judgement
    r"c'?est\s+grave",
    r"c'?est\s+(?:un\s+)?cancer",
    r"c'?est\s+dangereux",
    r"est[\s-]?(?:ce\s+)?(?:que\s+)?(?:c'?est|il\s+est|elle\s+est)?\s*dangereu",
    r"\bdangereux\b|\bdangereuse\b",
    r"\brisqué\b|\bris[qc]ue\s+pour\b",
    r"\bnormal\s*\?",
    r"est\s+normal",
    r"ça\s+veut\s+dire\s+quoi",
    r"(?:veut|voulez)\s+dire\s+quoi",
    r"qu'?est[\s-]ce\s+que\s+(?:ça|ca)\s+veut\s+dire",
    # Findings named outright
    r"\bnodule\b|\btumeur\b|\bmasse\b|\blésion\b|\bhernie\b",
    r"\bdiagnostic\b",
    # Medication and allergies
    r"\bmédicament",
    r"(?:arrêter|continuer|prendre)\s+mon\s+traitement",
    r"\ballergi",
    r"دواء",
    r"حساسية",
    r"خطير",
    r"يوجع",
)

#: **Weak**: medical only when the sentence is not asking a logistics question.
#: "Vous pouvez m'expliquer mon compte rendu ?" is a request to interpret;
#: "Où est-ce que je prends mon compte rendu ?" is a request for a counter.
#: The same noun, two different calls — which is exactly why a single
#: `results` intent could never have worked.
WEAK_ADVICE_PATTERNS: tuple[str, ...] = (
    r"compte[\s-]rendu",
    r"expliquer\s+(?:mon|le|les|ce)",
    r"\brésultat|\bresultat",
    r"\bimages?\s+montre",
)

#: Logistics phrasings that mention results without asking what they mean.
#: "Comment récupérer mes résultats ?" is `info`, and must not be swept up by
#: the rule that catches "mon résultat est normal ?". This is the distinction a
#: single `results` intent would have destroyed.
RESULTS_LOGISTICS_PATTERNS: tuple[str, ...] = (
    # Verb stems rather than infinitives, so "je prends" counts as well as
    # "prendre" — the conjugation is what a caller actually says.
    r"(?:récup|recup|retir|cherch|prend|pren|avoir|envoi|envoy|transmis|"
    r"transmett|télécharg|telecharg|reçoi|recevoir|donn|remet|voir|consult|"
    r"accéder|acced)\w*\s+"
    r"(?:\w+\s+)?(?:mes|les|mon|le|un)\s+"
    r"(?:résultat|resultat|cd|compte[\s-]rendu|document|image)",
    r"où\s+(?:est[\s-]ce\s+)?(?:que\s+)?(?:je\s+)?\w*\s*"
    r"(?:mes|mon|le|les)\s+(?:résultat|resultat|cd|compte[\s-]rendu)",
    # Word order varies: "quand sont prêts mes résultats" and "mes résultats
    # sont prêts quand" are the same question asked two ways.
    # "-ils" covers the inverted question form: "les résultats sont-ils prêts".
    r"(?:sont|seront|est|sera)(?:[\s-]\w+)?\s+(?:prêt|pret|disponible)",
    r"quand\s+(?:sont|seront|est|sera)\b",
    r"combien\s+de\s+(?:jours|temps)\s+pour",
    r"par\s+mail\b|par\s+email\b|par\s+la\s+poste",
    r"(?:transmis|envoyé)\s+(?:directement\s+)?(?:à|a|au)\s+mon\s+médecin",
)


def urgency_keywords() -> frozenset[str]:
    """Return the phrases that mean a caller may be having an emergency."""
    return frozenset(URGENCY_PATTERNS)


def advice_keywords() -> frozenset[str]:
    """Return the phrases that mean the caller is asking for medical advice."""
    return frozenset(STRONG_ADVICE_PATTERNS + WEAK_ADVICE_PATTERNS)


def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace, keeping accents and Arabic script."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _matches(text: str, patterns: tuple[str, ...]) -> str | None:
    """Return the first pattern that matches, or None."""
    folded = _normalise(text)
    for pattern in patterns:
        if re.search(pattern, folded):
            return pattern
    return None


def detect_urgency(text: str) -> bool:
    """Return whether an utterance describes a possible medical emergency."""
    return _matches(text, URGENCY_PATTERNS) is not None


def detect_advice_request(text: str) -> bool:
    """Return whether an utterance asks for a diagnosis or medical advice.

    A strong pattern decides on its own. A weak one only counts when the
    sentence is not asking a logistics question: asking where to collect a CD
    is not asking what the CD says.
    """
    if _matches(text, STRONG_ADVICE_PATTERNS):
        return True
    if not _matches(text, WEAK_ADVICE_PATTERNS):
        return False
    return _matches(text, RESULTS_LOGISTICS_PATTERNS) is None


def medical_rule_hit(text: str) -> str | None:
    """Return which rule fired on this utterance, or None.

    One call for both layers: the gate uses it before routing, and the
    post-LLM check uses it again on whatever the model produced.
    """
    if detect_urgency(text):
        return "urgency"
    if detect_advice_request(text):
        return "advice"
    return None


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
