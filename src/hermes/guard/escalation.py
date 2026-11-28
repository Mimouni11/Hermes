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
from typing import Any

from hermes.agent import AgentDecision, DialogueState
from hermes.understand.frustration import FrustrationScore
from hermes.understand.intent_gate import IntentDecision


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
#: **Tier B — absolute.** The answer depends on this caller's body, device or
#: history, so no document can supply it and no retrieval score may override
#: it. These transfer whatever else the sentence says and whatever the FAQ
#: scores.
#:
#: Note what is *not* here any more: the bare words "enceinte" and
#: "claustrophobe". The centre has written answers for the procedure those
#: callers are asking about, and blocking them meant the `grossesse` and
#: `claustrophobie` FAQ entries could never be reached. What stays absolute is
#: the *risk* phrasing — "est-ce dangereux pour le bébé" is a judgement about
#: one pregnancy, "que dois-je vous signaler" is the centre's own procedure.
ABSOLUTE_PATTERNS: tuple[str, ...] = (
    # Implants and contraindications: the answer depends on the device.
    r"pace[\s-]?maker",
    r"\bpacemaker\b",
    r"\bimplant",
    r"\bprothèse",
    r"\bstimulateur\s+cardiaque",
    r"\b(?:agrafes?|vis|plaques?|tiges?)\s+(?:\w+\s+)?"
    r"(?:métal|metal|métallique|chirurgical)",
    # Pregnancy and breastfeeding, but only where risk is being asked about.
    # The preposition varies: "dangereux pour le bébé", "mal au bébé".
    r"(?:dangereu|risqu|mal)\w*\s+(?:pour\s+|au\s+|à\s+|aux\s+)?"
    r"(?:le\s+|mon\s+|l'\s*)?bébé",
    r"(?:rayons?|radio|scanner|irm)\s+\w*\s*(?:dangereu|risqu)",
    # "La grossesse pose un problème ?" asks whether a pregnancy is a risk,
    # which is a judgement. "Je suis enceinte, que dois-je vous signaler ?"
    # asks the centre's procedure, which the grossesse FAQ entry answers.
    r"pose\s+(?:un\s+)?problème",
    r"(?:grossesse|enceinte)[^.?!]*\b(?:problème|contre[\s-]indic|empêche)",
    r"passe\s+dans\s+le\s+lait",
    r"j'?allaite",
    r"حامل",
    # Asking for a judgement about their own case.
    r"c'?est\s+grave",
    r"c'?est\s+(?:un\s+)?cancer",
    r"c'?est\s+dangereux",
    r"\bdangereux\b|\bdangereuse\b",
    r"\brisqué\b|\bris[qc]ue\s+pour\b",
    r"je\s+risque\s+quelque\s+chose",
    r"\bnormal\s*\?",
    r"est\s+normal",
    r"ça\s+veut\s+dire\s+quoi",
    r"(?:veut|voulez)\s+dire\s+quoi",
    r"qu'?est[\s-]ce\s+que\s+(?:ça|ca)\s+veut\s+dire",
    r"je\s+dois\s+m'?inquiéter",
    r"\binquiétant\b",
    # Asking what the images showed. "elle montre quoi" and "a vu une tache"
    # are interpretation requests that reached the FAQ and matched nonsense —
    # one scored 0.819 against "dois-je être à jeun".
    r"montre\s+quoi|montrent\s+quoi",
    r"(?:a|ont)\s+vu\s+(?:une?|des)\b",
    r"c'?est\s+(?:mauvais|bon|inquiétant|anormal)",
    r"\bmauvais\s+ou\s+pas\b",
    # Radiation dose is a clinical question, not a price list.
    r"combien\s+de\s+rayons",
    r"\bdose\s+de\s+rayons|\birradiation\b",
    # Findings named outright.
    r"\bnodule\b|\btumeur\b|\bmasse\b|\blésion\b|\bhernie\b|\bkyste\b",
    r"\barthrose\b|\bfracture\b|\bsclérose\b",
    r"\bdiagnostic\b",
    # Symptoms. Describing a body rather than asking about a procedure.
    r"j'?ai\s+mal\s+(?:à|a|au|aux)\b",
    r"\bvertiges?\b",
    r"\bfièvre\b",
    r"\bboule\s+au\b",
    r"\bsaigne",
    r"depuis\s+(?:hier|une\s+semaine|\w+\s+jours)",
    # Which exam to have: that is a prescription, not a booking.
    r"(?:quel|quels)\s+examens?\s+(?:je|dois|il\s+faut)",
    # "une IRM ou un scanner" — the article varies with the noun, and only
    # allowing "une" let "IRM ou un scanner pour mon dos" through to the FAQ.
    r"(?:scanner|irm|radio|échographie)\s+ou\s+(?:une?\s+)?"
    r"(?:irm|scanner|radio|échographie)",
    r"(?:suffit|suffisant)\s+(?:non|\?)",
    # Medication, allergies, organ function.
    r"\bmédicament",
    r"\bcalmant\b|\bsédatif\b|\banxiolytique\b",
    r"(?:arrêter|continuer|prendre)\s+(?:mes|mon)\s+(?:traitement|médicament)",
    r"\banticoagulant",
    r"\bdiabétique\b",
    r"\breins?\s+ne\s+marche|insuffisance\s+rénale",
    r"\ballergi",
    r"\bréaction\s+à\s+l'?iode",
    r"دواء",
    r"حساسية",
    r"خطير",
    r"يوجع",
)

#: Kept for callers that still import the old name.
STRONG_ADVICE_PATTERNS = ABSOLUTE_PATTERNS

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


#: Metal that is *inside* the caller. The object's name is deliberately not
#: part of the match: Whisper transcribed "broches" as "brosses", and a list of
#: implant nouns cannot be what a safety net depends on when the transcriber
#: mangles exactly these words. What carries the meaning is the metal being in
#: a body — "dans mes bras", "dans la tête" — whatever the noun was heard as.
METAL_IN_BODY_PATTERNS: tuple[str, ...] = (
    r"(?:métal|métallique|metal)\w*\s+dans\s+"
    r"(?:la|le|les|mon|ma|mes|leur|l'|un|une)\b",
    r"\bdans\s+(?:la|le|les|mon|ma|mes)\s+"
    r"(?:bras|tête|jambe|corps|genou|épaule|dos|hanche|crâne|poitrine|cou)\b",
    r"j'?ai\s+(?:des?|du|une?|plusieurs)\s+\w+\s+(?:en|de)\s+m[ée]tal",
    r"j'?ai\s+du\s+m[ée]tal",
    r"opéré\w*\s+(?:avec|et\s+j'?ai)",
)

#: Metal the caller is wearing, which comes off before the exam. A removal or
#: wearing verb beside one of these vetoes the in-body rule — "retirer mes
#: bijoux en métal" is the `metal_bijoux` FAQ entry, not a contraindication.
#: Without this veto the in-body patterns swallow the whole preparation topic,
#: which is how this rule broke the first time it was written.
REMOVABLE_METAL_PATTERNS: tuple[str, ...] = (
    r"(?:retir|enlev|ôt|garde|port|laiss)\w*\s+(?:\w+\s+){0,3}?"
    r"(?:bijou|montre|boucle|piercing|bague|collier|chaîne|lunette|"
    r"barrette|épingle|ceinture|soutien)",
    r"(?:bijou|montre|boucle|piercing|bague|collier|chaîne|lunette)\w*"
    r"[^.?!]*\b(?:retir|enlev|ôt|garde|port)",
)


def wears_removable_metal(text: str) -> bool:
    """Whether the sentence is about metal the caller takes off."""
    return _matches(text, REMOVABLE_METAL_PATTERNS) is not None


def has_metal_in_body(text: str) -> bool:
    """Whether the sentence describes metal inside the caller.

    Vetoed by :func:`wears_removable_metal`, because jewellery and an implant
    share every noun the sentence is likely to use.
    """
    if wears_removable_metal(text):
        return False
    return _matches(text, METAL_IN_BODY_PATTERNS) is not None


def is_absolute(text: str) -> bool:
    """Whether this is tier B: a question no document may answer.

    Implants, allergies, organ function, symptoms, result interpretation and
    "which exam should I have" all depend on the individual. A high-scoring
    FAQ match must not override this.
    """
    if has_metal_in_body(text):
        return True
    if wears_removable_metal(text) and not _matches(
        text, tuple(p for p in ABSOLUTE_PATTERNS if "métal" not in p)
    ):
        # Preparation: jewellery comes off, and nothing else in the sentence
        # asks for a judgement.
        return False
    return _matches(text, ABSOLUTE_PATTERNS) is not None


def detect_advice_request(text: str) -> bool:
    """Return whether an utterance asks for a diagnosis or medical advice.

    An absolute pattern decides on its own. A weak one only counts when the
    sentence is not asking a logistics question: asking where to collect a CD
    is not asking what the CD says.
    """
    if is_absolute(text):
        return True
    if not _matches(text, WEAK_ADVICE_PATTERNS):
        return False
    return _matches(text, RESULTS_LOGISTICS_PATTERNS) is None


def check_transcript(
    raw: str,
    repaired: str | None = None,
    safety_flags: tuple[Any, ...] = (),
) -> str | None:
    """Apply the medical rules to everything known about one turn.

    Checks the raw transcript **and** the repaired one, because the repair can
    go either way: it can restore a term the rules need ("écographie"), and it
    can in principle remove one. Either text matching is enough — the rules
    are a safety net, and a net is not improved by being choosy about which
    thread catches.

    A lexicon safety flag counts on its own. The flag means a word in the
    transcript is probably a safety term that was misheard, and "probably" is
    the right standard for deciding to involve a person.
    """
    for text in (raw, repaired):
        if not text:
            continue
        hit = medical_rule_hit(text)
        if hit:
            return hit
    if safety_flags:
        return "lexicon_flag"
    return None


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
