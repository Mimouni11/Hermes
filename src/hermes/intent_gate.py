"""First routing decision of every turn: what does the caller want, if anything.

Embeds the caller's utterance and compares it against the labelled examples
indexed in Qdrant, routing to one of the six intents the corpus defines — or
refusing the turn as out of scope. This is the gate that keeps a medical
imaging call line from answering questions it has no business answering.

**The intents are the corpus's, not an invention.** An earlier sketch of this
module named its own set (`faq`, `book`, `reschedule`, `results`, …) which did
not match `data/raw/intents.csv`, and had no `medical_question` at all — the
one category the brief makes safety-critical. A `results` label was the other
hazard: "where do I pick up my results" is logistics and "what do my results
mean" is a diagnosis request, and a single label cannot hold both. The labels
here are exactly the six in the data.

Routing is nearest-neighbour rather than centroid. A centroid averages "je veux
annuler" with "je dois déplacer mon rendez-vous de jeudi" and lands near
neither; the neighbours keep both.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from hermes.config import get_settings


class Intent(str, Enum):
    """The six labels in `data/raw/intents.csv`, plus the two verdicts.

    ``OFF_TOPIC`` and ``OUT_OF_SCOPE`` are different things and both are
    needed. Off-topic is a *recognised* category — the corpus has 35 examples
    of callers asking about gym memberships and vaccinations — and the agent
    knows what it is and declines it. Out of scope is the gate admitting it
    does not know what the caller said at all.
    """

    NEW_APPOINTMENT = "new_appointment"
    EXISTING_APPOINTMENT = "existing_appointment"
    INFO = "info"
    MEDICAL_QUESTION = "medical_question"
    HUMAN_REQUEST = "human_request"
    OFF_TOPIC = "off_topic"

    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN = "unknown"

    @classmethod
    def from_label(cls, label: str) -> "Intent":
        """Map a corpus label onto an intent, falling back to UNKNOWN."""
        try:
            return cls((label or "").strip())
        except ValueError:
            return cls.UNKNOWN


#: The intents the agent handles itself. Everything else reaches a human.
HANDLED = {
    Intent.NEW_APPOINTMENT,
    Intent.EXISTING_APPOINTMENT,
    Intent.INFO,
}

#: Intents that go straight to a human, however confident the gate is.
#:
#: ``MEDICAL_QUESTION`` is here because the brief requires it: the system never
#: judges severity, urgency or clinical relevance. Confidence is irrelevant —
#: a *correctly* identified medical question is exactly the one to hand over.
ALWAYS_TRANSFER = {
    Intent.MEDICAL_QUESTION,
    Intent.HUMAN_REQUEST,
    Intent.OFF_TOPIC,
}


class Action(str, Enum):
    """What the pipeline should do with the turn."""

    #: Hand the turn to the branch that owns the intent.
    ROUTE = "route"
    #: Two intents are too close to call; ask which one they mean.
    CLARIFY = "clarify"
    #: Nothing matched well enough; ask them to say it differently, once.
    REPHRASE = "rephrase"
    #: Give the turn to a human.
    TRANSFER = "transfer"


@dataclass(frozen=True)
class Neighbour:
    """One labelled example retrieved from the index."""

    text: str
    intent: Intent
    lang: str
    score: float


@dataclass(frozen=True)
class IntentDecision:
    """The gate's verdict on one utterance.

    ``score`` and ``probability`` answer different questions and both are
    kept. The score is the cosine similarity to the nearest labelled example,
    which is what decides whether the caller said anything the agent has seen
    before. The probability is the classifier's confidence in *which* intent,
    which is only meaningful once the first question is settled.
    """

    intent: Intent
    action: Action
    score: float
    probability: float
    margin: float
    runner_up: Intent | None
    in_scope: bool
    reason: str
    scores: dict[Intent, float]
    probabilities: dict[Intent, float]
    neighbours: tuple[Neighbour, ...]

    @property
    def needs_human(self) -> bool:
        """Whether this turn ends with a transfer."""
        return self.action is Action.TRANSFER


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def neighbours_for(
    text: str,
    top_k: int | None = None,
    lang: str | None = None,
) -> list[Neighbour]:
    """Return the nearest labelled examples, best first."""
    from hermes import embeddings, vector_store

    settings = get_settings()
    hits = vector_store.search(
        settings.intent_collection,
        embeddings.encode_one(text),
        limit=top_k or settings.intent_top_k,
        where={"lang": lang} if lang else None,
    )
    return [
        Neighbour(
            text=str(hit.get("text", "")),
            intent=Intent.from_label(str(hit.get("intent", ""))),
            lang=str(hit.get("lang", "")),
            score=hit.score,
        )
        for hit in hits
    ]


def is_too_short(text: str) -> bool:
    """Whether an utterance is too short to be classified by similarity.

    Not a stylistic rule — a measured one. On the out-of-scope set, "oui",
    "hmm", "euh..." and "allô ?" all retrieve "Opérateur !" at about 0.78,
    scoring higher than almost every genuine in-scope example. Sentence
    embeddings put short strings near other short strings whatever they mean,
    so a similarity threshold cannot separate them and should not be asked to.
    They are caught here, before anything is embedded.

    A short reply *inside* a booking ("oui", "le deuxième") is a different
    thing and belongs to dialogue state, which reads it in the context of the
    question just asked rather than routing it from scratch.
    """
    settings = get_settings()
    stripped = (text or "").strip()
    if not stripped:
        return True

    words = [w for w in stripped.split() if any(c.isalnum() for c in w)]
    letters = sum(1 for c in stripped if c.isalpha())
    return (
        len(words) < settings.min_utterance_words
        or letters < settings.min_utterance_letters
    )


def score_intents(neighbours: list[Neighbour]) -> dict[Intent, float]:
    """Score each intent by its *best* neighbour, not by a vote count.

    A count would let four mediocre neighbours outvote one near-identical
    sentence, which is the wrong answer when the corpus is unevenly spread —
    and it is: `new_appointment` has 49 examples against `off_topic`'s 35.
    Taking the best similarity per intent keeps the scores on the same scale as
    the threshold, which is a raw cosine.
    """
    best: dict[Intent, float] = {}
    for neighbour in neighbours:
        if neighbour.score > best.get(neighbour.intent, -1.0):
            best[neighbour.intent] = neighbour.score
    return best


# --------------------------------------------------------------------------
# The classifier
# --------------------------------------------------------------------------

_CLASSIFIER: dict[str, Any] = {}


def load_classifier(path: Any = None) -> dict[str, Any]:
    """Load and cache the fitted intent classifier.

    Routing is a trained classifier rather than the nearest neighbours,
    because it is measurably better on this corpus: 78.9% accuracy against
    kNN's 68.9%, and 92.3% recall on medical questions against 79.5%. The
    neighbours are still retrieved — they decide scope, and they are what the
    bench shows a human when a route looks wrong.
    """
    from pathlib import Path

    settings = get_settings()
    target = Path(path) if path else settings.paths.models / settings.intent_model_file
    key = str(target)

    if key not in _CLASSIFIER:
        if not target.exists():
            raise RuntimeError(
                f"no intent classifier at {target}. Build it with:\n"
                "    uv run python scripts/train_intent_classifier.py"
            )
        import joblib

        _CLASSIFIER[key] = joblib.load(target)
    return _CLASSIFIER[key]


def predict_probabilities(text: str) -> dict[Intent, float]:
    """Return the classifier's probability for every intent."""
    from hermes import embeddings

    bundle = load_classifier()
    model = bundle["model"]
    vector = embeddings.encode_one(text).reshape(1, -1)
    row = model.predict_proba(vector)[0]
    return {
        Intent.from_label(str(label)): float(probability)
        for label, probability in zip(model.classes_, row)
    }


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------

def classify(
    text: str,
    top_k: int | None = None,
    lang: str | None = None,
    threshold: float | None = None,
    margin: float | None = None,
) -> IntentDecision:
    """Route one utterance to an intent, or mark it out of scope.

    Three layers, cheapest and most certain first:

    1. **too short** — "oui", "hmm" and "allô ?" cannot be classified by
       similarity at all, and are rejected before anything is embedded;
    2. **the deterministic medical rules**, before any routing. They catch the
       pacemaker and pregnancy questions the classifier leaks, and they cost
       nothing;
    3. **scope**, from the nearest labelled example's cosine similarity. Below
       τ the caller said something with no precedent in the corpus;
    4. **the classifier**, with an asymmetric floor: if `medical_question`
       clears :attr:`Settings.medical_probability_floor` the turn is medical
       even when another intent scores higher. A false alarm costs a transfer;
       a miss costs safety;
    5. **confidence** — below the minimum probability the model is guessing
       between classes, so the agent asks rather than acts;
    6. the always-transfer intents, then routing.

    Layer 2 runs before scope on purpose. A medical emergency phrased in a way
    the corpus has never seen is still a medical emergency, and "I don't
    recognise this" is not an acceptable answer to it.
    """
    settings = get_settings()
    accept = settings.intent_accept_threshold if threshold is None else threshold
    least_gap = settings.intent_margin if margin is None else margin

    stripped = (text or "").strip()

    def decide(
        intent: Intent,
        action: Action,
        reason: str,
        in_scope: bool,
        score: float = 0.0,
        probability: float = 0.0,
        gap: float = 0.0,
        runner: Intent | None = None,
        scores: dict[Intent, float] | None = None,
        probabilities: dict[Intent, float] | None = None,
        found: tuple[Neighbour, ...] = (),
    ) -> IntentDecision:
        return IntentDecision(
            intent=intent,
            action=action,
            score=score,
            probability=probability,
            margin=gap,
            runner_up=runner,
            in_scope=in_scope,
            reason=reason,
            scores=scores or {},
            probabilities=probabilities or {},
            neighbours=found,
        )

    # 1. Too short to mean anything on its own.
    if is_too_short(stripped):
        return decide(
            Intent.OUT_OF_SCOPE, Action.REPHRASE,
            "empty" if not stripped else "too_short", False,
        )

    # 2. The deterministic medical rules, before any routing.
    from hermes import escalation

    rule = escalation.medical_rule_hit(stripped)
    if rule:
        return decide(
            Intent.MEDICAL_QUESTION, Action.TRANSFER, f"medical_rule:{rule}", True,
        )

    # 3. Scope, from the nearest labelled example.
    found = tuple(neighbours_for(stripped, top_k=top_k, lang=lang))
    scores = score_intents(list(found))
    if not scores:
        return decide(
            Intent.OUT_OF_SCOPE, Action.REPHRASE, "nothing_retrieved", False,
            found=found,
        )

    nearest = max(scores.values())
    if nearest < accept:
        return decide(
            Intent.OUT_OF_SCOPE, Action.REPHRASE, "below_threshold", False,
            score=nearest, scores=scores, found=found,
        )

    # 4. Which intent, and the asymmetric medical floor.
    probabilities = predict_probabilities(stripped)
    ranked = sorted(probabilities.items(), key=lambda pair: pair[1], reverse=True)
    top_intent, top_probability = ranked[0]
    runner_up, runner_probability = ranked[1] if len(ranked) > 1 else (None, 0.0)
    gap = top_probability - runner_probability

    common = {
        "score": nearest,
        "scores": scores,
        "probabilities": probabilities,
        "found": found,
        "runner": runner_up,
        "gap": gap,
    }

    medical = probabilities.get(Intent.MEDICAL_QUESTION, 0.0)
    if top_intent is not Intent.MEDICAL_QUESTION and (
        medical >= settings.medical_probability_floor
    ):
        return decide(
            Intent.MEDICAL_QUESTION, Action.TRANSFER, "medical_floor", True,
            probability=medical, **common,
        )

    # 5. Too unsure between classes to act.
    if top_probability < settings.intent_min_probability or gap < least_gap:
        return decide(
            top_intent, Action.CLARIFY, "ambiguous", True,
            probability=top_probability, **common,
        )

    # 6. Recognised, and either handed over or routed.
    if top_intent in ALWAYS_TRANSFER:
        return decide(
            top_intent, Action.TRANSFER, f"always_transfer:{top_intent.value}",
            True, probability=top_probability, **common,
        )

    return decide(
        top_intent, Action.ROUTE, "routed", True,
        probability=top_probability, **common,
    )


def is_out_of_scope(decision: IntentDecision) -> bool:
    """Return whether a decision should be refused rather than handled."""
    return not decision.in_scope or decision.intent is Intent.OUT_OF_SCOPE


# --------------------------------------------------------------------------
# What the agent says
# --------------------------------------------------------------------------

OUT_OF_SCOPE_REPLY = (
    "Je ne suis pas sûr d'avoir compris. Pouvez-vous reformuler votre demande ?"
)

CLARIFY_REPLY = (
    "Je veux être sûr de bien comprendre. Souhaitez-vous prendre un nouveau "
    "rendez-vous, ou modifier un rendez-vous existant ?"
)

TRANSFER_REPLIES: dict[Intent, str] = {
    Intent.MEDICAL_QUESTION: (
        "Je ne peux pas répondre aux questions médicales. Je vous passe "
        "quelqu'un qui pourra vous renseigner."
    ),
    Intent.HUMAN_REQUEST: "Bien sûr, je vous passe un collègue.",
    Intent.OFF_TOPIC: (
        "Je m'occupe uniquement des rendez-vous d'imagerie médicale. "
        "Je vous passe quelqu'un."
    ),
}

DEFAULT_TRANSFER_REPLY = "Je vous passe un collègue."


def out_of_scope_reply() -> str:
    """Return the line the agent speaks when a request is out of scope."""
    return OUT_OF_SCOPE_REPLY


def reply_for(decision: IntentDecision) -> str:
    """Return the line that matches a decision, or "" when the turn continues.

    An empty string means the gate has nothing to say: the turn was routed and
    the branch that owns the intent produces the reply.
    """
    if decision.action is Action.REPHRASE:
        return OUT_OF_SCOPE_REPLY
    if decision.action is Action.CLARIFY:
        return CLARIFY_REPLY
    if decision.action is Action.TRANSFER:
        return TRANSFER_REPLIES.get(decision.intent, DEFAULT_TRANSFER_REPLY)
    return ""
