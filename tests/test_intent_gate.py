"""Tests for the intent gate.

Split in two. The logic that needs nothing — the label mapping, the length
guard, the neighbour scoring — is always tested. The routing itself needs
Qdrant running and the classifier trained, so those tests skip rather than
fail on a machine where the index has not been built.

What the routing tests check is the *order of the layers*, not the accuracy.
Accuracy is measured by `scripts/calibrate_intent.py` against the whole corpus
and belongs in a report; a unit test asserting "78.9%" would just encode one
run of a model into the suite.
"""

from __future__ import annotations

import pytest

from hermes import intent_gate
from hermes.config import get_settings
from hermes.intent_gate import Action, Intent, Neighbour


def _needs_index() -> None:
    """Skip unless Qdrant is up, populated, and the classifier is trained."""
    from hermes import vector_store

    settings = get_settings()
    if not vector_store.ping():
        pytest.skip("Qdrant is not running: docker compose up -d")
    if vector_store.count(settings.intent_collection) == 0:
        pytest.skip("intent collection is empty: scripts/build_intent_index.py")
    if not (settings.paths.models / settings.intent_model_file).exists():
        pytest.skip("no classifier: scripts/train_intent_classifier.py")


# --------------------------------------------------------------------------
# The labels
# --------------------------------------------------------------------------

def test_the_six_corpus_labels_exist() -> None:
    """The enum is the data's, not an invention.

    An earlier version named its own intents and had no `medical_question`,
    which is the category the whole safety story rests on.
    """
    for label in [
        "new_appointment",
        "existing_appointment",
        "info",
        "medical_question",
        "human_request",
        "off_topic",
    ]:
        assert Intent.from_label(label).value == label


def test_an_unknown_label_does_not_explode() -> None:
    """A corpus row with a typo becomes UNKNOWN, not an exception."""
    assert Intent.from_label("booking") is Intent.UNKNOWN
    assert Intent.from_label("") is Intent.UNKNOWN


def test_off_topic_and_out_of_scope_are_different() -> None:
    """One is recognised and declined, the other is the gate admitting defeat.

    The asymmetry is the point. Knowing the answer is no — the centre does not
    do physiotherapy — means the agent can say so itself. *Not understanding*
    someone is a reason to involve a person, because a person might.
    """
    assert Intent.OFF_TOPIC is not Intent.OUT_OF_SCOPE
    assert Intent.OFF_TOPIC in intent_gate.ALWAYS_DECLINE
    assert Intent.OFF_TOPIC not in intent_gate.ALWAYS_TRANSFER
    assert Intent.OUT_OF_SCOPE not in intent_gate.ALWAYS_DECLINE
    assert Intent.OUT_OF_SCOPE not in intent_gate.ALWAYS_TRANSFER


def test_a_complaint_still_reaches_a_person() -> None:
    """Declining off_topic must not swallow anyone who needs a human.

    It is safe only because the corpus labels complaints `human_request`.
    If that ever changed, declining would start stranding callers.
    """
    assert Intent.HUMAN_REQUEST in intent_gate.ALWAYS_TRANSFER
    assert Intent.HUMAN_REQUEST not in intent_gate.ALWAYS_DECLINE


def test_declining_offers_the_service_it_does_have() -> None:
    """A decline ends with what the centre does, not just what it doesn't."""
    line = intent_gate.DECLINE_REPLIES[Intent.OFF_TOPIC]

    assert "imagerie" in line
    assert "IRM" in line
    assert "aider" in line


def test_medical_questions_always_transfer() -> None:
    """The brief's guiding principle, encoded."""
    assert Intent.MEDICAL_QUESTION in intent_gate.ALWAYS_TRANSFER
    assert Intent.MEDICAL_QUESTION not in intent_gate.HANDLED


# --------------------------------------------------------------------------
# The length guard
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["oui", "hmm", "euh...", "allô ?", "", "   ", "?"])
def test_short_utterances_are_guarded(text: str) -> None:
    """These retrieved "Opérateur !" at ~0.78 — higher than real examples."""
    assert intent_gate.is_too_short(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "je veux un rendez-vous",
        "vous ouvrez quand ?",
        "c'est combien ?",
    ],
)
def test_real_sentences_pass_the_guard(text: str) -> None:
    """The guard must not eat short but genuine requests."""
    assert intent_gate.is_too_short(text) is False


def test_the_guard_reads_its_thresholds_from_config() -> None:
    """Proof the numbers are configuration, not constants in the module."""
    from hermes import config

    assert intent_gate.is_too_short("un mot") is False
    relaxed = config.with_overrides(min_utterance_words=5)
    config.set_settings(relaxed)
    try:
        assert intent_gate.is_too_short("un mot") is True
    finally:
        config.reset_settings()


# --------------------------------------------------------------------------
# Scoring the neighbours
# --------------------------------------------------------------------------

def test_an_intent_scores_by_its_best_neighbour() -> None:
    """Best, not a vote: one near-identical sentence beats four mediocre ones.

    The corpus is unevenly spread — `info` has 87 French examples against
    `human_request`'s 14 — so counting neighbours would hand the decision to
    whichever class happens to be largest.
    """
    neighbours = [
        Neighbour("a", Intent.INFO, "fr", 0.60),
        Neighbour("b", Intent.INFO, "fr", 0.55),
        Neighbour("c", Intent.INFO, "fr", 0.52),
        Neighbour("d", Intent.NEW_APPOINTMENT, "fr", 0.88),
    ]

    scores = intent_gate.score_intents(neighbours)

    assert scores[Intent.NEW_APPOINTMENT] == pytest.approx(0.88)
    assert scores[Intent.INFO] == pytest.approx(0.60)
    assert max(scores, key=lambda i: scores[i]) is Intent.NEW_APPOINTMENT


def test_scoring_an_empty_neighbourhood() -> None:
    """No neighbours is an empty mapping, not a crash."""
    assert intent_gate.score_intents([]) == {}


# --------------------------------------------------------------------------
# The layers, in order
# --------------------------------------------------------------------------

def test_a_short_utterance_never_reaches_the_index() -> None:
    """Layer 1 answers without Qdrant or the classifier being available."""
    decision = intent_gate.classify("hmm")

    assert decision.action is Action.REPHRASE
    assert decision.reason == "too_short"
    assert decision.in_scope is False
    assert decision.neighbours == ()


def test_an_empty_utterance_is_reported_as_empty() -> None:
    """Distinguished from "too short" because they are different problems."""
    assert intent_gate.classify("").reason == "empty"


def test_the_medical_rule_fires_before_any_routing() -> None:
    """Layer 2 needs neither the index nor the classifier.

    This is the test that matters most: the rule has to beat the classifier
    even when the classifier would have been confident about something else.
    """
    decision = intent_gate.classify("J'ai un pacemaker, je peux faire l'IRM ?")

    assert decision.intent is Intent.MEDICAL_QUESTION
    assert decision.action is Action.TRANSFER
    assert decision.reason.startswith("medical_rule:")
    assert decision.needs_human is True


def test_an_emergency_transfers_even_if_unrecognisable() -> None:
    """A medical emergency phrased oddly is still a medical emergency.

    Layer 2 runs before the scope check on purpose: "I don't recognise this"
    is not an acceptable answer to someone describing chest pain.
    """
    decision = intent_gate.classify(
        "Zzz qwerty j'ai une douleur à la poitrine zzz"
    )

    assert decision.action is Action.TRANSFER
    assert decision.reason == "medical_rule:urgency"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Je voudrais prendre un rendez-vous pour une IRM du genou.",
         Intent.NEW_APPOINTMENT),
        ("Je dois annuler mon rendez-vous de jeudi.",
         Intent.EXISTING_APPOINTMENT),
        ("Quels sont vos horaires d'ouverture ?", Intent.INFO),
        ("Passez-moi quelqu'un s'il vous plaît.", Intent.HUMAN_REQUEST),
    ],
)
def test_clear_requests_route_to_their_intent(text: str, expected: Intent) -> None:
    """The ordinary path, on sentences with no ambiguity."""
    _needs_index()
    decision = intent_gate.classify(text)

    assert decision.intent is expected
    assert decision.in_scope is True


def test_a_handled_intent_is_routed_not_transferred() -> None:
    """A booking is the agent's job; it must not hand it over."""
    _needs_index()
    decision = intent_gate.classify(
        "Je voudrais prendre un rendez-vous pour une IRM du genou."
    )

    assert decision.action is Action.ROUTE
    assert decision.needs_human is False


def test_a_request_for_a_human_is_transferred() -> None:
    """Recognised and handed over, which is not the same as not understood."""
    _needs_index()
    decision = intent_gate.classify("Je veux parler à une vraie personne.")

    assert decision.action is Action.TRANSFER
    assert decision.in_scope is True


def test_something_unrelated_is_out_of_scope() -> None:
    """Below τ the gate says so rather than guessing an intent."""
    _needs_index()
    decision = intent_gate.classify(
        "Je voudrais vendre ma voiture d'occasion au meilleur prix."
    )

    assert decision.action in (Action.REPHRASE, Action.TRANSFER, Action.CLARIFY)
    assert decision.action is not Action.ROUTE


def test_a_decision_carries_both_numbers() -> None:
    """Similarity and probability answer different questions; both are kept."""
    _needs_index()
    decision = intent_gate.classify("Vous êtes ouverts le samedi matin ?")

    assert 0.0 <= decision.score <= 1.0
    assert 0.0 <= decision.probability <= 1.0
    assert decision.scores
    assert decision.probabilities
    assert decision.neighbours


def test_the_threshold_can_be_overridden_per_call() -> None:
    """For sweeps: calibration needs to move τ without touching config."""
    _needs_index()
    strict = intent_gate.classify("Vous êtes ouverts le samedi ?", threshold=0.99)

    assert strict.action is Action.REPHRASE
    assert strict.reason == "below_threshold"


# --------------------------------------------------------------------------
# What the agent says
# --------------------------------------------------------------------------

def test_every_action_has_something_to_say() -> None:
    """Except routing, where the branch that owns the intent replies."""
    from hermes.intent_gate import IntentDecision

    def decision(action: Action, intent: Intent) -> IntentDecision:
        return IntentDecision(
            intent=intent, action=action, score=0.7, probability=0.7,
            margin=0.1, runner_up=None, in_scope=True, reason="test",
            scores={}, probabilities={}, neighbours=(),
        )

    assert intent_gate.reply_for(decision(Action.REPHRASE, Intent.OUT_OF_SCOPE))
    assert intent_gate.reply_for(decision(Action.CLARIFY, Intent.INFO))
    assert intent_gate.reply_for(decision(Action.DECLINE, Intent.OFF_TOPIC))
    assert intent_gate.reply_for(decision(Action.TRANSFER, Intent.MEDICAL_QUESTION))
    assert intent_gate.reply_for(decision(Action.ROUTE, Intent.INFO)) == ""


def test_an_off_topic_call_is_declined_not_transferred() -> None:
    """The agent answers it itself rather than costing someone their time."""
    _needs_index()
    decision = intent_gate.classify("Je veux m'inscrire à la salle de sport.")

    assert decision.intent is Intent.OFF_TOPIC
    assert decision.action is Action.DECLINE
    assert decision.needs_human is False
    assert "imagerie" in intent_gate.reply_for(decision)


def test_the_medical_transfer_line_does_not_diagnose() -> None:
    """The words spoken on a medical transfer must not venture an opinion."""
    line = intent_gate.TRANSFER_REPLIES[Intent.MEDICAL_QUESTION]

    assert "médicale" in line
    assert "passe" in line
