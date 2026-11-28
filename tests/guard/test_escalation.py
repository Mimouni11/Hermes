"""Tests for the deterministic medical rules.

These run before any routing and again after the LLM, and they are the layer
that catches what a 78.9%-accurate classifier leaks. They need no model, no
server and no network, which is the point: the safety net must not depend on
anything that can be unavailable.

The hard case is the one these tests spend most of their effort on — asking
*where* to collect a result is logistics, asking *what it says* is a diagnosis
request, and the two share every noun.
"""

from __future__ import annotations

import pytest

from hermes.guard import escalation


# --------------------------------------------------------------------------
# Emergencies
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "J'ai une douleur à la poitrine, qu'est-ce que je fais ?",
        "J'ai mal à la poitrine depuis ce matin.",
        "Je n'arrive pas à respirer.",
        "J'ai du mal à respirer.",
        "Je saigne beaucoup.",
        "Il a fait une perte de connaissance.",
        "Je me suis évanoui hier.",
        "C'est une urgence !",
    ],
)
def test_emergencies_are_detected(text: str) -> None:
    """A caller describing an emergency is never handled by the bot."""
    assert escalation.detect_urgency(text) is True
    assert escalation.medical_rule_hit(text) == "urgency"


@pytest.mark.parametrize(
    "text",
    [
        "Je voudrais un rendez-vous pour une IRM du genou.",
        "Quels sont vos horaires ?",
        "Je dois annuler mon rendez-vous de jeudi.",
    ],
)
def test_ordinary_requests_are_not_emergencies(text: str) -> None:
    """Routine logistics must not trip the emergency script."""
    assert escalation.detect_urgency(text) is False


# --------------------------------------------------------------------------
# Advice and contraindications
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "J'ai un pacemaker, je peux faire l'IRM ?",
        "J'ai un pace-maker, c'est compatible ?",
        "J'ai un implant cochléaire.",
        "Je suis enceinte, est-ce que la radio est dangereuse pour le bébé ?",
        "Mon scanner montre un nodule, c'est grave ?",
        "C'est un cancer ?",
        "Est-ce que mon résultat est normal ?",
        "Le compte rendu dit hernie discale, ça veut dire quoi ?",
        "Vous pouvez m'expliquer mon compte rendu ?",
        "Je suis allergique à l'iode, c'est risqué ?",
        "Est-ce que je peux arrêter mon traitement avant le scanner ?",
    ],
)
def test_advice_requests_are_detected(text: str) -> None:
    """Anything asking the agent to judge or interpret reaches a human."""
    assert escalation.detect_advice_request(text) is True
    assert escalation.medical_rule_hit(text) is not None


def test_the_three_classifier_leaks_are_caught() -> None:
    """The specific sentences the classifier routes to `info`.

    These are measured, not invented: cross-validating the shipped classifier
    leaks exactly these three out of 39 medical questions. The rules closing
    this gap is what makes the combined coverage complete, so a regression
    here is a safety regression.
    """
    leaks = [
        "Je suis enceinte, est-ce que la radio est dangereuse pour le bébé ?",
        "Est-ce que mon résultat est normal ?",
        "Vous pouvez m'expliquer mon compte rendu ?",
    ]
    for text in leaks:
        assert escalation.medical_rule_hit(text) == "advice", text


# --------------------------------------------------------------------------
# The distinction a single `results` label would have destroyed
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Comment récupérer mes résultats ?",
        "Je récupère mon CD où ?",
        "Où est-ce que je prends mon compte rendu ?",
        "Mes résultats sont prêts quand ?",
        "Quand les résultats sont-ils prêts ?",
        "Vous envoyez les résultats par mail ?",
        "Le compte rendu peut être transmis directement à mon médecin ?",
        "Je peux voir mes résultats sur internet ?",
        "Combien de jours pour avoir le compte rendu du scanner ?",
    ],
)
def test_results_logistics_is_not_medical(text: str) -> None:
    """Asking where to collect a result is not asking what it says."""
    assert escalation.detect_advice_request(text) is False
    assert escalation.medical_rule_hit(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "Je suis enceinte, que dois-je faire avant l'examen ?",
        "Je suis enceinte, que dois-je vous signaler ?",
        "Je suis claustrophobe, comment ça se passe pour l'IRM ?",
        "Faut-il retirer mes bijoux pour l'IRM ?",
        "Je dois enlever ma montre ?",
        "Est-ce que l'IRM fait mal ?",
        "Je dois me déshabiller pour l'examen ?",
    ],
)
def test_centre_procedure_is_not_blocked(text: str) -> None:
    """Tier A: the centre has written these answers, so the agent may read them.

    This is the half of the split that changed. These used to be blocked by
    the bare words "enceinte" and "claustrophobe", which meant the `grossesse`
    and `claustrophobie` FAQ entries existed and could never be reached —
    the agent transferred callers to a human to be told something it already
    had in writing.
    """
    assert escalation.is_absolute(text) is False
    assert escalation.medical_rule_hit(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "Je suis enceinte, est-ce que la radio est dangereuse pour le bébé ?",
        "Enceinte de deux mois, le scanner est risqué ?",
        "La grossesse pose un problème ?",
        "Je pense être enceinte, les rayons peuvent faire du mal au bébé ?",
        "Je suis claustrophobe, vous pouvez me donner un calmant ?",
    ],
)
def test_the_risk_half_of_the_same_topic_still_transfers(text: str) -> None:
    """Tier B: same subject, but the answer depends on this patient.

    The pair is the whole point of the split. "Que dois-je faire" is the
    centre's procedure; "est-ce dangereux pour le bébé" is a judgement about
    one pregnancy, and no document can make it.
    """
    assert escalation.is_absolute(text) is True


# --------------------------------------------------------------------------
# Plain logistics
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Je voudrais prendre un rendez-vous pour une mammographie.",
        "Vous êtes ouverts le samedi ?",
        "C'est combien une échographie ?",
        "Je dois déplacer mon rendez-vous de jeudi.",
        "Vous êtes où exactement ?",
        "Je peux payer par carte ?",
    ],
)
def test_logistics_never_escalates(text: str) -> None:
    """The rules must be quiet on the calls the agent exists to handle."""
    assert escalation.medical_rule_hit(text) is None


def test_empty_input_is_not_medical() -> None:
    """Nothing said is not an emergency."""
    assert escalation.medical_rule_hit("") is None
    assert escalation.medical_rule_hit("   ") is None


def test_detection_ignores_case() -> None:
    """Whisper's capitalisation is not a safety feature."""
    assert escalation.detect_advice_request("J'AI UN PACEMAKER") is True
    assert escalation.detect_urgency("C'EST UNE URGENCE") is True


def test_darija_patterns_are_present() -> None:
    """The Darija side is in the lists even though the corpus is French-first."""
    patterns = escalation.advice_keywords()

    assert any("حامل" in p for p in patterns)
    assert any("دواء" in p for p in patterns)


def test_keyword_accessors_return_every_pattern() -> None:
    """advice_keywords covers both the strong and the weak lists."""
    patterns = escalation.advice_keywords()

    assert set(escalation.STRONG_ADVICE_PATTERNS) <= patterns
    assert set(escalation.WEAK_ADVICE_PATTERNS) <= patterns
    assert escalation.urgency_keywords() == frozenset(escalation.URGENCY_PATTERNS)
