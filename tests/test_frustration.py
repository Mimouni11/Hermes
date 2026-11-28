"""Tests for frustration scoring: levels, scope, and graceful degradation.

Split the way the other modules' tests are. The lexicon, the folding and the
level logic need nothing on disk and always run. Anything that wants the
trained ensemble skips when `models/frustration.pt` is missing, so a fresh
clone still passes.

Accuracy is not asserted here. F1 0.93 is one measurement of one model fitted
to one corpus, and pinning it into the suite would turn adding a training row
into a test failure. It belongs in `docs/WP5.md`.
"""

from __future__ import annotations

import pytest

from hermes.understand import frustration
from hermes.config import get_settings


def _needs_model() -> None:
    """Skip unless the trained ensemble is on disk and loadable."""
    settings = get_settings()
    path = settings.paths.models / settings.frustration_model_file
    if not path.is_file():
        pytest.skip("no frustration model: scripts/train_frustration.py")
    frustration.reset_model()
    if frustration.load_model() is None:
        pytest.skip("frustration model present but not loadable")


# --------------------------------------------------------------------------
# Normalisation and the lexicon — no model needed
# --------------------------------------------------------------------------

def test_folding_strips_accents_and_apostrophes() -> None:
    """Terms and utterances are folded the same way, or nothing matches."""
    assert frustration.fold("Je RÉPÈTE !") == "je repete"
    assert frustration.fold("j'en ai marre") == "j en ai marre"
    assert frustration.fold("") == ""


@pytest.mark.parametrize("text", [
    "ça fait trois fois que je répète",
    "j'en ai marre de ce robot",
    "passez-moi quelqu'un",
    "ça ne marche pas du tout",
    "vous ne comprenez rien",
])
def test_the_lexicon_catches_explicit_complaints(text: str) -> None:
    assert frustration.lexicon_hits(text)


@pytest.mark.parametrize("text", [
    "Bonjour, je voudrais un rendez-vous",
    "merci beaucoup !",
    "c'est pour une IRM du genou",
])
def test_the_lexicon_is_quiet_on_ordinary_requests(text: str) -> None:
    assert frustration.lexicon_hits(text) == ()


def test_overlapping_terms_are_reported_once() -> None:
    """"je l'ai déjà dit" must not also report the "déjà dit" inside it."""
    hits = frustration.lexicon_hits("je l'ai déjà dit")

    assert "je l ai deja dit" in hits
    assert "deja dit" not in hits


def test_a_term_is_reported_with_its_family() -> None:
    assert "repetition" in frustration.lexicon_families("je répète")


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------

@pytest.mark.parametrize("lang", ["tn", "tn_latn", "mix", "ar", "en", ""])
def test_anything_but_french_is_declined(lang: str) -> None:
    """The model was trained and calibrated on 180 French rows only.

    Returning a number for Darija would be returning a number from a
    threshold that means nothing there. Measured reason in docs/WP5.md: 53 of
    the 56 `mix` corpus rows carry their frustration in Darija.
    """
    result = frustration.detect("yezzi, fadit mel robot", lang=lang)

    assert result.level == "none"
    assert result.source == "unsupported_lang"
    assert result.is_frustrated is False


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_nothing_said_is_not_frustration(text: str) -> None:
    result = frustration.detect(text)

    assert (result.level, result.source) == ("none", "empty")


# --------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------

def test_a_missing_model_degrades_instead_of_raising(tmp_path) -> None:
    """No weights on disk must not take the call down.

    The agent has to keep answering without this feature. A frustration
    score improves the handover; it is not a precondition for answering.

    Points `models/` at an empty directory through the real settings API
    rather than patching a private attribute — an earlier version of this
    test monkeypatched a name that may not exist, which made it pass
    whenever no model happened to be on disk and assert nothing once one was.
    """
    from dataclasses import replace

    from hermes import config

    settings = get_settings()
    empty = replace(settings, paths=replace(settings.paths, models=tmp_path))
    config.set_settings(empty)
    frustration.reset_model()
    try:
        assert get_settings().paths.models == tmp_path
        result = frustration.detect("ça fait trois fois que je répète !")
        assert result.level == "none"
        assert result.source == "no_model", result.source
        # The lexicon still reports, so the turn log is not blind.
        assert result.lexicon_hits
    finally:
        config.reset_settings()
        frustration.reset_model()


def test_a_model_from_another_embedder_is_refused() -> None:
    """384 numbers wide either way, so a mismatch loads cleanly and lies.

    Refusing is the whole point: a wrong-embedder model produces plausible
    probabilities from the wrong vector space, which is worse than an error.
    """
    _needs_model()
    meta = frustration.model_metadata()

    assert meta.get("embedder") == get_settings().embedding_model


# --------------------------------------------------------------------------
# The decision rule, with no model involved
# --------------------------------------------------------------------------

def test_the_level_rule_is_the_hybrid() -> None:
    """`level_for` is the shipped rule, and both arms reach `hard`.

    Pure function, no model, so the rule is pinned independently of whatever
    the current weights happen to score.
    """
    rule = frustration.level_for

    assert rule(0.95, False, 0.62, 0.86) == "hard"      # probability alone
    assert rule(0.70, True, 0.62, 0.86) == "hard"       # lexicon lifts it
    assert rule(0.70, False, 0.62, 0.86) == "soft"      # same prob, no term
    assert rule(0.40, True, 0.62, 0.86) == "none"       # term below soft
    assert rule(0.10, False, 0.62, 0.86) == "none"


def test_a_lexicon_hit_below_soft_does_not_lift_anything() -> None:
    """The word list never acts on a sentence the model disagrees with.

    This is the gate. Without it the lexicon's own false alarm — "Vous pouvez
    répéter s'il vous plaît ?" — would transfer a caller who merely asked the
    agent to say something again.
    """
    assert frustration.level_for(0.05, True, 0.62, 0.86) == "none"


# --------------------------------------------------------------------------
# The cases from the specification
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("calm", "annoyed"), [
    ("ça marche", "ça ne marche pas"),
    ("c'est bon comme ça", "ce n'est pas bon du tout"),
    ("oui j'ai compris", "non je n'ai pas compris"),
])
def test_polarity_pairs_separate(calm: str, annoyed: str) -> None:
    """One function word apart, opposite meaning.

    The pair that first exposed this — "Oui ça marche !" scoring 0.943 — made
    it look like a general blindness to negation. Measured on fresh pairs it
    is not: the separation holds. So the assertion is on the *ordering*, not
    on either side clearing a threshold.
    """
    _needs_model()
    low = frustration.detect(calm)
    high = frustration.detect(annoyed)

    assert low.prob < high.prob, f"{low.prob:.3f} vs {high.prob:.3f}"
    assert low.level == "none"


def test_an_explicit_repeated_complaint_is_flagged() -> None:
    """The case the whole feature exists for."""
    _needs_model()
    result = frustration.detect("ça fait trois fois que je répète")

    assert result.level in ("soft", "hard"), result
    assert result.source in ("model", "model+lexicon")


def test_a_lexicon_hit_can_reach_hard() -> None:
    """The hybrid's second arm, end to end through the real model."""
    _needs_model()
    result = frustration.detect(
        "j'en ai marre, ça fait trois fois que je répète, je vais raccrocher"
    )

    assert result.lexicon_hits
    assert result.level == "hard", f"scored {result.prob:.3f}"


@pytest.mark.parametrize("text", [
    "non, j'ai dit mardi",
    "pardon, je me suis trompé de date",
])
def test_a_correction_stays_below_hard(text: str) -> None:
    """Correcting yourself is the most common thing a caller does.

    It is allowed to reach `soft` — an apology costs nothing — but it must
    not transfer the call.
    """
    _needs_model()
    result = frustration.detect(text)

    assert result.level != "hard", f"{text!r} scored {result.prob:.3f}"


@pytest.mark.parametrize("text", [
    "j'ai peur que ce soit grave",
    "merci beaucoup !!",
])
def test_worry_and_thanks_are_not_frustration(text: str) -> None:
    """Worry is about the exam, not about us; gratitude is the opposite."""
    _needs_model()
    result = frustration.detect(text)

    assert result.level == "none", f"{text!r} scored {result.prob:.3f}"


# --------------------------------------------------------------------------
# The level contract
# --------------------------------------------------------------------------

def test_the_thresholds_decide_the_level() -> None:
    """Levels come from config, not from constants in the module."""
    _needs_model()
    settings = get_settings()

    assert settings.frustration_soft_threshold < settings.frustration_hard_threshold
    for text in ["je n'en peux plus de ce robot", "bonjour"]:
        result = frustration.detect(text)
        if result.prob >= settings.frustration_hard_threshold:
            assert result.level == "hard"
        elif result.prob >= settings.frustration_soft_threshold:
            assert result.level == "soft"
        else:
            assert result.level == "none"


def test_the_score_is_a_probability() -> None:
    _needs_model()
    result = frustration.detect("c'est inadmissible, passez-moi quelqu'un")

    assert 0.0 <= result.prob <= 1.0
    assert result.level in frustration.LEVELS


def test_the_pipeline_hook_records_without_acting() -> None:
    """Stage 6 returns a score and takes no action — WP6 owns the policy."""
    from hermes import pipeline

    result = pipeline.score_frustration("yezzi", lang="tn")

    assert isinstance(result, frustration.FrustrationScore)
    assert result.source == "unsupported_lang"

