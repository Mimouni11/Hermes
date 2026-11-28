"""Tests for the lexicon repair layer.

The layer exists because Whisper mangles exactly the words that matter, and it
has to do two opposite things at once: rewrite a mistranscribed exam name, and
*never* rewrite a word that was right. The tests are mostly about the second.

They need the built lexicon (`scripts/lexicon/build_lexicon.py`) and skip without it,
so a fresh clone still passes.
"""

from __future__ import annotations

import pytest

from hermes.listen import lexicon
from hermes.config import get_settings


def _needs_lexicon() -> None:
    settings = get_settings()
    if not (settings.paths.processed / settings.lexicon_file).exists():
        pytest.skip("no lexicon: scripts/lexicon/build_lexicon.py")


# --------------------------------------------------------------------------
# Leaving things alone
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Quels sont vos horaires d'ouverture ?",
        "Je voudrais un rendez-vous pour une IRM du genou droit.",
        "Faut-il retirer mes bijoux pour l'IRM ?",
        "Je dois être à jeun pour l'échographie abdominale ?",
        "Vous êtes ouverts le samedi matin ?",
    ],
)
def test_a_clean_sentence_is_untouched(text: str) -> None:
    """Repair must be invisible when there is nothing to repair."""
    _needs_lexicon()
    result = lexicon.repair(text)

    assert result.repaired == text
    assert result.edits == ()


def test_repair_is_idempotent() -> None:
    """Repairing twice changes nothing the second time."""
    _needs_lexicon()
    once = lexicon.repair("Je dois être à jeu pour l'écographie abdominale ?")

    assert lexicon.repair(once.repaired).repaired == once.repaired


def test_empty_input_is_handled() -> None:
    """Nothing said is not a crash."""
    _needs_lexicon()
    for text in ("", "   ", "\n"):
        result = lexicon.repair(text)
        assert result.repaired == text
        assert result.edits == ()
        assert result.safety_flags == ()


def test_punctuation_and_case_survive() -> None:
    """The sentence is rebuilt from the original, not re-joined from tokens."""
    _needs_lexicon()
    text = "Bonjour ! L'écographie, c'est quand ? (merci)"

    repaired = lexicon.repair(text).repaired

    assert repaired.startswith("Bonjour ! L'")
    assert repaired.endswith("c'est quand ? (merci)")
    assert "," in repaired


# --------------------------------------------------------------------------
# Repairing domain words
# --------------------------------------------------------------------------

def test_a_mistranscribed_exam_name_is_repaired() -> None:
    """The measured failure: Whisper writes "écographie" for "échographie"."""
    _needs_lexicon()
    result = lexicon.repair("Je veux une écographie abdominale.")

    assert "échographie" in result.repaired
    assert result.changed
    assert result.edits[0].original.endswith("cographie")


def test_a_real_french_word_is_never_rewritten() -> None:
    """"brioche" is 0.86 similar to "broches" and is a pastry.

    This is the guard that matters: a false repair would invent an implant.
    """
    _needs_lexicon()
    for text in [
        "Je voudrais une brioche et un café.",
        "Où est ma brosse à dents ?",
        "On a trouvé un plan du centre.",
    ]:
        assert lexicon.repair(text).repaired == text, text


def test_an_edit_records_what_it_did() -> None:
    """Every rewrite is traceable: from, to, score and which rule fired."""
    _needs_lexicon()
    edits = lexicon.repair("Je veux une écographie.").edits

    assert edits
    edit = edits[0]
    assert edit.original and edit.replacement
    assert 0.0 < edit.score <= 1.0
    assert edit.rule in {"exact", "known_variant", "fuzzy"}


# --------------------------------------------------------------------------
# Safety flags
# --------------------------------------------------------------------------

def test_the_measured_failure_raises_a_flag() -> None:
    """"des brosses en métal dans le bras" is an implant, however it is spelled.

    This is the sentence Whisper actually produced from a real recording.
    Character similarity alone cannot catch it — "brosses ~ broches" scores
    0.71 while "brioches ~ broches" scores 0.93 — so the flag depends on the
    sentence being about imaging.
    """
    _needs_lexicon()
    result = lexicon.repair(
        "Bonjour, j'ai des brosses en métal dans mes bras, je peux faire une IRM ?"
    )

    assert "broches" in result.flagged_terms


def test_a_flag_never_rewrites() -> None:
    """Safety terms are flagged, never corrected. The transcript stands."""
    _needs_lexicon()
    text = "j'ai des brosses en métal dans le bras, une IRM est possible ?"

    result = lexicon.repair(text)

    assert result.safety_flags
    assert "brosses" in result.repaired
    assert "broches" not in result.repaired


def test_pastry_out_of_context_is_not_flagged() -> None:
    """The same word shape, with no imaging around it, is left alone."""
    _needs_lexicon()
    for text in [
        "Je voudrais une brioche et un café.",
        "Des brioches pour le petit-déjeuner.",
        "Où est ma brosse à dents ?",
    ]:
        assert lexicon.repair(text).flagged_terms == (), text


def test_removing_jewellery_is_not_a_safety_flag() -> None:
    """Preparation stays preparation, even with "IRM" in the sentence."""
    _needs_lexicon()
    result = lexicon.repair("Je retire mes bijoux avant l'IRM ?")

    assert result.flagged_terms == ()


def test_a_flag_records_its_evidence() -> None:
    """term, token, score and where the match came from."""
    _needs_lexicon()
    flags = lexicon.repair(
        "j'ai des brosses en métal dans le bras, IRM possible ?"
    ).safety_flags

    assert flags
    flag = flags[0]
    assert flag.term == "broches"
    assert flag.matched.lower() == "brosses"
    assert 0.0 < flag.score <= 1.0
    assert flag.source


def test_context_is_what_lowers_the_bar() -> None:
    """The same token flags with imaging context and not without it."""
    _needs_lexicon()
    with_context = lexicon.repair("des brosses en métal dans le bras")
    without = lexicon.repair("des brosses neuves dans le placard")

    assert with_context.flagged_terms == ("broches",)
    assert without.flagged_terms == ()


# --------------------------------------------------------------------------
# What the agent says about a flag
# --------------------------------------------------------------------------

def test_a_flag_produces_a_question_not_a_statement() -> None:
    """A flag is a suspicion. The agent checks it rather than asserting it."""
    _needs_lexicon()
    flags = lexicon.repair(
        "j'ai des brosses en métal dans le bras, IRM possible ?"
    ).safety_flags

    question = lexicon.confirmation_question(flags)

    assert question.endswith("?")
    assert "broches" in question
    assert "de des" not in question


def test_no_flags_means_nothing_to_ask() -> None:
    """Silence when there is no suspicion."""
    assert lexicon.confirmation_question(()) == ""


# --------------------------------------------------------------------------
# The French word guard
# --------------------------------------------------------------------------

def test_the_french_guard_separates_words_from_manglings() -> None:
    """Frequency, not mere attestation: "écographie" exists and is still wrong."""
    assert lexicon.is_french_word("brosse") is True
    assert lexicon.is_french_word("brioche") is True
    assert lexicon.is_french_word("alleraute") is False
    assert lexicon.is_french_word("écographie") is False


# --------------------------------------------------------------------------
# Phrase repair
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Je dois être à jeu pour l'échographie abdominale ?",
        "Faut-il être à jeu ?",
        "Je dois venir à jeu ?",
        "Il faut rester a jeu combien de temps ?",
    ],
)
def test_a_jeu_becomes_a_jeun(text: str) -> None:
    """The token repair cannot reach this; the phrase can.

    "jeun" is heard as "jeu", and "jeu" is a common French word that the same
    guard protecting "brioche" must also protect. The phrase is the way out:
    "à jeu" in a call to an imaging centre is never about a game.
    """
    _needs_lexicon()
    assert "à jeun" in lexicon.repair(text).repaired


@pytest.mark.parametrize(
    "text",
    [
        "Mon fils joue à un jeu vidéo.",
        "C'est un jeu d'enfant.",
        "Le jeu est terminé.",
        "Je cherche un jeu de société.",
    ],
)
def test_jeu_on_its_own_is_untouched(text: str) -> None:
    """The word stays protected; only the phrase is repaired."""
    _needs_lexicon()
    assert lexicon.repair(text).repaired == text


def test_a_phrase_repair_is_recorded_as_an_edit() -> None:
    """Traceable like any other edit, with its own rule name."""
    _needs_lexicon()
    edits = lexicon.repair("Je dois être à jeu ?").edits

    assert any(e.rule == "phrase" for e in edits)


# --------------------------------------------------------------------------
# Multi-word safety terms
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, term",
    [
        ("J'ai un stimulateur cardiake, je peux faire une IRM ?",
         "stimulateur cardiaque"),
        ("Le produit de contrasste pose un souci pour l'IRM ?",
         "produit de contraste"),
    ],
)
def test_a_multiword_safety_term_is_flagged(text: str, term: str) -> None:
    """Token-by-token matching structurally cannot see these.

    "cardiake" alone is nowhere near "stimulateur cardiaque", so the whole
    term was invisible until the window matched pairs of words. These were
    the only two safety cases the token pass missed, and both are two words.
    """
    _needs_lexicon()
    assert term in lexicon.repair(text).flagged_terms


def test_multiword_matching_does_not_invent_flags() -> None:
    """Scanning word pairs must not start flagging ordinary sentences."""
    _needs_lexicon()
    for text in [
        "Je voudrais une brioche et un café.",
        "Quels sont vos horaires d'ouverture ?",
        "Je cherche un plan de la ville.",
        "Le centre est ouvert le samedi matin.",
    ]:
        assert lexicon.repair(text).flagged_terms == (), text


def test_imaging_context_detection() -> None:
    """The gate that makes loose safety matching possible."""
    assert lexicon.has_imaging_context("j'ai du métal dans le bras") is True
    assert lexicon.has_imaging_context("je veux une IRM") is True
    assert lexicon.has_imaging_context("une brioche et un café") is False
