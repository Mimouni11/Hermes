"""Tests for the agent's voice.

Most of these need no voice model and no network: what can go wrong silently is
`prepare_text`, where a reference code read as a word or a time read as digits
makes the agent useless without anything raising. The synthesis tests skip
themselves when no Piper voice is installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hermes import tts
from hermes.config import get_settings


# --------------------------------------------------------------------------
# Numbers and times
# --------------------------------------------------------------------------

def test_numbers_are_spelled_in_french() -> None:
    """0 to 59, since that covers hours, minutes and days of the month."""
    assert tts.number_fr(0) == "zéro"
    assert tts.number_fr(7) == "sept"
    assert tts.number_fr(14) == "quatorze"
    assert tts.number_fr(20) == "vingt"
    assert tts.number_fr(21) == "vingt et un"
    assert tts.number_fr(30) == "trente"
    assert tts.number_fr(45) == "quarante-cinq"
    assert tts.number_fr(59) == "cinquante-neuf"


def test_out_of_range_numbers_are_left_as_digits() -> None:
    """Better a spoken digit than a wrong word."""
    assert tts.number_fr(60) == "60"
    assert tts.number_fr(-1) == "-1"


def test_one_oclock_is_singular() -> None:
    """"une heure", not "un heures" — the mistake a lookup table makes."""
    assert tts.say_time(1) == "une heure"
    assert tts.say_time(1, 30) == "une heure trente"


def test_times_are_spoken_not_spelled() -> None:
    """14h30 is read as a time, with no digits surviving."""
    assert tts.say_time(14, 30) == "quatorze heures trente"
    assert tts.say_time(9) == "neuf heures"
    assert tts.say_time(9, 0) == "neuf heures"


# --------------------------------------------------------------------------
# prepare_text
# --------------------------------------------------------------------------

def test_a_time_in_a_sentence_is_expanded() -> None:
    """Every written form of a time reaches the voice as words."""
    for written in ("14h30", "14:30", "14 h 30"):
        spoken = tts.prepare_text(f"Votre rendez-vous est à {written}.")
        assert "quatorze heures trente" in spoken
        assert "14" not in spoken


def test_a_bare_hour_is_expanded() -> None:
    """9h with no minutes is still a time."""
    assert "neuf heures" in tts.prepare_text("Nous ouvrons à 9h.")


def test_a_reference_code_is_read_letter_by_letter() -> None:
    """A code spoken as a word is useless to a caller writing it down."""
    spoken = tts.prepare_text("Votre code est K7M2PQ.")

    assert "K, 7, M, 2, P, Q" in spoken
    assert "K7M2PQ" not in spoken


def test_a_reference_code_survives_the_acronym_pass() -> None:
    """A code is expanded before anything else can rewrite its letters.

    TDM4KP is a code the agenda could really issue, and its first three
    characters are an acronym in ACRONYMS — so this pins the ordering.
    """
    spoken = tts.prepare_text("Code TDM4KP pour votre TDM.")

    assert "T, D, M, 4, K, P" in spoken
    assert "T.D.M." in spoken


def test_a_code_cannot_contain_the_ambiguous_letters() -> None:
    """I, O, 0 and 1 are not in the agenda's alphabet, so they are not codes."""
    assert "I, R, M" not in tts.prepare_text("IRMKJ9")


def test_an_ordinary_word_is_not_mistaken_for_a_code() -> None:
    """Six lower-case letters are a word, not a reference code."""
    assert tts.prepare_text("samedi") == "samedi"
    assert tts.prepare_text("Bonjour") == "Bonjour"


def test_medical_acronyms_are_spelled_out() -> None:
    """IRM is three letters, not a word."""
    assert "I.R.M." in tts.prepare_text("Vous avez une IRM.")
    assert "O.R.L." in tts.prepare_text("Service ORL")


def test_acronyms_inside_words_are_left_alone() -> None:
    """Whole words only, so a longer term is not cut apart."""
    assert "I.R.M." not in tts.prepare_text("IRMA est un prénom.")


def test_abbreviations_are_expanded() -> None:
    """rdv is written constantly and spoken never."""
    assert "rendez-vous" in tts.prepare_text("Votre rdv est confirmé.")
    assert "rdv" not in tts.prepare_text("Votre rdv est confirmé.").lower()
    assert "docteur" in tts.prepare_text("Dr Benali")


def test_iso_dates_become_spoken_dates() -> None:
    """2026-11-26 is unreadable aloud."""
    spoken = tts.prepare_text("Le 2026-11-26 à 14h30.")

    assert "vingt-six novembre" in spoken
    assert "2026-11-26" not in spoken
    # No doubled article: the sentence supplies "Le" already.
    assert "Le le" not in spoken


def test_an_impossible_time_is_left_alone() -> None:
    """25h99 is not a time; leave it rather than invent words for it."""
    assert "25:99" in tts.prepare_text("Code 25:99")


def test_empty_text_is_handled() -> None:
    """Nothing to say is not a crash."""
    assert tts.prepare_text("") == ""
    assert tts.prepare_text("   ") == ""


def test_whitespace_is_collapsed() -> None:
    """Expansion must not leave ragged spacing for the voice to pause on."""
    assert tts.prepare_text("Bonjour    madame") == "Bonjour madame"


# --------------------------------------------------------------------------
# The cache key
# --------------------------------------------------------------------------

def test_cache_key_is_stable() -> None:
    """The same line always lands on the same file."""
    assert tts.cache_key("bonjour", "piper", "v1") == tts.cache_key(
        "bonjour", "piper", "v1"
    )


def test_cache_key_changes_with_text_backend_and_voice() -> None:
    """Change any of the three and the line is synthesised again."""
    base = tts.cache_key("bonjour", "piper", "v1")

    assert tts.cache_key("bonsoir", "piper", "v1") != base
    assert tts.cache_key("bonjour", "gtts", "v1") != base
    assert tts.cache_key("bonjour", "piper", "v2") != base


def test_cache_key_names_its_backend() -> None:
    """A glance at the cache directory says which engine made each line."""
    assert tts.cache_key("bonjour", "piper", None).startswith("piper-")
    assert tts.cache_key("bonjour", "piper", None).endswith(".wav")


def test_clear_cache_counts_what_it_removes(tmp_path: Path) -> None:
    """Clearing reports how many lines it dropped."""
    cache = get_settings().paths.tts_cache
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / "piper-testmarker0000.wav"
    marker.write_bytes(b"")

    removed = tts.clear_cache()

    assert removed >= 1
    assert not marker.exists()


# --------------------------------------------------------------------------
# Synthesis, when a voice is installed
# --------------------------------------------------------------------------

def _voice_or_skip() -> None:
    if tts.find_voice_file() is None:
        pytest.skip("no Piper voice installed in models/piper")


def test_synthesise_writes_a_playable_wav(tmp_path: Path) -> None:
    """The agent's reply lands on disk at the configured rate."""
    _voice_or_skip()
    import soundfile as sf

    target = tmp_path / "reply.wav"
    speech = tts.synthesise("Bonjour, votre rendez-vous est confirmé.", target)

    assert speech.path == target
    assert target.exists()
    assert speech.duration_s > 0.3
    assert speech.from_cache is False
    assert sf.info(target).samplerate == get_settings().tts_sample_rate
    assert sf.info(target).channels == 1


def test_the_second_call_comes_from_the_cache() -> None:
    """The agent repeats itself constantly; it should synthesise once."""
    _voice_or_skip()
    line = "Pouvez-vous répéter, s'il vous plaît ?"
    tts.clear_cache()

    first = tts.synthesise(line)
    second = tts.synthesise(line)

    assert first.from_cache is False
    assert second.from_cache is True
    assert second.path == first.path
    assert second.duration_s == pytest.approx(first.duration_s, abs=0.01)


def test_samples_skip_the_disk() -> None:
    """The live path returns audio without writing anything."""
    _voice_or_skip()
    samples, rate = tts.synthesise_samples("Bonjour.")

    assert rate == get_settings().tts_sample_rate
    assert samples.dtype == np.float32
    assert samples.size > 0
    assert float(np.max(np.abs(samples))) <= 1.0


def test_an_unknown_backend_is_refused() -> None:
    """A typo in TTS_BACKEND fails loudly rather than silently falling back."""
    with pytest.raises(ValueError):
        tts.synthesise_samples("Bonjour.", backend="espeak")
