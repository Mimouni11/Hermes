"""Tests for the speech-to-text wrapper that need no model and no network.

No test here loads Whisper. What is testable without the weights is the text
normaliser the WER is computed through, the confidence derived from the
segments' log-probabilities, and the initial prompt built from the catalogue —
and those are where the bugs that would silently skew an evaluation live.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes import stt
from hermes.config import get_settings
from hermes.stt import Segment, Transcript

pytestmark: list = []


# --------------------------------------------------------------------------
# The WER normaliser
# --------------------------------------------------------------------------

def test_normaliser_lowercases() -> None:
    """Case is not a recognition error."""
    assert stt.normalise_for_wer("IRM Cérébrale") == "irm cérébrale"


def test_normaliser_removes_punctuation() -> None:
    """The reference has question marks, Whisper is inconsistent about them."""
    assert stt.normalise_for_wer("Bonjour, c'est possible ?") == "bonjour c est possible"


def test_normaliser_keeps_accents() -> None:
    """Accents stay: échographie and echographie are different words.

    This is the assertion that matters most. If the normaliser stripped
    diacritics, every accent mistake in the transcripts would score as correct
    and the evaluation would be measuring nothing on exactly the words the
    initial prompt exists to fix.
    """
    normalised = stt.normalise_for_wer("Échographie abdominale à jeun")

    assert normalised == "échographie abdominale à jeun"
    assert "é" in normalised
    assert "echographie" not in normalised


def test_normaliser_splits_on_apostrophes() -> None:
    """Elision becomes a word break, so l'IRM and l IRM score alike."""
    assert stt.normalise_for_wer("l'IRM") == stt.normalise_for_wer("l IRM")
    assert stt.normalise_for_wer("l'IRM") == "l irm"


def test_normaliser_handles_the_typographic_apostrophe() -> None:
    """Whisper emits U+2019, the corpus uses U+0027; they must agree."""
    assert stt.normalise_for_wer("j’ai") == stt.normalise_for_wer("j'ai")


def test_normaliser_collapses_whitespace() -> None:
    """Ragged spacing is not a recognition error either."""
    assert stt.normalise_for_wer("  deux   mots \n ici ") == "deux mots ici"


def test_normaliser_keeps_digits() -> None:
    """Numbers survive: the corpus writes them as digits and so does Whisper."""
    assert stt.normalise_for_wer("le 12 mars à 14h30") == "le 12 mars à 14h30"


def test_normaliser_is_idempotent() -> None:
    """Normalising twice changes nothing the second time."""
    once = stt.normalise_for_wer("L'Échographie, c'est quand ?")

    assert stt.normalise_for_wer(once) == once


def test_normaliser_handles_empty_and_none() -> None:
    """An empty transcript normalises to an empty string, not a crash."""
    assert stt.normalise_for_wer("") == ""
    assert stt.normalise_for_wer("   ") == ""
    assert stt.normalise_for_wer(None) == ""  # type: ignore[arg-type]


def test_normaliser_makes_matching_transcripts_identical() -> None:
    """A correct transcript scores zero errors despite differing typography."""
    reference = "Je dois être à jeun pour l'échographie abdominale ?"
    hypothesis = "je dois être à jeun pour l’échographie abdominale"

    assert stt.normalise_for_wer(reference) == stt.normalise_for_wer(hypothesis)


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------

def _segment(start: float, end: float, logprob: float | None, no_speech: float = 0.0):
    return Segment(
        start=start, end=end, text="x", avg_logprob=logprob, no_speech_prob=no_speech
    )


def test_confidence_is_between_zero_and_one() -> None:
    """A log-probability becomes a probability-like number."""
    value = stt.confidence_from([_segment(0.0, 2.0, -0.2)])

    assert 0.0 < value <= 1.0


def test_higher_logprob_means_higher_confidence() -> None:
    """The ordering survives the transform."""
    better = stt.confidence_from([_segment(0.0, 2.0, -0.1)])
    worse = stt.confidence_from([_segment(0.0, 2.0, -1.5)])

    assert better > worse


def test_confidence_is_duration_weighted() -> None:
    """A long confident segment outweighs a short uncertain one."""
    weighted = stt.confidence_from(
        [_segment(0.0, 9.0, -0.1), _segment(9.0, 9.5, -3.0)]
    )
    unweighted_mean = stt.confidence_from(
        [_segment(0.0, 1.0, -0.1), _segment(1.0, 2.0, -3.0)]
    )

    assert weighted > unweighted_mean


def test_confidence_without_logprobs_is_zero() -> None:
    """A backend that reports no log-probabilities yields no confidence."""
    assert stt.confidence_from([_segment(0.0, 1.0, None)]) == 0.0
    assert stt.confidence_from([]) == 0.0


def test_no_speech_takes_the_maximum() -> None:
    """One confidently-silent segment is the signal, not the average."""
    assert stt.no_speech_from(
        [_segment(0.0, 1.0, -0.2, 0.01), _segment(1.0, 2.0, -0.2, 0.93)]
    ) == pytest.approx(0.93)


# --------------------------------------------------------------------------
# The initial prompt
# --------------------------------------------------------------------------

def test_initial_prompt_holds_the_modality_names() -> None:
    """The vocabulary the call line actually uses is in the prompt."""
    prompt = stt.build_initial_prompt()

    for term in ("IRM", "scanner", "échographie", "rachis lombaire"):
        assert term in prompt


def test_initial_prompt_mines_the_catalogue(tmp_path: Path) -> None:
    """Long technical names from the CSV reach the prompt."""
    csv_path = tmp_path / "exams.csv"
    csv_path.write_text(
        "id,modality,duration_min,name_fr,region_fr\n"
        "EX001,IRM,25,Hystérosalpingographie bilatérale,Pelvis\n"
        "EX002,scanner,10,Arthroscanner du poignet,Poignet\n",
        encoding="utf-8",
    )

    prompt = stt.build_initial_prompt(csv_path).lower()

    # Case-insensitive: a term already in CORE_VOCABULARY keeps the core
    # spelling rather than the catalogue's capitalisation.
    assert "hystérosalpingographie" in prompt
    assert "arthroscanner" in prompt


def test_initial_prompt_drops_short_and_common_words(tmp_path: Path) -> None:
    """Short words carry no acoustic difficulty and would crowd the prompt."""
    csv_path = tmp_path / "exams.csv"
    csv_path.write_text(
        "id,modality,duration_min,name_fr,region_fr\n"
        "EX001,IRM,25,Scanner du coude droit,Coude\n",
        encoding="utf-8",
    )

    vocabulary = stt.exam_vocabulary(csv_path)
    mined = [term for term in vocabulary if term not in stt.CORE_VOCABULARY]

    assert "du" not in mined
    assert "coude" not in mined
    assert "droit" not in mined


def test_initial_prompt_survives_a_missing_catalogue(tmp_path: Path) -> None:
    """No CSV is not a crash: the core vocabulary still goes in."""
    prompt = stt.build_initial_prompt(tmp_path / "absent.csv")

    assert "IRM" in prompt
    assert "échographie" in prompt


def test_vocabulary_has_no_duplicates(tmp_path: Path) -> None:
    """A term in both the core list and the catalogue appears once."""
    csv_path = tmp_path / "exams.csv"
    csv_path.write_text(
        "id,modality,duration_min,name_fr,region_fr\n"
        "EX001,IRM,25,Échographie abdominale,Abdomen\n",
        encoding="utf-8",
    )

    vocabulary = stt.exam_vocabulary(csv_path)
    folded = [stt._fold(term) for term in vocabulary]

    assert len(folded) == len(set(folded))


# --------------------------------------------------------------------------
# Transcript plumbing
# --------------------------------------------------------------------------

def test_transcript_as_dict_has_the_documented_keys() -> None:
    """transcribe() returns exactly the five fields the brief asked for."""
    transcript = Transcript(
        text="bonjour",
        language="fr",
        duration_s=1.5,
        segments=(),
        confidence=0.8,
        backend="faster-whisper",
        no_speech_prob=0.02,
        latency_s=0.4,
    )

    assert set(transcript.as_dict()) == {
        "text",
        "confidence",
        "no_speech_prob",
        "duration_s",
        "latency_s",
    }


def test_tidying_a_transcript_collapses_whitespace() -> None:
    """Whisper pads segment text with spaces; joining them must not double up."""
    assert stt.normalise_text("  bonjour   madame  ") == "bonjour madame"
    assert stt.normalise_text("bonjour , madame") == "bonjour, madame"


def test_default_device_is_cpu_int8() -> None:
    """The shipped default is CPU int8, read from config, not probed.

    Pinning it matters: probing would make the measured latency depend on
    whichever machine ran the evaluation.
    """
    assert stt.resolve_device() == ("cpu", "int8")


def test_an_explicit_device_wins_over_config() -> None:
    """A caller can override the configured device and quantisation."""
    assert stt.resolve_device("cuda", "float16") == ("cuda", "float16")


def test_auto_opts_in_to_probing() -> None:
    """STT_DEVICE=auto probes for CUDA and still returns a usable pair."""
    device, compute = stt.resolve_device("auto")

    assert device in ("cpu", "cuda")
    assert compute in ("int8", "int8_float16")


def test_the_model_size_default_is_small() -> None:
    """small, not base: it halves the WER and stays realtime on CPU."""
    assert get_settings().stt_model_size == "small"


def test_is_empty_detects_nothing_said() -> None:
    """A blank transcript is empty, a real one is not."""
    blank = Transcript("", "fr", 1.0, (), 0.0, "faster-whisper")
    spoken = Transcript("bonjour", "fr", 1.0, (), 0.9, "faster-whisper")

    assert stt.is_empty(blank) is True
    assert stt.is_empty(spoken) is False
