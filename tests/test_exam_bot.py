"""Offline regression tests for the WP4 gazetteer and decision layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes.understand import exam_bot
from scripts.build_exam_index import build, read_csv


def write_index(
    path: Path, rows: list[dict[str, str]], pairs: list[dict[str, str]],
) -> exam_bot.ExamIndex:
    path.write_text(json.dumps(build(rows, pairs, ['fr'])), encoding='utf-8')
    return exam_bot.load_index(path)


@pytest.fixture
def catalogue(tmp_path):
    entries = [
        ('EX034', 'IRM', 'droite', 'non', 'IRM du genou droit', 'IRM genou droit'),
        ('EX035', 'IRM', 'gauche', 'non', 'IRM du genou gauche', 'IRM genou gauche'),
        ('EX001', 'IRM', 'aucune', 'non', 'IRM cérébrale', 'IRM cerveau|IRM tête'),
        ('EX002', 'IRM', 'aucune', 'oui', 'IRM cérébrale avec injection', 'IRM cerveau avec injection'),
        ('EX040', 'scanner', 'aucune', 'non', 'Scanner cérébral', 'TDM cérébrale|scanner tête'),
        ('EX075', 'radiographie', 'aucune', 'non', 'Radiographie du thorax', 'radio des poumons'),
        ('EX080', 'radiographie', 'aucune', 'non', 'Radiographie du rachis cervical', 'radio cervicales'),
        ('EX081', 'radiographie', 'aucune', 'non', 'Radiographie du rachis cervical dynamique', 'radio cervicales dynamique'),
        ('EX121', 'échographie', 'aucune', 'non', 'Échographie abdominale', 'écho du ventre'),
        ('EX139', 'échographie', 'aucune', 'non', 'Échographie appendiculaire', 'écho appendice'),
    ]
    rows = [
        dict(id=exam_id, modality=modality, laterality=side, contrast=contrast,
             name_fr=name, keywords_fr=keywords, duration_min='20',
             escalate='1' if exam_id == 'EX139' else '0')
        for exam_id, modality, side, contrast, name, keywords in entries
    ]
    pairs = [{
        'exam_a': 'EX080', 'exam_b': 'EX081', 'type': 'variante',
        'label': 'Radiographie du rachis cervical / Radiographie du rachis cervical dynamique',
    }]
    return write_index(tmp_path / 'exam_index.json', rows, pairs)


def identify(text: str, index: exam_bot.ExamIndex) -> exam_bot.ExamMatch:
    return exam_bot.identify(text, index, threshold=82, margin=8, repair=False, use_llm=False)


# --------------------------------------------------------------------------
# Reading the slots out of what the caller said
# --------------------------------------------------------------------------

@pytest.mark.parametrize(('text', 'expected'), [
    ('IRM cérébrale avec injection', 'oui'),
    ('IRM cérébrale injectée', 'oui'),
    ('IRM cérébrale sans injection', 'non'),
    ('IRM du cerveau pas injectée', 'non'),
    ('une IRM du cerveau non injectée', 'non'),
    ("IRM du cerveau sans produit de contraste", 'non'),
    ("IRM du cerveau pas d'injection", 'non'),
    ('IRM cérébrale', None),
])
def test_negated_injection_is_not_read_as_injected(text: str, expected) -> None:
    """"pas injectée" is the opposite of "injectée", not a mention of it.

    The measured failure: "Une IRM du cerveau, pas injectée" resolved to
    `EX002 IRM cérébrale avec injection`. The caller got the opposite of what
    they asked for, with the injected exam's preparation attached — bilan
    rénal, metformin and allergy warnings. Only `sans` was handled.
    """
    assert exam_bot.contrast_in(text) == expected


@pytest.mark.parametrize(('text', 'expected'), [
    ("j'ai le droit de passer une IRM du genou", None),
    ("est-ce que j'ai le droit à une IRM du genou ?", None),
    ('les droits du patient', None),
    ('IRM du genou droit', 'droite'),
    ('IRM du genou côté droit', 'droite'),
    ("la radio de ma cheville, celle de droite", 'droite'),
    ('IRM du genou gauche', 'gauche'),
    ('radiographie des deux genoux', 'bilatéral'),
])
def test_droit_as_entitlement_is_not_a_side(text: str, expected) -> None:
    """"le droit" is a noun. "le genou droit" is a side.

    The measured failure: "J'ai le droit de passer une IRM du genou ?" matched
    `EX034 IRM du genou droit` — a silent booking of the right knee from a
    question about entitlement, with the word stripped from the query so the
    trace did not show where the side came from.

    Only the determiner form is excluded, which is why "celle de droite" — a
    real row of `exam_test_t3.csv` — still reads as a side.
    """
    assert exam_bot.laterality_in(text) == expected


def test_an_explicitly_empty_index_is_honoured() -> None:
    """An empty catalogue means nothing matches, not "use the real one".

    `ExamIndex.__len__` makes an empty index falsy, so `index or get_index()`
    silently substituted the process-wide catalogue and answered from it with
    five confident candidates. A harness asserting "nothing should match" was
    being told about the real 173 exams instead.
    """
    empty = exam_bot.ExamIndex(bases=(), exams={}, confusables=(), langs=('fr',))

    result = identify('IRM du genou droit', empty)

    assert result.status is exam_bot.Status.NOT_FOUND
    assert result.exam_id is None
    assert result.candidates == ()


@pytest.mark.parametrize(('text', 'exam_id'), [
    ('IRM du genou droit', 'EX034'),
    ('Radio des poumons', 'EX075'),
    ('Une IRM du genouu gauche', 'EX035'),
    ("S'il vous plaît, une ECHOGRAPHIE ABDOMINALE !", 'EX121'),
    ('TDM cérébrale', 'EX040'),
    ('Une écho du ventre', 'EX121'),
    ('سلام، je voudrais une IRM du genou droit، وقتاش تنجمو؟', 'EX034'),
    ('IRM cérébrale', 'EX001'),
    ('IRM cérébrale avec injection', 'EX002'),
])
def test_resolves_exam(catalogue, text, exam_id):
    result = identify(text, catalogue)
    assert (result.status, result.exam_id) == (exam_bot.Status.MATCHED, exam_id)


def test_exact_name_scores_full_confidence(catalogue):
    assert identify('IRM du genou droit', catalogue).score == 100


def test_unrelated_phrase_does_not_resolve(catalogue):
    result = identify('Une pizza margherita', catalogue)
    assert result.status in (exam_bot.Status.CLARIFY, exam_bot.Status.NOT_FOUND)
    assert result.exam_id is None
    assert result.question == exam_bot.OPEN_QUESTION


def test_missing_side_requires_clarification(catalogue):
    result = identify('IRM du genou', catalogue)
    assert result.status == exam_bot.Status.CLARIFY
    assert result.reason == exam_bot.Reason.MISSING_SIDE
    assert result.exam_id is None
    assert result.question == exam_bot.SIDE_QUESTION
    assert result.candidates[0].exam_ids == ('EX034', 'EX035')


def test_cross_base_tie_asks_about_both_modalities(catalogue):
    result = identify('IRM cérébrale ou scanner cérébral', catalogue)
    assert result.status == exam_bot.Status.CLARIFY
    assert result.reason == exam_bot.Reason.CLOSE_BASES
    assert result.exam_id is None
    assert 'IRM cérébrale' in result.question and 'Scanner cérébral' in result.question


def test_confusable_pair_uses_catalogue_labels(catalogue):
    result = identify(
        'Radiographie du rachis cervical ou radiographie du rachis cervical dynamique', catalogue,
    )
    assert result.status == exam_bot.Status.CLARIFY
    assert result.reason == exam_bot.Reason.CONFUSABLE_PAIR
    assert result.exam_id is None
    assert result.question == (
        'Vous souhaitez Radiographie du rachis cervical '
        'ou Radiographie du rachis cervical dynamique ?'
    )


def test_flagged_exam_transfers_after_resolution(catalogue):
    result = identify('Échographie appendiculaire', catalogue)
    assert (result.status, result.exam_id) == (exam_bot.Status.TRANSFER, 'EX139')
    assert result.reason == exam_bot.Reason.ESCALATE
    assert result.needs_human


def test_gazetteer_covers_every_project_exam(tmp_path):
    raw = Path(__file__).resolve().parents[1] / 'data' / 'raw'
    if not (raw / 'exams.csv').is_file() or not (raw / 'confusables.csv').is_file():
        pytest.skip('local WP4 catalogue CSVs unavailable')
    rows = read_csv(raw / 'exams.csv')
    index = write_index(tmp_path / 'exam_index.json', rows, read_csv(raw / 'confusables.csv'))
    expected = {row['id'] for row in rows}
    members = [exam_id for base in index.bases for exam_id in base.exam_ids]
    assert len(members) == len(expected) == 173
    assert set(members) == expected == set(index.exams)
    assert len(index.bases) == 132
    assert all(base.phrases for base in index.bases)
    escalations = {exam.exam_id for exam in index.exams.values() if exam.escalate}
    assert escalations == {'EX020', 'EX042', 'EX052', 'EX057', 'EX139'}
    for exam_id in escalations:
        result = identify(index.exams[exam_id].name, index)
        assert (result.status, result.exam_id) == (exam_bot.Status.TRANSFER, exam_id)
