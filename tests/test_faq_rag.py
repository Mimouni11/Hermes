"""Tests for FAQ retrieval.

The retrieval itself needs Qdrant and an indexed FAQ, so those tests skip when
either is missing. What is always tested is the behaviour that does not depend
on a server: reading the source, rendering the grounding block, and the rule
that matters most — returning nothing rather than guessing.

Accuracy is not asserted here. 75% top-1 is a measurement of one index built
from one corpus with one model, and pinning it into the suite would turn a
corpus edit into a test failure. It belongs in docs/WP3.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.understand import faq_rag
from hermes.config import get_settings
from hermes.understand.faq_rag import FaqEntry, Retrieved


def _needs_index() -> None:
    """Skip unless Qdrant is up and the FAQ collection is populated."""
    from hermes.infra import vector_store

    settings = get_settings()
    if not vector_store.ping():
        pytest.skip("Qdrant is not running: docker compose up -d")
    if vector_store.count(settings.faq_collection) == 0:
        pytest.skip("FAQ collection is empty: scripts/build_faq_index.py")


def _entry(faq_id: str, score: float) -> Retrieved:
    return Retrieved(
        entry=FaqEntry(
            faq_id=faq_id,
            question=f"question {faq_id}",
            answer=f"réponse {faq_id}",
        ),
        score=score,
    )


# --------------------------------------------------------------------------
# The source
# --------------------------------------------------------------------------

def test_the_faq_loads_with_its_languages() -> None:
    """Every entry carries a French question and a French answer."""
    rows = faq_rag.load_faq()

    assert rows
    for row in rows:
        assert row["faq_id"]
        assert row["question_fr"].strip()
        assert row["answer_fr"].strip()


def test_every_entry_has_paraphrases() -> None:
    """The variants column is what lifts a caller's wording onto an entry."""
    rows = faq_rag.load_faq()

    for row in rows:
        variants = [v for v in row.get("question_fr_variants", "").split("|") if v.strip()]
        assert len(variants) >= 2, f"{row['faq_id']} has {len(variants)} variants"


# --------------------------------------------------------------------------
# The grounding block
# --------------------------------------------------------------------------

def test_context_carries_questions_answers_and_scores() -> None:
    """The model needs to see which passage is the strong one."""
    block = faq_rag.format_context([_entry("horaires", 0.81), _entry("parking", 0.64)])

    assert "horaires" in block
    assert "réponse parking" in block
    assert "0.81" in block
    assert block.index("horaires") < block.index("parking")


def test_an_empty_context_says_so() -> None:
    """Silence would let the model fill the gap from its own knowledge."""
    block = faq_rag.format_context([])

    assert block
    assert "aucune" in block.lower()


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def test_a_question_retrieves_its_entry() -> None:
    """The ordinary path."""
    _needs_index()
    hits = faq_rag.search("Vous êtes ouverts jusqu'à quelle heure ?")

    assert hits
    assert hits[0].score > 0.0
    assert hits[0].entry.answer


def test_results_are_one_per_entry() -> None:
    """Five phrasings of one entry are one answer, not five.

    Without deduplication the top 3 could be three paraphrases of the same
    FAQ row, which looks like confidence and is actually one match.
    """
    _needs_index()
    hits = faq_rag.search("Quels sont vos horaires ?", k=3)

    ids = [h.faq_id for h in hits]
    assert len(ids) == len(set(ids))


def test_results_come_back_sorted() -> None:
    """Best first, because the caller only hears the first one."""
    _needs_index()
    hits = faq_rag.search("Je peux payer comment ?", k=3)

    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_an_empty_question_retrieves_nothing() -> None:
    """No embedding call, no result, no crash."""
    assert faq_rag.search("") == []
    assert faq_rag.search("   ") == []


def test_best_match_refuses_a_weak_hit() -> None:
    """Returning None is a real answer: it means hand the call over."""
    _needs_index()
    strict = faq_rag.best_match("Vous êtes ouverts le samedi ?", threshold=0.999)

    assert strict is None


def test_best_match_accepts_a_strong_hit() -> None:
    """With the floor on the ground, a reasonable question gets an entry."""
    _needs_index()
    loose = faq_rag.best_match("Vous êtes ouverts le samedi ?", threshold=0.0)

    assert loose is not None
    assert loose.entry.answer


def test_an_unanswerable_question_is_refused() -> None:
    """The "don't know" case, at the configured threshold.

    Refusing is scored as correct for these: the plan asks for a "don't know"
    rate on unanswerable questions, and the measured threshold refuses 100%
    of them.
    """
    _needs_index()
    assert faq_rag.is_answerable("Vous avez un compte Instagram ?") is False
    assert faq_rag.answer("Vous faites des réductions pour les étudiants ?") is None


def test_the_threshold_comes_from_config() -> None:
    """Changing the configured floor changes what is answerable."""
    from hermes import config

    _needs_index()
    question = "Vous êtes ouverts le samedi ?"

    config.set_settings(config.with_overrides(faq_accept_threshold=0.0))
    try:
        assert faq_rag.is_answerable(question) is True
    finally:
        config.reset_settings()

    config.set_settings(config.with_overrides(faq_accept_threshold=0.999))
    try:
        assert faq_rag.is_answerable(question) is False
    finally:
        config.reset_settings()


def test_the_answer_is_the_centres_own_text() -> None:
    """Retrieval returns the FAQ's wording, not something generated."""
    _needs_index()
    hit = faq_rag.best_match("Vous ouvrez à quelle heure le matin ?", threshold=0.0)

    assert hit is not None
    answers = {row["answer_fr"] for row in faq_rag.load_faq()}
    assert hit.entry.answer in answers
