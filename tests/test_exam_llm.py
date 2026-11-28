"""Verify WP4 fallback decisions against the real LLM wrapper with a fake backend."""

import json
from dataclasses import replace

import pytest

from hermes import config, exam_bot, llm


@pytest.fixture
def fallback_index(monkeypatch, tmp_path):
    settings = config.get_settings()
    isolated = replace(
        settings,
        paths=replace(settings.paths, processed=tmp_path),
        exam_fuzzy_threshold=82,
        exam_margin=8,
        exam_llm_threshold=65,
        exam_llm_candidates=20,
        use_local_llm=False,
    )
    monkeypatch.setattr(config, "_SETTINGS", isolated)
    exams = {
        f"EX{i:03}": exam_bot.Exam(
            f"EX{i:03}", f"IRM exemple {i}", "IRM", "aucune", "non", "", 20,
            i == 2, "validation médicale" if i == 2 else "",
        )
        for i in range(1, 7)
    }
    bases = tuple(
        exam_bot.Base(
            exam_id, "IRM", exam.name, exam.name, (exam.name,), (exam_id,),
            frozenset({"IRM"}), ("aucune",), ("non",),
        )
        for exam_id, exam in exams.items()
    )
    return exam_bot.ExamIndex(bases, exams)


def rank_at(monkeypatch, score):
    def rank(state, index, limit):
        return [
            exam_bot.Scored(base, score - i, base.display)
            for i, base in enumerate(index.bases[:limit])
        ]

    monkeypatch.setattr(exam_bot, "rank", rank)


def completion(text):
    return llm.Completion(text, "test", "gemini", None, None, 0.01, False)


def test_selection_cache_and_transfer(monkeypatch, fallback_index):
    rank_at(monkeypatch, 75)
    requests = []

    def complete(prompt, system, temperature, json_mode, timeout):
        requests.append(json.loads(prompt))
        assert json_mode
        # The exam path has its own budget: its fallback is a question, not a
        # publishable answer, so it is allowed to wait longer than the FAQ.
        assert timeout == config.get_settings().llm_timeout_exam_s
        return completion('{"exam_id":"EX002","reason":"ok"}')

    monkeypatch.setattr(llm, "complete_gemini", complete)
    first = exam_bot.identify("Une IRM, merci", fallback_index, repair=False)
    repeat = exam_bot.identify("  UNE IRM,  merci ! ", fallback_index, repair=False)
    assert (first.status, first.exam_id, first.used_llm) == (
        exam_bot.Status.TRANSFER, "EX002", True,
    )
    assert repeat.exam_id == first.exam_id
    assert "llm:cache" in repeat.trace
    assert len(requests) == 1
    assert [item["id"] for item in requests[0]["candidats"]] == [
        "EX001", "EX002", "EX003", "EX004", "EX005",
    ]

    changed = replace(fallback_index, bases=fallback_index.bases[1:])
    exam_bot.identify("Une IRM, merci", changed, repair=False)
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("answer", "trace"),
    [
        ('{"exam_id":"EX003","reason":"ok"}', "llm:selected:EX003"),
        ('{"exam_id":"EX006","reason":"outside shortlist"}', "llm:rejected_id"),
        ('{"exam_id":null,"reason":"ambiguous"}', "llm:abstained"),
        ('{"exam_id":"EX003"}', "llm:invalid_reply"),
        ("invalid JSON", "llm:invalid_json"),
        (None, "llm:timeout"),
    ],
)
def test_fallback_answers(monkeypatch, fallback_index, answer, trace):
    rank_at(monkeypatch, 65)

    def complete(*args):
        if answer is None:
            raise llm.LLMTimeout("test timeout")
        return completion(answer)

    monkeypatch.setattr(llm, "complete_gemini", complete)
    result = exam_bot.identify("Une IRM", fallback_index, repair=False)
    assert result.used_llm and trace in result.trace
    if trace == "llm:selected:EX003":
        assert result.status is exam_bot.Status.MATCHED
        assert result.exam_id == "EX003"
    else:
        assert result.status is exam_bot.Status.CLARIFY
        assert result.exam_id is None
        assert result.question == exam_bot.OPEN_QUESTION


@pytest.mark.parametrize(("score", "enabled"), [(64, True), (82, True), (75, False)])
def test_no_llm_outside_band_or_when_disabled(monkeypatch, fallback_index, score, enabled):
    rank_at(monkeypatch, score)

    def unexpected(*args, **kwargs):
        pytest.fail("no LLM call expected")

    monkeypatch.setattr(llm, "complete_or", unexpected)
    result = exam_bot.identify("Une IRM", fallback_index, repair=False, use_llm=enabled)
    assert result.status is exam_bot.Status.CLARIFY
    assert not result.used_llm


@pytest.mark.parametrize("guard", ["confusable", "unresolved_top"])
def test_ambiguity_guards(monkeypatch, fallback_index, guard):
    rank_at(monkeypatch, 75)

    def unexpected(*args, **kwargs):
        pytest.fail("no LLM call expected")

    monkeypatch.setattr(llm, "complete_or", unexpected)
    if guard == "confusable":
        index = replace(
            fallback_index,
            confusables=({"a": "EX001", "b": "EX002", "label": "A / B"},),
        )
    else:
        first, *rest = fallback_index.bases
        exams = {
            **fallback_index.exams,
            "EX001": replace(fallback_index.exams["EX001"], laterality="droite"),
            "EX007": replace(fallback_index.exams["EX001"], exam_id="EX007", laterality="gauche"),
        }
        first = replace(
            first, exam_ids=("EX001", "EX007"), laterality_options=("droite", "gauche"),
        )
        index = replace(fallback_index, bases=(first, *rest), exams=exams)
    result = exam_bot.identify("Une IRM", index, repair=False)
    assert result.status is exam_bot.Status.CLARIFY
    assert not result.used_llm and f"llm:skipped_{guard}" in result.trace
