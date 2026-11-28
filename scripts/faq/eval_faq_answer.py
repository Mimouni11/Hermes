"""Compare two ways of answering an FAQ question: verbatim + τ, or the LLM.

The question this answers is whether the LLM earns its place, and the honest
measure is not phrasing — it is **wrong answers delivered to a caller**.

The case that motivates it: "Est-ce que je peux venir accompagné pour le
scanner ?" retrieves `jeun` at 0.840, while genuine correct retrievals sit
around 0.6. No threshold refuses the first without also refusing most of the
second. A similarity score says *how close* a passage is, never *whether it
answers the question* — so the only thing that can catch it is something that
reads both.

Two policies over the same retrieval:

- **verbatim** — if the best hit clears τ_faq, speak its stored answer.
- **llm** — hand the top-3 passages to the model, which rephrases, fixes the
  polarity, or refuses when none of them fit.

Reported for each: wrong answers, correct answers, refusals (wanted and
unwanted), latency, and the timeout rate. Writes
`reports/faq_answer_eval.csv`.

    python scripts/faq/eval_faq_answer.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hermes.understand import faq_rag  # noqa: E402
from hermes.infra import llm  # noqa: E402
from hermes.config import get_settings  # noqa: E402

RESULT_COLUMNS = [
    "id", "question", "expected", "retrieved", "retrieval_score",
    "policy", "source", "seconds", "outcome", "answer",
]

#: How the model says it cannot answer. Matched loosely because the wording is
#: the model's own; the prompt only tells it to say so.
REFUSAL = re.compile(
    r"n'?ai pas|ne dispose pas|pas cette information|pas d'?information|"
    r"je ne (?:sais|peux) pas",
    re.IGNORECASE,
)


@dataclass
class Row:
    """One question under one policy."""

    id: str
    question: str
    expected: str
    retrieved: str
    retrieval_score: float
    policy: str
    source: str
    seconds: float
    outcome: str
    answer: str


def attribute(answer: str, entries: dict[str, str]) -> str:
    """Which FAQ entry the delivered answer actually came from.

    Judged on the text handed to the caller, not on which entry ranked first.
    The two differ, and the difference is the point: "Je peux me garer près du
    centre ?" retrieves `duree` first, and the model answered from `parking`
    because `parking` was in the top-3 passages it was given. Scoring by the
    retrieved id would have logged that recovery as a failure.
    """
    from rapidfuzz import fuzz

    if not answer.strip():
        return ""
    best, score = "", 0.0
    for faq_id, stored in entries.items():
        ratio = fuzz.partial_ratio(answer.lower(), stored.lower()) / 100.0
        if ratio > score:
            best, score = faq_id, ratio
    return best if score >= 0.6 else ""


def classify(
    expected: str,
    delivered: str,
    answered: bool,
    retrieved: str = "",
) -> str:
    """What the caller actually got.

    `expected` is empty for the deliberately unanswerable questions, where
    refusing is the correct outcome rather than a failure.

    Attribution is by the delivered text, which catches the model answering
    from a passage that was not top-1. But text matching fails in the other
    direction when the model rewrites heavily — the polarity fix turned "Non,
    le centre est fermé le dimanche. Le samedi…" into "Le centre est ouvert le
    samedi… mais fermé le dimanche", which no longer resembles its source.
    So a correct top-1 counts as correct even when attribution cannot see it.
    """
    wanted = bool(expected)
    if not answered:
        return "refused_right" if not wanted else "refused_wrong"
    if not wanted:
        return "WRONG_unanswerable"
    if delivered == expected or retrieved == expected:
        return "correct"
    return "WRONG_entry"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the test set and output paths."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test", type=Path,
                        default=settings.paths.raw / "faq_test.csv")
    parser.add_argument("--out", type=Path,
                        default=settings.paths.reports / "faq_answer_eval.csv")
    parser.add_argument("--lang", default="fr")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run both policies over the test set and report the difference."""
    args = parse_args(argv)
    settings = get_settings()

    with open(args.test, encoding="utf-8-sig", newline="") as handle:
        tests = [
            r for r in csv.DictReader(handle)
            if r.get("lang") == args.lang and (r.get("text") or "").strip()
        ]

    answerable = sum(1 for t in tests if (t.get("faq_id") or "").strip())
    print(f"{len(tests)} questions: {answerable} answerable, "
          f"{len(tests) - answerable} unanswerable by design")
    print(f"tau_faq = {settings.faq_accept_threshold}, "
          f"LLM budget = {settings.llm_timeout_faq_s}s\n")

    # The stored answers, so a delivered reply can be traced to its source.
    entries = {
        row["faq_id"]: (row.get("answer_fr") or "")
        for row in faq_rag.load_faq()
    }

    llm.clear_cache()
    llm.reset_usage()
    rows: list[Row] = []

    for test in tests:
        question = test["text"].strip()
        expected = (test.get("faq_id") or "").strip()
        hits = faq_rag.search(question, k=settings.faq_top_k)
        top = hits[0] if hits else None
        retrieved = top.faq_id if top else ""
        score = top.score if top else 0.0

        # --- policy 1: speak the stored answer when it clears the threshold
        started = time.perf_counter()
        best = faq_rag.best_match(question)
        elapsed = time.perf_counter() - started
        rows.append(Row(
            id=test["id"], question=question, expected=expected,
            retrieved=retrieved, retrieval_score=score,
            policy="verbatim", source="stored" if best else "below_tau",
            seconds=elapsed,
            outcome=classify(expected, best.faq_id if best else "",
                             best is not None, retrieved),
            answer=(best.entry.answer if best else ""),
        ))

        # --- policy 2: let the model read the passages and decide
        started = time.perf_counter()
        text, source = faq_rag.spoken_answer(question)
        elapsed = time.perf_counter() - started
        answered = text is not None and not REFUSAL.search(text or "")
        rows.append(Row(
            id=test["id"], question=question, expected=expected,
            retrieved=retrieved, retrieval_score=score,
            policy="llm", source=source, seconds=elapsed,
            outcome=classify(
                expected, attribute(text or "", entries), answered, retrieved
            ),
            answer=text or "",
        ))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id": row.id, "question": row.question, "expected": row.expected,
                "retrieved": row.retrieved,
                "retrieval_score": round(row.retrieval_score, 4),
                "policy": row.policy, "source": row.source,
                "seconds": round(row.seconds, 3), "outcome": row.outcome,
                "answer": row.answer,
            })

    outcomes = ["correct", "WRONG_entry", "WRONG_unanswerable",
                "refused_right", "refused_wrong"]
    header = f"{'outcome':<22}{'verbatim + tau':>16}{'LLM check':>12}"
    print(header)
    print("-" * len(header))
    for outcome in outcomes:
        verbatim = sum(1 for r in rows if r.policy == "verbatim" and r.outcome == outcome)
        with_llm = sum(1 for r in rows if r.policy == "llm" and r.outcome == outcome)
        print(f"{outcome:<22}{verbatim:>16}{with_llm:>12}")
    print("-" * len(header))

    def wrong(policy: str) -> int:
        return sum(
            1 for r in rows
            if r.policy == policy and r.outcome.startswith("WRONG")
        )

    print(f"{'WRONG ANSWERS':<22}{wrong('verbatim'):>16}{wrong('llm'):>12}")

    for policy in ("verbatim", "llm"):
        times = [r.seconds for r in rows if r.policy == policy]
        print(f"\n{policy}: mean {sum(times) / len(times):.2f}s, "
              f"max {max(times):.2f}s")
    print(f"\n{llm.USAGE.summary()}")

    print("\n=== the wrong answers the LLM caught ===")
    by_id: dict[str, dict[str, Row]] = {}
    for row in rows:
        by_id.setdefault(row.id, {})[row.policy] = row
    for test_id, pair in by_id.items():
        before, after = pair.get("verbatim"), pair.get("llm")
        if not before or not after:
            continue
        if before.outcome.startswith("WRONG") and not after.outcome.startswith("WRONG"):
            print(f"\n  {before.question}")
            print(f"    retrieved {before.retrieved} @ {before.retrieval_score:.3f}"
                  f" (wanted {before.expected or 'nothing'})")
            print(f"    verbatim would say: {before.answer[:76]}")
            print(f"    LLM said          : {after.answer[:76]}")

    regressions = [
        (p["verbatim"], p["llm"]) for p in by_id.values()
        if "verbatim" in p and "llm" in p
        and not p["verbatim"].outcome.startswith("WRONG")
        and p["llm"].outcome.startswith("WRONG")
    ]
    if regressions:
        print("\n=== wrong answers the LLM introduced ===")
        for before, after in regressions:
            print(f"  {before.question}\n    {after.answer[:76]}")

    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
