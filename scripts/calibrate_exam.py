"""Evaluate WP4 before LLM fallback. Run with uv run python scripts/calibrate_exam.py."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from build_exam_index import build, read_csv
from hermes import exam_bot
from hermes.config import get_settings, set_settings


@dataclass(frozen=True)
class Case:
    text: str
    expected: frozenset[str]


@dataclass(frozen=True)
class Observation:
    case: Case
    result: exam_bot.ExamMatch
    correct_rank: int | None


def hold_out(index: exam_bot.ExamIndex, text: str) -> exam_bot.ExamIndex:
    """Remove the query from every base, including duplicates across slot variants.

    A base left with no phrases is dropped, because a base with nothing to
    match on cannot be ranked. That makes the query unanswerable, which is why
    callers must filter such samples out with :func:`survives_holdout` rather
    than score them.
    """
    phrases = {exam_bot.match_phrase(text), exam_bot.prepare(text).query}
    bases = []
    for base in index.bases:
        retained = tuple(phrase for phrase in base.phrases if phrase not in phrases)
        if retained == base.phrases:
            bases.append(base)
        elif retained:
            aliases = frozenset().union(*(exam_bot.modalities_in(phrase) for phrase in retained))
            bases.append(replace(base, phrases=retained, modalities=aliases | {base.modality}))
    return replace(index, bases=tuple(bases))


def survives_holdout(
    index: exam_bot.ExamIndex, text: str, expected: frozenset[str],
) -> bool:
    """Whether the expected exam is still reachable once the query is removed.

    Without this the T1 sweep asks the matcher to find an exam it has just
    deleted, and scores the inevitable mismatch as a **wrong** resolution.

    The cases this excludes are not random. After side-stripping and filler
    removal, all ten French keywords of a lateral pair collapse to one phrase —
    every wording of `EX022`/`EX023` reduces to "irm epaule" — so holding that
    phrase out empties the base. The effect therefore falls hardest on exactly
    the bases the base-collapsing design works best on, and leaving it in
    biases the margin sweep towards punishing them.
    """
    held = {base.key for base in hold_out(index, text).bases}
    return any(
        (base := index.base_for(exam_id)) is not None and base.key in held
        for exam_id in expected
    )


def observe(case: Case, index: exam_bot.ExamIndex) -> Observation:
    # A zero floor and margin expose the top base's ungated slot/transfer decision.
    result = exam_bot.identify(case.text, index, threshold=0, margin=0, use_llm=False)
    correct_rank = None
    for rank, candidate in enumerate(result.candidates, start=1):
        base = index.base_for(candidate.exam_ids[0])
        assert base is not None
        exam_id, _ = exam_bot.resolve(base, result.laterality, result.contrast, index)
        if exam_id in case.expected:
            correct_rank = rank
            break
    return Observation(case, result, correct_rank)


def outcome(case: Case, exam_id: str | None) -> str:
    if exam_id is None:
        return "clarify" if not case.expected else "miss"
    return "right" if exam_id in case.expected else "wrong"


def measure(rows: list[Observation], floor: int, margin: float) -> Counter[str]:
    # Both gates withhold any exam ID. Resolution and transfer come from identify().
    return Counter(
        outcome(
            row.case,
            row.result.exam_id
            if row.result.score >= floor and row.result.margin >= margin
            else None,
        )
        for row in rows
    )


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor", type=int, default=settings.exam_fuzzy_threshold)
    parser.add_argument("--margin", type=float, default=settings.exam_margin)
    args = parser.parse_args()

    # Sweep the permitted shortlist sizes even when the runtime cap is lower.
    set_settings(replace(settings, exam_llm_candidates=5))
    raw = settings.paths.raw
    exams = read_csv(raw / "exams.csv")
    payload = build(exams, read_csv(raw / "confusables.csv"), ["fr"])
    with TemporaryDirectory(prefix="hermes-exam-") as directory:
        path = Path(directory) / "exam_index.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        index = exam_bot.load_index(path)

    t1_all = [
        Case(text.strip(), frozenset([row["id"]]))
        for row in exams
        for text in dict.fromkeys(row["keywords_fr"].split("|"))
        if text.strip()
    ]
    t1_cases = [case for case in t1_all if survives_holdout(index, case.text, case.expected)]
    excluded = len(t1_all) - len(t1_cases)
    print(
        f"T1: {len(t1_cases)} answerable of {len(t1_all)} "
        f"({excluded} excluded — holding the query out would delete its only base phrase)"
    )
    t2_cases = [
        Case(row["text"], frozenset([row["exam_id"]]))
        for row in read_csv(raw / "intents.csv")
        if row["intent"] == "new_appointment" and row["lang"] in {"fr", "mix"} and row["exam_id"]
    ]
    t3_cases = [
        Case(row["text"], frozenset(filter(None, [row["exam_id"], *row["alt_ids"].split("|")])))
        if row["expected_status"] == "matched"
        else Case(row["text"], frozenset())
        for row in read_csv(raw / "exam_test_t3.csv")
    ]
    t1 = [observe(case, hold_out(index, case.text)) for case in t1_cases]
    t2 = [observe(case, index) for case in t2_cases]
    t3 = [observe(case, index) for case in t3_cases]
    transfer_cases = [
        Case(f"Je voudrais {exam.name}", frozenset([exam.exam_id]))
        for exam in index.exams.values()
        if exam.escalate
    ]
    transfers = [observe(case, index) for case in transfer_cases]

    print("T1 exams covered:", len({next(iter(case.expected)) for case in t1_cases}))
    print("Counts as right/wrong/withheld (T3 includes 7 expected clarifications)")
    print("floor margin    T1             T2             T3          transfers")
    for floor in (75, 80, 82, 85, 90, 95, 100):
        for margin in (3, 5, 8, 10, 12, 14):
            columns = []
            for rows in (t1, t2, t3):
                counts = measure(rows, floor, margin)
                columns.append(
                    f"{counts['right']}/{counts['wrong']}/{counts['miss'] + counts['clarify']}"
                )
            print(
                f"{floor:>5} {margin:>6}    {columns[0]:<14} "
                f"{columns[1]:<14} {columns[2]:<11} {measure(transfers, floor, margin)['right']}"
            )

    print(f"Selected score floor {args.floor}, margin {args.margin:g}")
    sets = [("T1", t1), ("T2", t2), ("T3", t3)]
    for name, rows in sets:
        counts = measure(rows, args.floor, args.margin)
        print(f"{name} ({len(rows)}): {dict(counts)}")
    for name, rows in sets:
        eligible = [row for row in rows if row.result.score < args.floor and row.result.candidates]
        print(f"{name} below-floor candidates: calls and recoverable labels in top 1 / 3 / 5")
        for minimum in (0, 50, 55, 60, 65, 70, 75, 80):
            band = [row for row in eligible if row.result.score >= minimum]
            hits = [
                sum(row.correct_rank is not None and row.correct_rank <= k for row in band)
                for k in (1, 3, 5)
            ]
            print(f"  {minimum:>2}: {len(band):>3} calls  {hits}")

    for name, rows in [*sets, ("Transfers", transfers)]:
        actual = Counter()
        statuses = Counter()
        for row in rows:
            case = row.case
            catalogue = hold_out(index, case.text) if name == "T1" else index
            match = exam_bot.identify(
                case.text, catalogue, threshold=args.floor, margin=args.margin, use_llm=False,
            )
            actual[outcome(case, match.exam_id)] += 1
            statuses[match.status.value] += 1
        assert actual == measure(rows, args.floor, args.margin), actual
        print(f"{name} live decision statuses: {dict(statuses)}")


if __name__ == "__main__":
    main()
