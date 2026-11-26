"""Measure the exam LLM fallback twice, because there are two questions.

The previous evaluation asked both at once and answered neither. It ran 292
model calls in a burst, 254 came back `error:LLMError` once the free-tier
quota was gone, and the resulting "LLM enabled" column went into the accuracy
table as though it were a property of the model. It is a property of the
quota.

**Quality run** — *can the model pick the right exam when it answers?*
Throttled, a generous timeout, retried with backoff, and cached. Only the
cases the fallback can actually act on: the T3 and T4 rows where the gazetteer
declines to resolve, plus a seeded T1 sample. This is the honest
gazetteer-versus-gazetteer+LLM comparison, measured on the population where
the two can differ at all.

**Availability run** — *does it answer inside the production budget?*
A small sample at `LLM_TIMEOUT_EXAM_S`, no retries, no cache, reporting p50
and p90 latency and the timeout rate. This is what a caller would experience.

Keeping them apart matters because the two pull in opposite directions: the
quality run must remove every obstacle to getting an answer, and the
availability run must preserve them.

    uv run python scripts/eval_exam_llm.py
    uv run python scripts/eval_exam_llm.py --t1-sample 40 --availability 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from build_exam_index import build, read_csv
from calibrate_exam import hold_out
from eval_exam import Sample, replay_rules, samples
from hermes import exam_bot, llm
from hermes.config import get_settings, set_settings

#: Seconds between calls in the quality run. The burst that produced 254
#: errors had no gap at all; the free tier recovers if asked politely.
THROTTLE_S = 2.0

#: Attempts per case in the quality run, and the backoff between them. Only
#: here — production gets one call per turn and no retry, which is the point
#: of the budget in the first place.
ATTEMPTS = 3
BACKOFF_S = (0.0, 4.0, 12.0)

#: A generous budget for the quality run: what the model can do, not what it
#: can do in 3.5 seconds.
QUALITY_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class Call:
    """One fallback case and what came of it."""

    tier: str
    sample_id: str
    text: str
    expected: frozenset[str]
    gazetteer_id: str | None
    gazetteer_status: str
    llm_id: str | None
    llm_status: str
    source: str
    attempts: int
    latency_s: float


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Small samples make interpolation pretend."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


def source_of(match: exam_bot.ExamMatch) -> str:
    """The `llm:…` label `identify()` recorded, or "not_invoked"."""
    for entry in match.trace:
        if entry.startswith("llm:") and not entry.startswith("llm:selected"):
            return entry[4:]
    return "not_invoked"


def fallback_cases(
    cases: list[Sample],
    index: exam_bot.ExamIndex,
    t1_sample: int,
    seed: int,
) -> list[tuple[Sample, exam_bot.ExamMatch]]:
    """The rows where the fallback can change the answer, and nothing else.

    A row qualifies when the gazetteer resolves no exam *and* its top score
    falls inside the band `[exam_llm_threshold, exam_fuzzy_threshold)`.

    Both bounds matter, and the upper one is easy to forget. A first version
    required only that no exam resolved, which pulled in every missing-side
    and close-pair clarification — those score 100 and are answered with a
    question **by design**, never by asking the model. The result was that 31
    of 67 "fallback cases" were never invoked, and the quality table divided
    its successes by a denominator containing rows production never sends.
    """
    settings = get_settings()
    chosen: list[tuple[Sample, exam_bot.ExamMatch]] = []
    t1_pool: list[tuple[Sample, exam_bot.ExamMatch]] = []

    for case in cases:
        catalogue = hold_out(index, case.text) if case.tier == "T1" else index
        plain = exam_bot.identify(case.text, catalogue, use_llm=False)
        if plain.exam_id is not None or not plain.candidates:
            continue
        if not (
            settings.exam_llm_threshold
            <= plain.score
            < settings.exam_fuzzy_threshold
        ):
            continue
        if case.tier == "T1":
            t1_pool.append((case, plain))
        else:
            chosen.append((case, plain))

    random.Random(seed).shuffle(t1_pool)
    return chosen + t1_pool[:t1_sample]


def ask(case: Sample, index: exam_bot.ExamIndex) -> tuple[exam_bot.ExamMatch, str, int]:
    """Run the fallback with retries, until the model actually replies.

    Retrying `identify()` rather than the call underneath it is deliberate:
    ranking is microseconds, failures are not cached, and a success is — so a
    second attempt re-issues exactly one model request and the third never
    happens once one lands.
    """
    catalogue = hold_out(index, case.text) if case.tier == "T1" else index
    match = exam_bot.identify(case.text, catalogue, use_llm=True)
    source = source_of(match)
    for attempt in range(2, ATTEMPTS + 1):
        if source in ("llm", "cache"):
            return match, source, attempt - 1
        time.sleep(BACKOFF_S[min(attempt - 1, len(BACKOFF_S) - 1)])
        match = exam_bot.identify(case.text, catalogue, use_llm=True)
        source = source_of(match)
    return match, source, ATTEMPTS


def quality_run(
    pairs: list[tuple[Sample, exam_bot.ExamMatch]],
    index: exam_bot.ExamIndex,
) -> list[Call]:
    """Ask the model about every fallback case, with every obstacle removed."""
    settings = get_settings()
    set_settings(replace(settings, llm_timeout_exam_s=QUALITY_TIMEOUT_S))
    calls: list[Call] = []
    try:
        for position, (case, plain) in enumerate(pairs, start=1):
            started = time.perf_counter()
            match, source, attempts = ask(case, index)
            calls.append(Call(
                tier=case.tier, sample_id=case.id, text=case.text,
                expected=case.expected,
                gazetteer_id=plain.exam_id, gazetteer_status=plain.status.value,
                llm_id=match.exam_id, llm_status=match.status.value,
                source=source, attempts=attempts,
                latency_s=round(time.perf_counter() - started, 3),
            ))
            print(f"  {position}/{len(pairs)} {case.tier} {source}", flush=True)
            time.sleep(THROTTLE_S)
    finally:
        set_settings(settings)
    return calls


def availability_run(
    pairs: list[tuple[Sample, exam_bot.ExamMatch]],
    index: exam_bot.ExamIndex,
    size: int,
    seed: int,
) -> list[Call]:
    """Ask under production settings: one attempt, real budget, no cache."""
    settings = get_settings()
    sample = list(pairs)
    random.Random(seed).shuffle(sample)
    sample = sample[:size]

    calls: list[Call] = []
    for position, (case, plain) in enumerate(sample, start=1):
        catalogue = hold_out(index, case.text) if case.tier == "T1" else index
        llm.clear_cache()
        started = time.perf_counter()
        match = exam_bot.identify(case.text, catalogue, use_llm=True)
        elapsed = time.perf_counter() - started
        calls.append(Call(
            tier=case.tier, sample_id=case.id, text=case.text,
            expected=case.expected,
            gazetteer_id=plain.exam_id, gazetteer_status=plain.status.value,
            llm_id=match.exam_id, llm_status=match.status.value,
            source=source_of(match), attempts=1, latency_s=round(elapsed, 3),
        ))
        print(f"  {position}/{len(sample)} {source_of(match)} {elapsed:.2f}s", flush=True)
        time.sleep(THROTTLE_S)
    print(f"  budget was {settings.llm_timeout_exam_s:g}s", flush=True)
    return calls


def report(quality: list[Call], availability: list[Call], settings) -> str:
    recovered = [c for c in quality if c.llm_id is not None and c.llm_id in c.expected]
    wrong = [c for c in quality if c.llm_id is not None and c.expected and c.llm_id not in c.expected]
    premature = [c for c in quality if c.llm_id is not None and not c.expected]
    answered = [c for c in quality if c.source in ("llm", "cache")]
    labelled = [c for c in quality if c.expected]

    lines = [
        "# WP4 exam LLM fallback", "",
        f"Model `{settings.gemini_model}`. Shortlist capped at "
        f"{settings.exam_llm_candidates}, fallback band "
        f"[{settings.exam_llm_threshold}, {settings.exam_fuzzy_threshold}).",
        "",
        "Two runs, because there are two questions and one run cannot answer "
        "both: the quality run removes every obstacle to getting an answer, "
        "and the availability run keeps them.",
        "",
        "## Quality — can it pick the right exam when it answers?",
        "",
        f"Throttled {THROTTLE_S:g} s between calls, up to {ATTEMPTS} attempts "
        f"with {'/'.join(f'{b:g}' for b in BACKOFF_S[1:])} s backoff, "
        f"{QUALITY_TIMEOUT_S:g} s budget, cached.",
        "",
        f"Only the {len(quality)} cases the fallback can act on: the gazetteer "
        "resolved no exam **and** the top score sits inside the band. Rows the "
        "gazetteer already answers cannot be improved by asking, and a "
        "missing-side or close-pair clarification scores above the band and is "
        "answered with a question by design, never by asking the model.",
        "",
        "| | count |",
        "|---|---:|",
        f"| fallback cases asked about | {len(quality)} |",
        f"| of which carry a labelled exam | {len(labelled)} |",
        f"| model replied | {len(answered)} |",
        f"| recovery rate among replies | "
        + (f"{len(recovered) / len(answered):.0%} |" if answered else "— |"),
        f"| **recovered the right exam** | **{len(recovered)}** |",
        f"| **gave a wrong exam** | **{len(wrong)}** |",
        f"| named an exam where a question was expected | {len(premature)} |",
        "",
    ]

    by_tier = Counter(c.tier for c in quality)
    rec_tier = Counter(c.tier for c in recovered)
    wrong_tier = Counter(c.tier for c in wrong)
    lines.extend([
        "| Set | Asked | Recovered | Wrong |", "|---|---:|---:|---:|",
        *(
            f"| {tier} | {by_tier[tier]} | {rec_tier[tier]} | {wrong_tier[tier]} |"
            for tier in sorted(by_tier)
        ),
        "",
        "### Why the model declined",
        "",
        "| Outcome | count |", "|---|---:|",
        *(
            f"| `{source}` | {count} |"
            for source, count in Counter(c.source for c in quality).most_common()
        ),
        "",
    ])

    attempted = [c for c in availability if c.source in ("llm", "timeout")]
    live = [c.latency_s for c in attempted]
    replied = [c.latency_s for c in attempted if c.source == "llm"]
    timeouts = sum(c.source == "timeout" for c in attempted)
    lines.extend([
        "## Availability — does it answer inside the production budget?", "",
        f"{len(availability)} cases, one attempt each, cache cleared before "
        f"each call, budget {settings.llm_timeout_exam_s:g} s "
        "(`LLM_TIMEOUT_EXAM_S`). No retries.",
        "",
        "| | value |", "|---|---:|",
        f"| cases sent to the model | {len(attempted)} |",
        f"| answered inside budget | {len(replied)} |",
        f"| timed out | {timeouts} |",
        f"| timeout rate | "
        f"{timeouts / len(attempted):.0%} |" if attempted else "| timeout rate | — |",
        f"| p50 turn latency, all attempts | {percentile(live, 0.50):.2f} s |",
        f"| p90 turn latency, all attempts | {percentile(live, 0.90):.2f} s |",
        f"| p50 of the replies that landed | {percentile(replied, 0.50):.2f} s |",
        f"| p90 of the replies that landed | {percentile(replied, 0.90):.2f} s |",
        "",
        "Two pairs of percentiles, because they answer different things. The "
        "all-attempts pair is what a caller waits — a timed-out call "
        "contributes the budget, so once the timeout rate passes half, the "
        "p50 *is* the budget and says nothing about the model. The replies "
        "pair is the model's real latency, and comparing it with the budget "
        "is what shows whether the budget is set anywhere useful.",
        "",
        "## Reading the two together", "",
        "The quality run says whether the shortlist-plus-model idea works. "
        "The availability run says how often a caller gets it. They must not "
        "be multiplied into a single accuracy figure, because the gazetteer "
        "answer is what ships when the model is late — and that number is in "
        "`exam_eval.md`.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", type=Path, default=settings.paths.root / "reports")
    parser.add_argument("--t1-sample", type=int, default=40,
                        help="how many T1 fallback cases to include")
    parser.add_argument("--availability", type=int, default=20,
                        help="how many cases to re-ask under the production budget")
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args()

    rules = replay_rules(args.reports / "stt_results.csv")
    exams = read_csv(settings.paths.raw / "exams.csv")
    payload = build(exams, read_csv(settings.paths.raw / "confusables.csv"), ["fr"])

    with TemporaryDirectory(prefix="hermes-exam-llm-") as directory:
        temp = Path(directory)
        path = temp / "exam_index.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        index = exam_bot.load_index(path)
        source = settings.paths.processed / settings.lexicon_file
        if source.is_file():
            shutil.copyfile(source, temp / settings.lexicon_file)
        set_settings(replace(settings, paths=replace(settings.paths, processed=temp)))

        print(f"client ready: {llm.warm()}", flush=True)
        cases = samples(settings.paths.raw, exams, rules, index)
        pairs = fallback_cases(cases, index, args.t1_sample, args.seed)
        print(f"{len(pairs)} fallback cases "
              f"({Counter(c.tier for c, _ in pairs)})", flush=True)

        print("quality run:", flush=True)
        quality = quality_run(pairs, index)
        print("availability run:", flush=True)
        availability = availability_run(pairs, index, args.availability, args.seed)
        current = get_settings()

    rows = [
        {**{k: v for k, v in vars(call).items() if k != "expected"},
         "run": run, "expected": "|".join(sorted(call.expected))}
        for run, calls in (("quality", quality), ("availability", availability))
        for call in calls
    ]
    with (args.reports / "exam_llm_eval.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.reports / "exam_llm_eval.md").write_text(
        report(quality, availability, current), encoding="utf-8",
    )
    print(f"\nwrote {args.reports / 'exam_llm_eval.md'}")
    print(llm.USAGE.summary())


if __name__ == "__main__":
    main()
