"""Measure the WP4 gazetteer on T1–T4. No model is called.

Deliberately model-free. An earlier version ran each case twice, with and
without the LLM, and wrote both columns into one accuracy table. The LLM half
turned out to be measuring a rate limit: 254 of 292 T1 invocations came back
`error:LLMError` after the burst exhausted the free-tier quota, and the whole
"LLM enabled" gain on T3 was one successful call.

So the two questions are now asked separately. This script answers "how good
is the matcher on its own", which is also what the demo falls back to whenever
the model is slow. `eval_exam_llm.py` answers "does the model help, and is it
there when needed".

    uv run python scripts/eval_exam.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory

import jiwer

from build_exam_index import build, read_csv
from calibrate_exam import hold_out, survives_holdout
from hermes.understand import exam_bot
from hermes.config import get_settings, set_settings
from hermes.listen.stt import normalise_for_wer


@dataclass(frozen=True)
class Sample:
    tier: str
    id: str
    text: str
    expected: frozenset[str]
    keyword_overlap: bool = False
    replay: str = ""


def replay_rules(path: Path) -> dict[str, str]:
    """Learn the most frequent replacement per span from accepted small/raw clips."""
    observations: dict[str, Counter[str]] = defaultdict(Counter)
    clips: dict[tuple[str, str, str], None] = {}
    for row in read_csv(path):
        if (row["model"], row["variant"], row["gate_verdict"]) != ("small", "raw", "ok"):
            continue
        if row["reference"] != row["hypothesis"]:
            key = row["clip_id"].split("-snr", 1)[0], row["reference"], row["hypothesis"]
            clips[key] = None
    for _, reference, hypothesis in clips:
        aligned = jiwer.process_words(reference, hypothesis)
        ref, hyp = aligned.references[0], aligned.hypotheses[0]
        for equal, chunks in groupby(aligned.alignments[0], key=lambda chunk: chunk.type == "equal"):
            edit = list(chunks)
            if equal or not any(chunk.type == "substitute" for chunk in edit):
                continue
            # Adjacent chunks can encode one error, such as "mon scanner" to "monscaner".
            original = " ".join(ref[edit[0].ref_start_idx:edit[-1].ref_end_idx])
            replacement = " ".join(hyp[edit[0].hyp_start_idx:edit[-1].hyp_end_idx])
            observations[original][replacement] += 1
    if not observations:
        raise ValueError(f"No substitutions in accepted small/raw rows of {path}")
    return {
        original: min(counts, key=lambda replacement: (-counts[replacement], replacement))
        for original, counts in observations.items()
    }


def replay(sample: Sample, rules: dict[str, str]) -> Sample:
    # Use one longest matching observed span. Keep unchanged controls in the denominator.
    tokens = list(re.finditer(r"[^\W_]+", sample.text))
    words = [token.group().lower() for token in tokens]
    for original in sorted(rules, key=lambda item: (-len(item.split()), item)):
        span = original.split()
        end = len(span)
        for start in range(len(words) - end + 1):
            if words[start:start + end] == span:
                replacement = rules[original]
                text = (
                    sample.text[:tokens[start].start()] + replacement
                    + sample.text[tokens[start + end - 1].end():]
                )
                return replace(sample, tier="T4", text=text, replay=f"{original} → {replacement}")
    return replace(sample, tier="T4")


def samples(
    raw: Path,
    exams: list[dict[str, str]],
    rules: dict[str, str],
    index: exam_bot.ExamIndex,
) -> list[Sample]:
    keywords = {
        norm for row in exams for phrase in row["keywords_fr"].split("|")
        if (norm := normalise_for_wer(phrase))
    }
    # Samples whose base the holdout would empty are excluded, not scored:
    # see `calibrate_exam.survives_holdout`. Scoring them counted a deleted
    # answer as a wrong resolution.
    t1: list[Sample] = []
    t1_excluded = 0
    for row in exams:
        for i, phrase in enumerate(dict.fromkeys(row["keywords_fr"].split("|")), start=1):
            if not phrase.strip():
                continue
            expected = frozenset([row["id"]])
            if not survives_holdout(index, phrase.strip(), expected):
                t1_excluded += 1
                continue
            t1.append(Sample("T1", f"{row['id']}-{i}", phrase.strip(), expected))
    print(f"T1: {len(t1)} answerable samples, {t1_excluded} excluded by the holdout",
          flush=True)
    t2 = [
        Sample("T2", row["id"], row["text"], frozenset([row["exam_id"]]))
        for row in read_csv(raw / "intents.csv")
        if row["intent"] == "new_appointment" and row["lang"] in {"fr", "mix"} and row["exam_id"]
    ]
    t3 = []
    for row in read_csv(raw / "exam_test_t3.csv"):
        expected = frozenset(filter(None, [row["exam_id"], *row["alt_ids"].split("|")]))
        if row["expected_status"] == "clarify":
            expected = frozenset()
        text = f" {normalise_for_wer(row['text'])} "
        overlap = any(f" {keyword} " in text for keyword in keywords)
        t3.append(Sample("T3", row["id"], row["text"], expected, overlap))
    # Measure the small final sets first, then the 905 catalogue probes.
    return [*t2, *t3, *(replay(case, rules) for case in t2 + t3), *t1]


def evaluate(case: Sample, index: exam_bot.ExamIndex, use_llm: bool) -> dict:
    started = time.perf_counter()
    match = exam_bot.identify(case.text, index, use_llm=use_llm)
    elapsed = time.perf_counter() - started
    if match.exam_id is None:
        outcome = "withheld" if case.expected else "clarified"
    elif not case.expected:
        outcome = "premature"
    else:
        outcome = "correct" if match.exam_id in case.expected else "wrong"
    return {
        "tier": case.tier, "id": case.id, "mode": "LLM enabled" if use_llm else "Gazetteer",
        "expected": "|".join(sorted(case.expected)), "keyword_overlap": case.keyword_overlap,
        "replay": case.replay, "text": case.text, "repaired": match.repaired,
        "status": match.status.value, "exam_id": match.exam_id, "outcome": outcome,
        "llm": next((entry[4:] for entry in match.trace if entry.startswith("llm:")), ""),
        "used_llm": match.used_llm, "latency_s": round(elapsed, 4),
        "trace": " | ".join(match.trace),
    }


def report(
    rows: list[dict],
    rules: dict[str, str],
    settings,
    catalogue: dict[str, dict] | None = None,
) -> str:
    catalogue = catalogue or {}
    lines = [
        "# WP4 exam identification — the gazetteer", "",
        "**No model is called in this run.** Every number below is the "
        "gazetteer and the decision layer on their own, which is what the "
        "demo falls back to whenever the model is slow or unavailable.",
        "",
        "The LLM fallback is measured separately, in `exam_llm_eval.md`: "
        "mixing the two put a rate-limited API inside the accuracy table and "
        "made the gazetteer look worse than it is.",
        "",
        f"Score floor {settings.exam_fuzzy_threshold}, "
        f"margin {settings.exam_margin:g}, "
        f"fallback band floor {settings.exam_llm_threshold}.",
        "",
        "| Set | Mode | Correct IDs / labelled | Wrong IDs / labelled | Clarified / expected "
        "| Premature IDs | Clarify rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    groups = {tier: [r for r in rows if r["tier"] == tier] for tier in ("T1", "T2", "T3", "T4")}
    groups["T3 without keyword overlap"] = [
        r for r in groups["T3"] if r["expected"] and not r["keyword_overlap"]
    ]
    groups["T4 changed"] = [r for r in groups["T4"] if r["replay"]]
    for tier, group in groups.items():
        for mode in dict.fromkeys(r["mode"] for r in group):
            run = [r for r in group if r["mode"] == mode]
            counts = Counter(r["outcome"] for r in run)
            labelled = sum(bool(r["expected"]) for r in run)
            clarified = sum(r["status"] == "clarify" for r in run)
            expected_clarifications = len(run) - labelled
            clarification = (
                f"{counts['clarified']}/{expected_clarifications}" if expected_clarifications else "—"
            )
            lines.append(
                f"| {tier} | {mode} | {counts['correct']}/{labelled} "
                f"| {counts['wrong']}/{labelled} | {clarification} "
                f"| {counts['premature']} | {clarified}/{len(run)} |"
            )
    # Per-modality, which the brief asks for. Per-language is not reported:
    # the scope is French only, so every row would be in one bucket. Grouped
    # by the *expected* exam's modality, so a miss is counted against the
    # modality that was asked for rather than the one that won.
    lines.extend(["", "## By modality", "",
                  "T2, T3 and T4 together, keyed on the expected exam.", "",
                  "| Modality | Labelled | Correct | Wrong | Asked |",
                  "|---|---:|---:|---:|---:|"])
    buckets: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row["tier"] == "T1" or not row["expected"]:
            continue
        first = row["expected"].split("|")[0]
        modality = catalogue.get(first, {}).get("modality", "?")
        buckets[modality]["labelled"] += 1
        buckets[modality][row["outcome"]] += 1
        if row["status"] == "clarify":
            buckets[modality]["asked"] += 1
    for modality, counts in sorted(buckets.items(), key=lambda kv: -kv[1]["labelled"]):
        lines.append(
            f"| {modality} | {counts['labelled']} | {counts['correct']} "
            f"| {counts['wrong']} | {counts['asked']} |"
        )

    changed = Counter(r["replay"] for r in groups["T4 changed"] if r["mode"] == "Gazetteer")
    asked = sum(r["status"] == "clarify" for r in rows if r["tier"] in ("T3", "T4"))
    lines.extend([
        "", "## Reading the results", "",
        f"- {asked} T3/T4 rows end in a question. The LLM fallback can only "
        "act on the subset of those whose top score also sits in the band, "
        "counted exactly in `exam_llm_eval.md`. No other row in this table "
        "can change when the model is enabled.",
        "- T1 removes the normalized and repaired query from all catalogue bases before "
        "ranking, and **excludes the samples whose base that would empty**. For a lateral "
        "pair every French keyword collapses to one phrase, so holding it out deletes the "
        "base and the exam becomes unfindable by construction.",
        "- That exclusion moved the T1 denominator from 905 to 841 and the clarify count "
        "from 537 to 473. It left the correct and wrong counts untouched: every excluded "
        "sample had been scored as withheld, not as a wrong resolution. So the margin "
        "sweep that chose 8 was reading an inflated clarify rate, not an inflated error "
        "rate, and re-running it on the corrected sample selected the same thresholds.",
        "- T2 contains source-corpus wording. T3 uses hand-written requests. The overlap audit "
        "compares contiguous keyword sequences after the project's WER normalization.",
        "- T4 is a **text replay**, using one longest observed substitution per matching request "
        "and retaining unchanged requests as controls. The rules come from accepted small/raw "
        "transcripts in `reports/stt_results.csv`. Adjacent edits are kept together. Standalone "
        "insertions and deletions, and clips rejected by the audio gate, are outside this "
        "replay. Its edit frequency does not "
        "estimate caller STT error rates.",
        f"- Changed T4 inputs ({sum(changed.values())}/{len(groups['T4'])}): "
        + ", ".join(f"`{rule}` × {count}" for rule, count in sorted(changed.items())) + ".",
        f"- Observed substitution rules available to the replay: {len(rules)}.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=settings.paths.root / "reports")
    args = parser.parse_args()
    rules = replay_rules(args.reports / "stt_results.csv")
    exams = read_csv(settings.paths.raw / "exams.csv")
    payload = build(exams, read_csv(settings.paths.raw / "confusables.csv"), ["fr"])
    results = []
    with TemporaryDirectory(prefix="hermes-exam-eval-") as directory:
        temp = Path(directory)
        path = temp / "exam_index.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        index = exam_bot.load_index(path)
        source = settings.paths.processed / settings.lexicon_file
        if source.is_file():
            shutil.copyfile(source, temp / settings.lexicon_file)
        set_settings(replace(settings, paths=replace(settings.paths, processed=temp)))
        cases = samples(settings.paths.raw, exams, rules, index)
        for count, case in enumerate(cases, start=1):
            catalogue = hold_out(index, case.text) if case.tier == "T1" else index
            results.append(evaluate(case, catalogue, use_llm=False))
            if count % 200 == 0 or count == len(cases):
                print(f"{count}/{len(cases)}", flush=True)

    with (args.reports / "exam_eval.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (args.reports / "exam_eval.md").write_text(
        report(results, rules, settings, payload["exams"]), encoding="utf-8",
    )
    print(f"wrote {args.reports / 'exam_eval.md'}")


if __name__ == "__main__":
    main()
