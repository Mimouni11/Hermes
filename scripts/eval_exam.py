"""Run WP4 T1–T4 and write paired per-case and summary reports."""

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
from statistics import median
from tempfile import TemporaryDirectory

import jiwer

from build_exam_index import build, read_csv
from calibrate_exam import hold_out
from hermes import exam_bot, llm
from hermes.config import get_settings, set_settings
from hermes.stt import normalise_for_wer


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


def samples(raw: Path, exams: list[dict[str, str]], rules: dict[str, str]) -> list[Sample]:
    keywords = {
        norm for row in exams for phrase in row["keywords_fr"].split("|")
        if (norm := normalise_for_wer(phrase))
    }
    t1 = [
        Sample("T1", f"{row['id']}-{i}", phrase.strip(), frozenset([row["id"]]))
        for row in exams
        for i, phrase in enumerate(dict.fromkeys(row["keywords_fr"].split("|")), start=1)
        if phrase.strip()
    ]
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


def report(rows: list[dict], rules: dict[str, str], warm: bool, settings) -> str:
    lines = [
        "# WP4 exam identification evaluation", "",
        f"Score floor {settings.exam_fuzzy_threshold}, margin {settings.exam_margin:g}, "
        f"fallback floor {settings.exam_llm_threshold}, "
        f"budget {settings.llm_timeout_s:g} s per model request.",
        f"The configured model is `{settings.gemini_model}`. Client warm-up succeeded: {warm}.",
        "The run started with an empty isolated cache, shared within this evaluation.", "",
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
    lines.extend([
        "", "## Fallback usage", "",
        "| Set | Invoked / rows | Cache hits | Live replies | Timeouts / uncached "
        "| Other outcomes | Median uncached turn time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for tier in ("T1", "T2", "T3", "T4"):
        run = [r for r in groups[tier] if r["mode"] == "LLM enabled"]
        if not run:
            continue
        invoked = [r for r in run if r["used_llm"]]
        counts = Counter(r["llm"] for r in invoked)
        uncached = [r["latency_s"] for r in invoked if r["llm"] != "cache"]
        other = len(uncached) - counts["llm"] - counts["timeout"]
        latency = f"{median(uncached):.3f} s" if uncached else "—"
        timeout = f"{counts['timeout']}/{len(uncached)}" if uncached else "—"
        lines.append(
            f"| {tier} | {len(invoked)}/{len(run)} | {counts['cache']} | {counts['llm']} "
            f"| {timeout} | {other} | {latency} |"
        )
    changed = Counter(r["replay"] for r in groups["T4 changed"] if r["mode"] == "Gazetteer")
    lines.extend([
        "", "## Reading the results", "",
        "- T1 removes the normalized and repaired query from all catalogue bases before ranking.",
        "- T2 contains source-corpus wording. T3 uses hand-written requests. The overlap audit "
        "compares contiguous keyword sequences after the project's WER normalization.",
        "- T4 is a **text replay**, using one longest observed substitution per matching request "
        "and retaining unchanged requests as controls. The rules come from accepted small/raw "
        "transcripts in `reports/stt_results.csv`. Adjacent edits are kept together. Standalone "
        "insertions and deletions, and clips rejected by the audio gate, are outside this "
        "replay. Its edit frequency does not "
        "estimate caller STT error rates.",
        f"- Changed T4 inputs ({sum(changed.values())}/64): "
        + ", ".join(f"`{rule}` × {count}" for rule, count in sorted(changed.items())) + ".",
        "- The median measures matcher wall time for uncached fallback invocations, including "
        "timed-out calls. A live reply can contain a rejection or abstention.",
        f"- Observed substitution rules available to the replay: {len(rules)}.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-llm", action="store_true", help="only measure the gazetteer")
    parser.add_argument("--reports", type=Path, default=settings.paths.root / "reports")
    args = parser.parse_args()
    rules = replay_rules(args.reports / "stt_results.csv")
    exams = read_csv(settings.paths.raw / "exams.csv")
    cases = samples(settings.paths.raw, exams, rules)
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
        warm = False if args.no_llm else llm.warm()
        print(f"Client ready: {warm}", flush=True)
        for count, case in enumerate(cases, start=1):
            catalogue = hold_out(index, case.text) if case.tier == "T1" else index
            results.append(evaluate(case, catalogue, use_llm=False))
            if not args.no_llm:
                results.append(evaluate(case, catalogue, use_llm=True))
            if count % 50 == 0 or count == len(cases):
                print(f"{count}/{len(cases)} | {llm.USAGE.summary()}", flush=True)

    suffix = "_gazetteer" if args.no_llm else ""
    with (args.reports / f"exam_eval{suffix}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (args.reports / f"exam_eval{suffix}.md").write_text(
        report(results, rules, warm, settings), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
