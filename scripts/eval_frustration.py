"""Evaluate frustration detection on the French rows. Step 1: the lexicon.

Scored on the **same grouped folds** the models will use, so the comparison in
step 2 is like for like rather than a word list measured one way and a
classifier another.

The lexicon has nothing to fit, so folding does not change its predictions —
what the folds buy is the spread. A baseline quoted as one number hides how
much of its score depends on which rows it happened to see, and that spread is
exactly what step 2's "within noise" judgement needs to be measured against.

    uv run python scripts/eval_frustration.py
    uv run python scripts/eval_frustration.py --folds 5 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hermes import frustration  # noqa: E402

#: Kinds whose counts are reported as "n/total" and never as a percentage.
#: With 7 to 12 rows each, a percentage of eleven invites a confidence the
#: data cannot support.
HARD_KINDS = ("correction", "worry", "other_complaint", "frustrated_hard", "positive")


def prf(predicted: list[int], actual: list[int]) -> tuple[float, float, float]:
    """Precision, recall and F1 for the positive class."""
    tp = sum(p == 1 and a == 1 for p, a in zip(predicted, actual))
    fp = sum(p == 1 and a == 0 for p, a in zip(predicted, actual))
    fn = sum(p == 0 and a == 1 for p, a in zip(predicted, actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def folds(rows: list[dict[str, str]], splits: int, seed: int) -> list[list[int]]:
    """Grouped, stratified test folds over the `group` column.

    Grouped because 75 of the 180 French rows sit in a paraphrase group — six
    wordings of "you wasted my time" and the like. An ungrouped split trains on
    some of those and tests on the rest, then calls the result generalisation.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.array([int(row["label"]) for row in rows])
    groups = np.array([row["group"] for row in rows])
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    return [
        test.tolist()
        for _train, test in splitter.split(np.zeros(len(rows)), labels, groups)
    ]


def ungrouped_folds(rows: list[dict[str, str]], splits: int, seed: int) -> list[list[int]]:
    """The leaky split, reported once to show what grouping costs."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    labels = np.array([int(row["label"]) for row in rows])
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    return [test.tolist() for _train, test in splitter.split(np.zeros(len(rows)), labels)]


def score_baseline(
    rows: list[dict[str, str]], test_folds: list[list[int]],
) -> dict[str, object]:
    """Run the lexicon over every fold and pool the predictions."""
    predicted = [int(frustration.lexicon_flag(row["text"])) for row in rows]
    actual = [int(row["label"]) for row in rows]

    per_fold = [
        prf([predicted[i] for i in fold], [actual[i] for i in fold])
        for fold in test_folds
    ]
    return {
        "pooled": prf(predicted, actual),
        "per_fold": per_fold,
        "predicted": predicted,
        "actual": actual,
    }


def report(rows: list[dict[str, str]], result: dict[str, object], leak: tuple) -> None:
    predicted = result["predicted"]
    actual = result["actual"]
    precision, recall, f1 = result["pooled"]
    per_fold = result["per_fold"]

    print(f"\n{'=' * 64}")
    print("STEP 1 — French lexicon baseline")
    print(f"{'=' * 64}")
    print(f"{len(rows)} French rows: "
          f"{sum(actual)} frustrated, {len(actual) - sum(actual)} calm")
    terms = sum(len(t) for t in frustration.FRUSTRATION_TERMS.values())
    print(f"{terms} terms in {len(frustration.FRUSTRATION_TERMS)} families\n")

    print("pooled over all rows")
    print(f"  precision {precision:.2f}   recall {recall:.2f}   F1 {f1:.2f}")

    print("\nper grouped fold (mean +/- population sd)")
    for name, index in (("precision", 0), ("recall", 1), ("F1", 2)):
        values = [f[index] for f in per_fold]
        print(f"  {name:<10} {mean(values):.2f} +/- {pstdev(values):.2f}"
              f"   folds: {', '.join(f'{v:.2f}' for v in values)}")

    print(f"\nungrouped split, for the leak gap only")
    print(f"  F1 {leak[2]:.2f} grouped-vs-ungrouped difference "
          f"{abs(leak[2] - f1):.2f}")
    print("  (a lexicon cannot leak — it fits nothing — so this difference is"
          "\n   fold composition alone, and is the floor for reading step 2's gap)")

    # Confusion, then the kinds.
    tp = sum(p == 1 and a == 1 for p, a in zip(predicted, actual))
    fp = sum(p == 1 and a == 0 for p, a in zip(predicted, actual))
    fn = sum(p == 0 and a == 1 for p, a in zip(predicted, actual))
    tn = sum(p == 0 and a == 0 for p, a in zip(predicted, actual))
    print(f"\nconfusion: tp {tp}  fp {fp}  fn {fn}  tn {tn}")

    print("\nflagged, by kind (counts, never percentages)")
    totals = Counter(row["kind"] for row in rows)
    flagged = Counter(
        row["kind"] for row, p in zip(rows, predicted) if p == 1
    )
    for kind in sorted(totals, key=lambda k: (-totals[k], k)):
        label = next(int(r["label"]) for r in rows if r["kind"] == kind)
        want = "should flag" if label == 1 else "should NOT flag"
        mark = " <-- hard kind" if kind in HARD_KINDS else ""
        print(f"  {kind:<17} {flagged[kind]:>2}/{totals[kind]:<3} {want}{mark}")

    print("\nmissed frustration (false negatives), by kind")
    for kind, count in Counter(
        row["kind"] for row, p, a in zip(rows, predicted, actual)
        if a == 1 and p == 0
    ).most_common():
        print(f"  {kind:<17} {count}")

    print("\nfalse alarms, with the term that fired")
    for row, p, a in zip(rows, predicted, actual):
        if p == 1 and a == 0:
            hits = ", ".join(frustration.lexicon_hits(row["text"]))
            print(f"  [{row['kind']}] {row['text']}")
            print(f"      fired on: {hits}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = frustration.load_dataset(lang="fr")
    if not rows:
        print("no French rows in frustration.csv")
        return 1

    grouped = folds(rows, args.folds, args.seed)
    result = score_baseline(rows, grouped)
    leak_result = score_baseline(rows, ungrouped_folds(rows, args.folds, args.seed))
    report(rows, result, leak_result["pooled"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
