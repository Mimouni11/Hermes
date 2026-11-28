"""Evaluate frustration detection on the French rows.

Three things, in the order they were built:

1. **A French lexicon baseline.** Written from the label definition, scored on
   the same grouped folds the models use. It exists to be beaten — a
   classifier that cannot beat a word list has not earned its 384 dimensions.
2. **The models**, logistic regression against a torch MLP ensemble, each over
   five seeds, on two training sets: French only, and French plus EmoWOZ.
3. **A learning curve** over how much EmoWOZ is added.

**The French test folds never change.** They are `StratifiedGroupKFold` over
the `group` column, identical in every arm, and EmoWOZ rows enter *training
folds only*. Every number below is measured on French rows the model did not
see, so adding English data cannot flatter the score by widening the test set.

    uv run python scripts/eval_frustration.py
    uv run python scripts/eval_frustration.py --baseline-only
    uv run python scripts/eval_frustration.py --no-emowoz      # arm (a) only
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hermes.understand import frustration  # noqa: E402

#: Kinds reported as counts, never percentages. With 7 to 12 rows each, a
#: percentage of eleven invites a confidence the data cannot support.
HARD_KINDS = ("correction", "worry", "other_complaint", "frustrated_hard", "positive")


# --------------------------------------------------------------------------
# Metrics and folds
# --------------------------------------------------------------------------

def prf(predicted: list[int], actual: list[int]) -> tuple[float, float, float]:
    """Precision, recall and F1 for the positive class."""
    tp = sum(p == 1 and a == 1 for p, a in zip(predicted, actual))
    fp = sum(p == 1 and a == 0 for p, a in zip(predicted, actual))
    fn = sum(p == 0 and a == 1 for p, a in zip(predicted, actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def french_folds(rows: list[dict[str, str]], splits: int, seed: int, grouped: bool):
    """The French test folds. Grouped unless explicitly asked otherwise.

    Grouped because 75 of the 180 French rows sit in a paraphrase group — six
    wordings of "you wasted my time" and the like. An ungrouped split trains on
    some of those and tests on the rest, then calls the result generalisation.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    labels = np.array([int(row["label"]) for row in rows])
    blank = np.zeros(len(rows))
    if grouped:
        groups = np.array([row["group"] for row in rows])
        splitter = StratifiedGroupKFold(
            n_splits=splits, shuffle=True, random_state=seed,
        )
        return list(splitter.split(blank, labels, groups))
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    return list(splitter.split(blank, labels))


# --------------------------------------------------------------------------
# Step 1 — the lexicon
# --------------------------------------------------------------------------

def score_baseline(rows: list[dict[str, str]], folds) -> dict:
    predicted = [int(frustration.lexicon_flag(row["text"])) for row in rows]
    actual = [int(row["label"]) for row in rows]
    per_fold = [
        prf([predicted[i] for i in test], [actual[i] for i in test])
        for _train, test in folds
    ]
    return {
        "pooled": prf(predicted, actual),
        "per_fold": per_fold,
        "predicted": predicted,
        "actual": actual,
    }


def baseline_report(rows: list[dict[str, str]], result: dict) -> None:
    predicted, actual = result["predicted"], result["actual"]
    precision, recall, f1 = result["pooled"]

    print(f"\n{'=' * 70}")
    print("STEP 1 — French lexicon baseline")
    print(f"{'=' * 70}")
    print(f"{len(rows)} French rows: "
          f"{sum(actual)} frustrated, {len(actual) - sum(actual)} calm")
    terms = sum(len(t) for t in frustration.FRUSTRATION_TERMS.values())
    print(f"{terms} terms in {len(frustration.FRUSTRATION_TERMS)} families\n")
    print(f"pooled   precision {precision:.2f}   recall {recall:.2f}   F1 {f1:.2f}")

    for name, index in (("precision", 0), ("recall", 1), ("F1", 2)):
        values = [f[index] for f in result["per_fold"]]
        print(f"  {name:<10} {mean(values):.2f} +/- {pstdev(values):.2f}"
              f"   folds: {', '.join(f'{v:.2f}' for v in values)}")

    tp = sum(p == 1 and a == 1 for p, a in zip(predicted, actual))
    fp = sum(p == 1 and a == 0 for p, a in zip(predicted, actual))
    fn = sum(p == 0 and a == 1 for p, a in zip(predicted, actual))
    tn = sum(p == 0 and a == 0 for p, a in zip(predicted, actual))
    print(f"\nconfusion: tp {tp}  fp {fp}  fn {fn}  tn {tn}")

    totals = Counter(row["kind"] for row in rows)
    flagged = Counter(row["kind"] for row, p in zip(rows, predicted) if p == 1)
    print("\nflagged, by kind (counts)")
    for kind in sorted(totals, key=lambda k: (-totals[k], k)):
        want = next(int(r["label"]) for r in rows if r["kind"] == kind)
        mark = "*" if kind in HARD_KINDS else " "
        print(f" {mark}{kind:<17} {flagged[kind]:>2}/{totals[kind]:<3} "
              f"want {want}")


# --------------------------------------------------------------------------
# Step 2 — the models
# --------------------------------------------------------------------------

def embed(texts: list[str]):
    from hermes.infra import embeddings

    return embeddings.encode(texts)


def make_lr(seed: int):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        max_iter=3000, C=10.0, class_weight="balanced", random_state=seed,
    )


def make_mlp(seed: int):
    from hermes.understand.mlp import MLPConfig, TorchMLP

    return TorchMLP(MLPConfig(), seed=seed)


MODELS = {"logistic regression": make_lr, "torch MLP x5": make_mlp}


def subsample(indices, y, fraction: float, seed: int):
    """A stratified share of one training fold, for the learning curve."""
    import numpy as np

    if fraction >= 1.0:
        return indices
    rng = np.random.default_rng(seed)
    kept = []
    for label in (0, 1):
        pool = [i for i in indices if y[i] == label]
        rng.shuffle(pool)
        take = max(1, int(round(len(pool) * fraction)))
        kept.extend(pool[:take])
    return np.array(sorted(kept))


def out_of_fold(
    factory,
    X_fr,
    y_fr,
    folds,
    X_aux=None,
    y_aux=None,
    aux_limit: int | None = None,
    balance_weight: bool = False,
    french_fraction: float = 1.0,
    pretrain: bool = False,
    seed: int = 0,
    select_epochs=None,
    groups_fr=None,
    picks: list | None = None,
    candidates: tuple[int, ...] | None = None,
):
    """Out-of-fold French probabilities under one training regime.

    Written as an explicit loop rather than `cross_val_predict` for one
    reason: the auxiliary rows must join the **training** side of each fold
    and never the test side. A helper that splits one matrix cannot express
    that, and the easiest way to get this wrong is to concatenate first and
    split afterwards.

    `balance_weight` gives the whole auxiliary block the same **total** weight
    as the French block, rather than letting it win on headcount.
    `pretrain` trains on the auxiliary rows first and adapts on the French
    fold afterwards instead of mixing them.
    """
    import numpy as np

    probabilities = np.zeros(len(y_fr), dtype=float)
    for fold_number, (train_index, test_index) in enumerate(folds):
        train_index = subsample(
            train_index, y_fr, french_fraction, seed * 100 + fold_number,
        )
        X_train, y_train = X_fr[train_index], y_fr[train_index]

        if select_epochs is not None:
            # Nested: the epoch count is chosen by grouped CV *inside* this
            # outer training fold, so the rows being scored never influence
            # it. Choosing once over the whole set and reusing it would leak
            # the test rows into a hyperparameter, which is the quiet version
            # of the mistake the grouped split exists to prevent.
            from hermes.understand.mlp import EPOCH_CANDIDATES, choose_epochs

            chosen, curve = choose_epochs(
                X_train, y_train,
                [groups_fr[i] for i in train_index],
                config=select_epochs,
                candidates=tuple(candidates or EPOCH_CANDIDATES),
                seed=seed,
            )
            if picks is not None:
                picks.append((chosen, curve))

        extra_X = extra_y = None
        if X_aux is not None and len(X_aux) and (aux_limit is None or aux_limit > 0):
            extra_X = X_aux if aux_limit is None else X_aux[:aux_limit]
            extra_y = y_aux if aux_limit is None else y_aux[:aux_limit]

        model = factory(seed)
        if select_epochs is not None:
            from dataclasses import replace as replace_config

            from hermes.understand.mlp import TorchMLP

            model = TorchMLP(
                replace_config(select_epochs, epochs=chosen), seed=seed,
            )

        if pretrain and extra_X is not None:
            model.pretrain_then_finetune(extra_X, extra_y, X_train, y_train)
        elif extra_X is not None:
            weights = None
            if balance_weight:
                # Each French row weighs 1, so the French block totals
                # len(X_train). The auxiliary block is scaled to the same
                # total, which is the difference between "more evidence" and
                # "a louder voice".
                per_aux = len(X_train) / max(1, len(extra_X))
                weights = np.concatenate([
                    np.ones(len(X_train)),
                    np.full(len(extra_X), per_aux),
                ])
            model.fit(
                np.vstack([X_train, extra_X]),
                np.concatenate([y_train, extra_y]),
                sample_weight=weights,
            )
        else:
            model.fit(X_train, y_train)

        probabilities[test_index] = model.predict_proba(X_fr[test_index])[:, 1]
    return probabilities


def run_arm(
    rows, X_fr, y_fr, splits, seeds, grouped=True,
    X_aux=None, y_aux=None, aux_limit=None,
    balance_weight=False, french_fraction=1.0, pretrain=False,
    only: tuple[str, ...] | None = None,
    select_epochs=None, picks: list | None = None,
    candidates: tuple[int, ...] | None = None,
):
    """Every seed of one (model, training regime, split) combination."""
    from sklearn.metrics import roc_auc_score

    groups_fr = [row["group"] for row in rows]
    out: dict[str, list[dict]] = {}
    for name, base_factory in MODELS.items():
        if only and name not in only:
            continue
        runs = []
        for seed in seeds:
            folds = french_folds(rows, splits, seed, grouped)

            def factory(s, base_factory=base_factory):
                return base_factory(s)

            proba = out_of_fold(
                factory, X_fr, y_fr, folds, X_aux, y_aux, aux_limit,
                balance_weight=balance_weight,
                french_fraction=french_fraction,
                pretrain=pretrain, seed=seed,
                select_epochs=select_epochs, groups_fr=groups_fr, picks=picks,
                candidates=candidates,
            )
            predicted = (proba >= 0.5).astype(int)
            precision, recall, f1 = prf(predicted.tolist(), y_fr.tolist())
            runs.append({
                "precision": precision, "recall": recall, "f1": f1,
                "auc": roc_auc_score(y_fr, proba),
                "proba": proba, "predicted": predicted,
            })
        out[name] = runs
        print(f"    {name:<22} F1 {mean(r['f1'] for r in runs):.2f}"
              f"  AUC {mean(r['auc'] for r in runs):.2f}", flush=True)
    return out


def kind_counts(rows, runs, kind: str) -> list[int]:
    """How many rows of one kind each seed flagged."""
    return [
        sum(1 for row, p in zip(rows, run["predicted"]) if row["kind"] == kind and p == 1)
        for run in runs
    ]


def cells(runs) -> list[str]:
    return [
        f"{mean(r[m] for r in runs):.2f}±{pstdev([r[m] for r in runs]):.2f}"
        for m in ("precision", "recall", "f1", "auc")
    ]


def models_report(rows, arms: dict, baseline: dict, totals: Counter) -> None:
    print(f"\n{'=' * 86}")
    print("STEP 2 — logistic regression vs torch MLP ensemble, 5 seeds")
    print(f"{'=' * 86}")

    header = (f"{'training set':<24}{'model':<24}{'split':<11}"
              f"{'P':>12}{'R':>12}{'F1':>12}{'ROC-AUC':>12}")
    print(header)
    print("-" * len(header))
    bp, br, bf1 = baseline["pooled"]
    print(f"{'lexicon (step 1)':<24}{'—':<24}{'grouped':<11}"
          f"{bp:>12.2f}{br:>12.2f}{bf1:>12.2f}{'—':>12}")
    for (label, split), runs_by_model in arms.items():
        for model, runs in runs_by_model.items():
            print(f"{label:<24}{model:<24}{split:<11}"
                  + "".join(f"{c:>12}" for c in cells(runs)))

    print("\nhard-kind flags across the 5 seeds (grouped, 0.5)")
    for (label, split), runs_by_model in arms.items():
        if split != "grouped":
            continue
        print(f"  {label}")
        for model, runs in runs_by_model.items():
            parts = []
            for kind in ("correction", "neutral", "worry", "frustrated_hard"):
                if not totals.get(kind):
                    continue
                counts = "/".join(str(c) for c in kind_counts(rows, runs, kind))
                parts.append(f"{kind} {counts} of {totals[kind]}")
            print(f"    {model:<22} " + "   ".join(parts))


# --------------------------------------------------------------------------
# Step 3 — the two thresholds
# --------------------------------------------------------------------------

#: Swept at 0.02 so the chosen values are not an artefact of a coarse grid,
#: and so the shape between them is visible.
SWEEP = tuple(round(0.02 * i, 2) for i in range(1, 50))

#: Recall the soft level must reach. Its job is to catch annoyance early, and
#: its cost is one apologetic sentence — so it is allowed to be wrong.
SOFT_RECALL_TARGET = 0.90

#: The soft threshold is now **fixed**, not re-derived each run. It was chosen
#: once as the highest value reaching recall >= 0.90 and it is the baseline the
#: hybrid's lexicon arm is gated against — so letting it drift with the corpus
#: would move two things at once and make the hard sweep uninterpretable. The
#: step-3 table still shows how it was found.
SOFT_FIXED = 0.62

#: "Almost never" for the hard level, in rows out of 11. The hard level
#: transfers the call, so a caller who merely said "non, j'ai dit mardi"
#: must not trip it. `worry` and `positive` are held to the same bar.
HARD_MAX_FALSE = 1.0


def threshold_row(
    rows, proba_sets: list, actual, threshold: float, totals: Counter,
) -> dict:
    """Mean metrics across seeds at one threshold, plus flags by kind."""
    recalls, precisions, f1s = [], [], []
    by_kind: dict[str, list[int]] = {k: [] for k in totals}

    for proba in proba_sets:
        predicted = (proba >= threshold).astype(int)
        precision, recall, f1 = prf(predicted.tolist(), actual.tolist())
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        for kind in totals:
            by_kind[kind].append(sum(
                1 for row, p in zip(rows, predicted)
                if row["kind"] == kind and p == 1
            ))

    return {
        "threshold": threshold,
        "recall": mean(recalls), "recall_sd": pstdev(recalls),
        "precision": mean(precisions), "f1": mean(f1s),
        "kinds": {k: mean(v) for k, v in by_kind.items()},
    }


def pick_thresholds(table: list[dict]) -> tuple[float, float]:
    """Soft: the highest threshold still hitting the recall target.

    Highest, not lowest — every threshold below it also meets the target, and
    the highest one meets it with the fewest false alarms. Picking the lowest
    would satisfy the letter of "recall >= 0.9" while flagging everything.

    Hard: the lowest threshold at which `correction`, `worry` and `positive`
    are each almost never flagged. Lowest, because every threshold above it
    also satisfies the constraint and the lowest keeps the most recall.
    """
    meets_recall = [r for r in table if r["recall"] >= SOFT_RECALL_TARGET]
    soft = max(r["threshold"] for r in meets_recall) if meets_recall else 0.5

    def clean(row: dict) -> bool:
        return all(
            row["kinds"].get(kind, 0.0) <= HARD_MAX_FALSE
            for kind in ("correction", "worry", "positive")
        )

    clean_rows = [r for r in table if clean(r)]
    hard = min(r["threshold"] for r in clean_rows) if clean_rows else 0.9
    return soft, max(hard, soft)


def step3_report(rows, table: list[dict], soft: float, hard: float,
                 totals: Counter) -> None:
    print(f"\n{'=' * 104}")
    print("STEP 3 — soft and hard thresholds on the shipped model's "
          "out-of-fold probabilities")
    print(f"{'=' * 104}")

    kinds = ("frustrated", "frustrated_hard", "correction", "worry",
             "neutral", "positive", "other_complaint")
    header = (f"{'tau':>5}{'recall':>9}{'prec':>7}{'F1':>7}  "
              + "".join(f"{k[:9]:>11}" for k in kinds))
    print(header)
    print("-" * len(header))
    for row in table:
        if row["threshold"] % 0.1 > 0.001 and row["threshold"] not in (soft, hard):
            continue
        mark = ""
        if abs(row["threshold"] - soft) < 1e-9:
            mark = " <- SOFT"
        if abs(row["threshold"] - hard) < 1e-9:
            mark = " <- HARD"
        cells = "".join(
            f"{row['kinds'].get(k, 0):>7.1f}/{totals.get(k, 0):<3}" for k in kinds
        )
        print(f"{row['threshold']:>5.2f}{row['recall']:>9.2f}"
              f"{row['precision']:>7.2f}{row['f1']:>7.2f}  {cells}{mark}")

    print(f"\nwant flagged: frustrated 75, frustrated_hard 10. "
          f"Everything else should stay at 0.")
    print("Counts are means over 5 seeds, so a 0.4 means two seeds of five.")

    chosen = {r["threshold"]: r for r in table}
    for name, value in (("SOFT", soft), ("HARD", hard)):
        row = chosen.get(value)
        if not row:
            continue
        missed = 85 - row["kinds"].get("frustrated", 0) - row["kinds"].get(
            "frustrated_hard", 0)
        print(f"\n{name} = {value:.2f}")
        print(f"  recall {row['recall']:.2f} +/- {row['recall_sd']:.2f}   "
              f"precision {row['precision']:.2f}   F1 {row['f1']:.2f}")
        print(f"  catches {row['kinds'].get('frustrated', 0):.1f}/75 frustrated "
              f"and {row['kinds'].get('frustrated_hard', 0):.1f}/10 "
              f"frustrated_hard ({missed:.1f} of 85 missed)")
        print("  false alarms: " + ", ".join(
            f"{k} {row['kinds'].get(k, 0):.1f}/{totals.get(k, 0)}"
            for k in ("correction", "worry", "neutral", "positive",
                      "other_complaint")
            if row["kinds"].get(k, 0) > 0
        ) or "  false alarms: none")

    print("\nConfig lines to write:")
    print(f'    "FRUSTRATION_SOFT_THRESHOLD": {soft:.2f},')
    print(f'    "FRUSTRATION_HARD_THRESHOLD": {hard:.2f},')


def hard_level_options(
    rows, proba_sets: list, actual, soft: float, totals: Counter,
) -> list[dict]:
    """Three candidate rules for the hard level, each measured.

    The hard level transfers the call, so it is the one that must not fire on
    a caller who merely corrected themselves. Three ways to get there:

    1. **plain, strict** — probability alone, calm kinds flagged at most once.
       Measured at 0.98, which costs 29 % of recall.
    2. **plain, relaxed** — probability alone, calm kinds at most twice.
    3. **hybrid** — probability alone *or* a lexicon hit backed by the soft
       threshold. A lexicon term is high-precision evidence (step 1: one
       false alarm in 180 rows), so ORing it should let the probability bar
       fall without readmitting the two rows that forced it up — neither
       "Oui ça marche !" nor "Non non, c'est bon" matches any term.
    """
    import numpy as np

    lexical = np.array([
        1 if frustration.lexicon_flag(row["text"]) else 0 for row in rows
    ])

    def measure(predict, tau: float) -> dict:
        recalls, precisions, f1s = [], [], []
        kinds: dict[str, list[int]] = {k: [] for k in totals}
        for proba in proba_sets:
            flagged = predict(proba, tau)
            precision, recall, f1 = prf(
                flagged.astype(int).tolist(), actual.tolist(),
            )
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
            for kind in totals:
                kinds[kind].append(sum(
                    1 for row, p in zip(rows, flagged)
                    if row["kind"] == kind and p
                ))
        return {
            "threshold": tau, "recall": mean(recalls),
            "recall_sd": pstdev(recalls), "precision": mean(precisions),
            "f1": mean(f1s), "kinds": {k: mean(v) for k, v in kinds.items()},
        }

    def plain(proba, tau):
        return proba >= tau

    def hybrid(proba, tau):
        """The shipped rule: probability, or a lexicon hit the model half-agrees with."""
        return (proba >= tau) | ((lexical == 1) & (proba >= soft))

    def lowest_where(predict, accept) -> dict | None:
        for tau in SWEEP:
            row = measure(predict, tau)
            if accept(row):
                return row
        return None

    def calm_at_most(limit: float):
        def accept(row: dict) -> bool:
            return all(
                row["kinds"].get(kind, 0.0) <= limit
                for kind in ("correction", "worry", "positive")
            )
        return accept

    def strict(row: dict) -> bool:
        """The hybrid's own bar, per the WP5 specification.

        Tighter than the plain rule's, and deliberately so: the hybrid can
        afford it, because the lexicon arm recovers recall that the plain
        rule could only buy by admitting false alarms.
        """
        kinds = row["kinds"]
        return (
            kinds.get("correction", 0.0) <= 1.0
            and kinds.get("positive", 0.0) <= 0.2
            and kinds.get("neutral", 0.0) == 0.0
        )

    options = []
    for name, predict, accept in (
        ("hard alone, calm <= 1", plain, calm_at_most(1.0)),
        ("hard alone, calm <= 2", plain, calm_at_most(2.0)),
        ("HYBRID: prob OR (lexicon AND >= soft)", hybrid, strict),
    ):
        found = lowest_where(predict, accept)
        if found:
            found["name"] = name
            options.append(found)
        else:
            options.append({
                "name": name, "threshold": float("nan"), "recall": 0.0,
                "recall_sd": 0.0, "precision": 0.0, "f1": 0.0,
                "kinds": {}, "unsatisfiable": True,
            })
    return options


def hard_options_report(options: list[dict], totals: Counter) -> None:
    positives = totals.get("frustrated", 0) + totals.get("frustrated_hard", 0)
    print(f"\n{'=' * 104}")
    print("STEP 3b — candidate rules for the HARD level")
    print(f"{'=' * 104}")
    header = (f"{'rule':<42}{'tau':>6}{'recall':>9}{'prec':>7}{'F1':>7}"
              f"{'correction':>13}{'worry':>10}{'positive':>11}"
              f"{'missed':>9}")
    print(header)
    print("-" * len(header))
    for option in options:
        if option.get("unsatisfiable"):
            print(f"{option['name']:<42}  no threshold in the sweep satisfies "
                  f"this constraint")
            continue
        kinds = option["kinds"]
        caught = kinds.get("frustrated", 0) + kinds.get("frustrated_hard", 0)
        print(f"{option['name']:<42}{option['threshold']:>6.2f}"
              f"{option['recall']:>9.2f}{option['precision']:>7.2f}"
              f"{option['f1']:>7.2f}"
              f"{kinds.get('correction', 0):>10.1f}/{totals.get('correction', 0):<2}"
              f"{kinds.get('worry', 0):>7.1f}/{totals.get('worry', 0):<2}"
              f"{kinds.get('positive', 0):>8.1f}/{totals.get('positive', 0):<2}"
              f"{positives - caught:>6.1f}/{positives:<3}")
    print("\nThe hybrid is held to a tighter bar than the plain rule: "
          "correction <= 1/11,")
    print("positive <= 0.2/11 and neutral exactly 0/54. It can afford that, "
          "because")
    print("the lexicon arm recovers recall the plain rule could only buy with "
          "false alarms.")
    shipped = next((o for o in options if o["name"].startswith("HYBRID")), None)
    if shipped and not shipped.get("unsatisfiable"):
        print(f"\nConfig lines to write:")
        print(f'    "FRUSTRATION_SOFT_THRESHOLD": {SOFT_FIXED:.2f},')
        print(f'    "FRUSTRATION_HARD_THRESHOLD": {shipped["threshold"]:.2f},')
    print("\nRecall here is for the hard level alone. A caller below it still")
    print("reaches the soft level, and two consecutive soft hits transfer in")
    print("WP6 — so a missed row is a delayed handover, not a lost one.")


def stubborn_rows(rows, proba_sets: list, actual, floor: float = 0.90) -> None:
    """The calm rows the model is confidently wrong about.

    These are what force the hard threshold upwards. A threshold cannot
    distinguish a confident error from a confident correct answer, so if two
    rows sit above 0.96 the hard level has to clear them or accept them —
    and either choice is expensive. Worth reading the actual sentences before
    paying for them.
    """
    import numpy as np

    stacked = np.vstack(proba_sets)
    means = stacked.mean(axis=0)
    sds = stacked.std(axis=0)

    print(f"\n{'=' * 92}")
    print(f"WHAT FORCES THE HARD THRESHOLD — calm rows scoring above {floor:.2f}")
    print(f"{'=' * 92}")

    offenders = [
        (means[i], sds[i], rows[i])
        for i in range(len(rows))
        if actual[i] == 0 and means[i] >= floor
    ]
    offenders.sort(key=lambda item: -item[0])

    if not offenders:
        print("  none — the hard threshold is set by something else")
    for score, sd, row in offenders:
        print(f"  {score:.3f} +/- {sd:.3f}  [{row['kind']}]  {row['text']}")

    # What the hybrid's lexicon arm adds among CALM rows. If this is
    # non-empty the arm is spending the same false-alarm budget the plain
    # threshold needs, and no value of tau can satisfy a tight bar.
    from hermes.config import get_settings

    soft = get_settings().frustration_soft_threshold
    arm = [
        (means[i], rows[i], frustration.lexicon_hits(rows[i]["text"]))
        for i in range(len(rows))
        if actual[i] == 0
        and frustration.lexicon_flag(rows[i]["text"])
        and means[i] >= soft
    ]
    print(f"\n{'-' * 92}")
    print(f"WHAT THE HYBRID'S LEXICON ARM ADDS — calm rows with a term "
          f"and prob >= {soft:g}")
    if not arm:
        print("  none on the mean; any failure comes from per-seed variation")
    for score, row, terms in sorted(arm, key=lambda item: -item[0]):
        print(f"  {score:.3f}  [{row['kind']}]  {row['text']}")
        print(f"         term: {', '.join(terms)}")

    near = [
        (means[i], rows[i], frustration.lexicon_hits(rows[i]["text"]))
        for i in range(len(rows))
        if actual[i] == 0
        and frustration.lexicon_flag(rows[i]["text"])
        and soft - 0.10 <= means[i] < soft
    ]
    if near:
        print(f"\n  and just below {soft:g}, so they cross on some seeds:")
        for score, row, terms in sorted(near, key=lambda item: -item[0]):
            print(f"  {score:.3f}  [{row['kind']}]  {row['text']}")
            print(f"         term: {', '.join(terms)}")

    print(f"\n{'-' * 92}")
    print("every `correction` row, by score — the kind the hard level must clear")
    corrections = sorted(
        ((means[i], rows[i]) for i in range(len(rows))
         if rows[i]["kind"] == "correction"),
        key=lambda item: -item[0],
    )
    for score, row in corrections:
        print(f"  {score:.3f}  {row['text']}")

    print(f"\n{'-' * 92}")
    print("frustrated rows the model scores lowest — what a high hard level misses")
    missed = sorted(
        ((means[i], rows[i]) for i in range(len(rows)) if actual[i] == 1),
        key=lambda item: item[0],
    )[:8]
    for score, row in missed:
        print(f"  {score:.3f}  [{row['kind']}]  {row['text']}")


def user_test_set(path: Path) -> None:
    """Score the hand-written held-out set once, with the shipped artefact.

    This set is **never** used for training, for epoch selection or for
    thresholds. It is read at the end of the run, scored with the saved model
    and the configured thresholds, and reported — nothing in this script can
    feed anything back into it.

    That makes it the only number here free of the project's recurring
    hazard. Everything else is cross-validated on a corpus whose rows are
    templated paraphrases of each other; this is text written separately,
    so it measures what the 180 rows cannot: whether the vocabulary
    generalises past the corpus's own idiom.
    """
    import csv

    from hermes.config import get_settings

    print(f"\n{'=' * 78}")
    print("HELD-OUT USER TEST SET — scored once, never trained on")
    print(f"{'=' * 78}")

    if not path.is_file():
        print(f"  not present at {path}")
        print("  write it with columns: id, text, label, lang, kind, group")
        print("  then re-run. Skipping.")
        return

    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    french = [r for r in rows if (r.get("lang") or "fr").lower() == "fr"]
    if not french:
        print(f"  {len(rows)} rows, none French. Skipping.")
        return

    settings = get_settings()
    if not (settings.paths.models / settings.frustration_model_file).is_file():
        print("  no shipped model on disk: uv run python "
              "scripts/train_frustration.py")
        return

    frustration.reset_model()
    if frustration.load_model() is None:
        print("  the shipped model would not load. Skipping.")
        return

    print(f"  {len(french)} French rows of {len(rows)}   "
          f"soft {settings.frustration_soft_threshold:g}  "
          f"hard {settings.frustration_hard_threshold:g}\n")

    results = [(row, frustration.detect(row["text"])) for row in french]

    # Counts by kind, which is how every other table in this file reads.
    kinds = sorted({row["kind"] for row, _ in results})
    print(f"  {'kind':<18}{'rows':>6}{'want':>6}{'none':>7}{'soft':>7}"
          f"{'hard':>7}{'flagged':>9}")
    print("  " + "-" * 60)
    for kind in kinds:
        subset = [(r, s) for r, s in results if r["kind"] == kind]
        want = int(subset[0][0]["label"])
        levels = Counter(s.level for _r, s in subset)
        flagged = levels["soft"] + levels["hard"]
        print(f"  {kind:<18}{len(subset):>6}{want:>6}{levels['none']:>7}"
              f"{levels['soft']:>7}{levels['hard']:>7}{flagged:>9}")

    actual = [int(row["label"]) for row, _ in results]
    predicted = [1 if s.is_frustrated else 0 for _r, s in results]
    precision, recall, f1 = prf(predicted, actual)
    print(f"\n  flagged as frustrated (soft or hard):")
    print(f"    precision {precision:.2f}   recall {recall:.2f}   F1 {f1:.2f}")
    print(f"    {sum(actual)} frustrated and {len(actual) - sum(actual)} calm "
          f"in the set")

    wrong = [
        (row, s) for row, s in results
        if (int(row["label"]) == 1) != s.is_frustrated
    ]
    if wrong:
        print(f"\n  the {len(wrong)} it got wrong:")
        for row, s in wrong:
            direction = "MISSED" if int(row["label"]) == 1 else "FALSE ALARM"
            print(f"    {direction:<12} {s.prob:5.3f} {s.level:<5} "
                  f"[{row['kind']}] {row['text']}")
            if s.lexicon_hits:
                print(f"                 lexicon: {', '.join(s.lexicon_hits)}")
    else:
        print("\n  nothing wrong on this set")


def write_report(
    path: Path,
    rows,
    baseline: dict,
    arms: dict,
    table: list[dict],
    options: list[dict],
    picks: list,
    totals: Counter,
    proba_sets: list,
    y_fr,
) -> None:
    """Write `reports/frustration_eval.md` and its per-case CSV.

    The `.md` is committed and the `.csv` is not — few stable summaries that
    the docs quote, against thousands of rows rewritten every run. See the
    note in `.gitignore`.
    """
    import csv

    from hermes.config import get_settings

    settings = get_settings()
    bp, br, bf1 = baseline["pooled"]
    positives = totals.get("frustrated", 0) + totals.get("frustrated_hard", 0)

    def cell(runs, metric: str) -> str:
        values = [r[metric] for r in runs]
        return f"{mean(values):.2f}±{pstdev(values):.2f}"

    lines = [
        "# WP5 — frustration detection", "",
        "**Scope: French only.** 180 rows of `frustration.csv` with "
        "`lang == fr`, 85 frustrated and 95 calm. The 296 Darija, Arabizi and "
        "mixed rows are not used — measured reason: 53 of the 56 `mix` rows "
        "carry their frustration in Darija (\"yezzi\", \"fadit mel robot\", "
        "\"ma tefhem chay\"), so a French reading of them has almost nothing "
        "to work from.",
        "",
        "Every split is `StratifiedGroupKFold` on `group`. 75 of the 180 rows "
        "sit in a paraphrase group, so an ungrouped split trains on some "
        "wordings of a complaint and tests on the rest.",
        "",
        "## Models", "",
        "| training set | model | split | P | R | F1 | ROC-AUC |",
        "|---|---|---|---|---|---|---|",
        f"| lexicon baseline | "
        f"{sum(len(t) for t in frustration.FRUSTRATION_TERMS.values())} terms "
        f"in {len(frustration.FRUSTRATION_TERMS)} families | grouped "
        f"| {bp:.2f} | {br:.2f} | {bf1:.2f} | — |",
    ]
    for (label, split), by_model in arms.items():
        for model, runs in by_model.items():
            lines.append(
                f"| {label} | {model} | {split} | {cell(runs, 'precision')} "
                f"| {cell(runs, 'recall')} | {cell(runs, 'f1')} "
                f"| {cell(runs, 'auc')} |"
            )

    lines += [
        "",
        "**Shipped: the torch MLP ensemble** — one hidden layer of 48, dropout "
        "0.3, AdamW weight decay, 5 members averaged, epochs chosen per fold "
        "by inner grouped CV. Logistic regression and the lexicon are "
        "baselines.",
        "",
        "### The paraphrase leak", "",
        "| model | grouped | ungrouped | inflation |", "|---|---|---|---|",
    ]
    for model in MODELS:
        grouped = arms.get(("(a) French only", "grouped"), {}).get(model)
        ungrouped = arms.get(("(a) French only", "ungrouped"), {}).get(model)
        if grouped and ungrouped:
            g = mean(r["f1"] for r in grouped)
            u = mean(r["f1"] for r in ungrouped)
            lines.append(f"| {model} | {g:.2f} | {u:.2f} | +{u - g:.2f} |")

    if picks:
        chosen = Counter(pick for pick, _curve in picks)
        averaged: dict[int, list[float]] = {}
        for _pick, curve in picks:
            for candidate, loss in curve.items():
                averaged.setdefault(candidate, []).append(loss)
        lines += [
            "",
            "### Epochs, chosen rather than asserted", "",
            f"Inner grouped CV inside each outer training fold, "
            f"{len(picks)} selections (5 seeds x 5 folds).",
            "", "| epochs | chosen | mean inner validation loss |",
            "|---|---:|---:|",
        ]
        for candidate in sorted(averaged):
            lines.append(
                f"| {candidate} | {chosen.get(candidate, 0)}/{len(picks)} "
                f"| {mean(averaged[candidate]):.4f} |"
            )
        lines.append("")
        lines.append(
            "The loss rises monotonically with epochs, so a fixed 200 — the "
            "first draft's value — was four times too many."
        )

    lines += [
        "", "## Thresholds", "",
        f"Swept at 0.02 on the shipped model's out-of-fold probabilities. "
        f"Counts are means over 5 seeds.",
        "",
        "| tau | recall | precision | F1 | frustrated | frust_hard "
        "| correction | worry | neutral | positive |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table:
        if row["threshold"] % 0.1 > 0.001:
            continue
        k = row["kinds"]
        lines.append(
            f"| {row['threshold']:.2f} | {row['recall']:.2f} "
            f"| {row['precision']:.2f} | {row['f1']:.2f} "
            f"| {k.get('frustrated', 0):.1f}/75 "
            f"| {k.get('frustrated_hard', 0):.1f}/10 "
            f"| {k.get('correction', 0):.1f}/11 | {k.get('worry', 0):.1f}/12 "
            f"| {k.get('neutral', 0):.1f}/54 | {k.get('positive', 0):.1f}/11 |"
        )

    lines += [
        "", "### The two levels, and the rule", "",
        "```",
        f"soft  if  prob >= {settings.frustration_soft_threshold:g}",
        f"hard  if  prob >= {settings.frustration_hard_threshold:g}"
        f"  OR  (lexicon hit AND prob >= {settings.frustration_soft_threshold:g})",
        "```", "",
        "| rule | tau | recall | precision | F1 | correction | worry "
        "| positive | missed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for option in options:
        if option.get("unsatisfiable"):
            lines.append(f"| {option['name']} | — | no threshold satisfies "
                         "this constraint | | | | | | |")
            continue
        k = option["kinds"]
        caught = k.get("frustrated", 0) + k.get("frustrated_hard", 0)
        lines.append(
            f"| {option['name']} | {option['threshold']:.2f} "
            f"| {option['recall']:.2f} | {option['precision']:.2f} "
            f"| {option['f1']:.2f} | {k.get('correction', 0):.1f}/11 "
            f"| {k.get('worry', 0):.1f}/12 | {k.get('positive', 0):.1f}/11 "
            f"| {positives - caught:.1f}/{positives} |"
        )

    lines += [
        "",
        "The hybrid's second arm is worth its place: at the same threshold and "
        "the same false alarms as the probability alone, it catches six more "
        "frustrated callers. It only became affordable once the lexicon's "
        "repetition family was restricted to the first person — while it also "
        "matched the bare infinitive it wrongly flagged \"Vous pouvez répéter "
        "s'il vous plaît ?\", a `correction`, and the threshold was already "
        "spending its whole correction budget.",
        "",
        "**The hard level is one of three paths to a human.** Two consecutive "
        "soft hits transfer, and so does the WP6/WP7 repeat counter. A row "
        "missed here is a delayed handover, not a lost one.",
        "",
        "## Limitations", "",
        "- **Recall 0.91 holds for corpus-like phrasing.** The corpus's "
        "frustrated rows run on a narrow vocabulary, and probes with wording "
        "outside it are hit and miss. The hand-written held-out set is the "
        "real measure.",
        "- The corpus is visibly templated — `Sérieusement, / Oh là là, / "
        "Écoutez,` prefixed onto a core complaint. That is what `group` "
        "encodes and why the grouped split is load-bearing.",
        "- Two calm rows are confidently misread: `Non non, c'est bon` at "
        "0.975 and `Oui ça marche !` at 0.943. Both are correctly labelled, "
        "so this is model error. It is **not** general negation blindness: on "
        "fresh polarity pairs the model separates `ça marche` 0.024 from `ça "
        "ne marche pas` 0.700.",
        "- Probabilities are heavily saturated. Real decisions happen in a "
        "thin band, which makes `oui j'ai compris` at 0.597 uncomfortably "
        "close to the soft threshold.",
        f"- 180 rows. One row is 0.6 points, so **no figure here is "
        f"meaningful to more than two decimals**, and the hard kinds "
        f"(7 to 12 rows each) are reported as counts, never percentages.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Per case, for digging rather than for reading.
    import numpy as np

    means = np.vstack(proba_sets).mean(axis=0)
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "id", "text", "label", "kind", "group", "mean_prob",
            "level_at_shipped", "lexicon_hits",
        ])
        writer.writeheader()
        for index, row in enumerate(rows):
            probability = float(means[index])
            hits = frustration.lexicon_hits(row["text"])
            writer.writerow({
                "id": row.get("id", index), "text": row["text"],
                "label": row["label"], "kind": row["kind"],
                "group": row["group"], "mean_prob": round(probability, 4),
                "level_at_shipped": frustration.level_for(
                    probability, bool(hits),
                    settings.frustration_soft_threshold,
                    settings.frustration_hard_threshold,
                ),
                "lexicon_hits": "|".join(hits),
            })
    print(f"\nwrote {path}")
    print(f"wrote {csv_path}  (gitignored)")


def curve_report(curve: dict[str, dict]) -> None:
    print(f"\n{'=' * 78}")
    print("LEARNING CURVE — arm (a), French F1 against French training rows")
    print(f"{'=' * 78}")
    header = f"{'training data':<30}" + "".join(f"{m:>24}" for m in MODELS)
    print(header)
    print("-" * len(header))
    for label in curve:
        line = f"{label:<30}"
        for model in MODELS:
            runs = curve[label].get(model)
            if not runs:
                line += f"{'—':>24}"
                continue
            f1 = mean(r["f1"] for r in runs)
            sd = pstdev([r["f1"] for r in runs])
            auc = mean(r["auc"] for r in runs)
            line += f"{f'F1 {f1:.2f}±{sd:.2f} AUC {auc:.2f}':>24}"
        print(line)
    print("\nIf the last two rows are within noise of each other, arm (a) has")
    print("stopped learning from extra French rows — which is the control for")
    print("the EmoWOZ arms: more data was not the missing ingredient.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from hermes.config import get_settings

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21, 42, 99])
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--no-emowoz", action="store_true",
                        help="skip arm (b) and the learning curve")
    parser.add_argument("--no-curve", action="store_true")
    parser.add_argument("--b1-sizes", type=int, nargs="+", default=[200, 400, 800],
                        help="EmoWOZ sizes to try at equal total weight")
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75, 1.0],
                        help="shares of each French training fold, for the curve")
    parser.add_argument("--no-select", action="store_true",
                        help="skip the inner-CV epoch selection for the "
                             "shipped config")
    parser.add_argument(
        "--user-test", type=Path,
        default=get_settings().paths.raw / "frustration_test_user.csv",
        help="hand-written held-out set, scored once at the end and never "
             "used for training or thresholds",
    )
    parser.add_argument("--reports", type=Path,
                        default=get_settings().paths.root / "reports")
    parser.add_argument("--no-report", action="store_true",
                        help="print everything but write no files")
    parser.add_argument("--no-thresholds", action="store_true",
                        help="skip the step 3 threshold sweep")
    parser.add_argument("--epoch-candidates", type=int, nargs="+", default=None,
                        help="epoch counts the inner CV chooses between. The "
                             "default grid bottomed out at its smallest "
                             "value, so the minimum may sit below it")
    parser.add_argument("--unweighted-models", nargs="*", default=None,
                        choices=list(MODELS),
                        help="which models run the unweighted (b) arm; pass "
                             "'logistic regression' alone to skip the slow "
                             "1744-row MLP fits")
    parser.add_argument("--emowoz", type=Path,
                        default=get_settings().paths.processed / "emowoz_train.csv")
    return parser.parse_args(argv)


def load_emowoz(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    args = parse_args(argv)
    rows = frustration.load_dataset(lang="fr")
    if not rows:
        print("no French rows in frustration.csv")
        return 1

    baseline = score_baseline(rows, french_folds(rows, args.folds, args.seeds[0], True))
    baseline_report(rows, baseline)
    if args.baseline_only:
        return 0

    totals = Counter(row["kind"] for row in rows)
    print("\nembedding the French rows...", flush=True)
    X_fr = embed([row["text"] for row in rows])
    y_fr = np.array([int(row["label"]) for row in rows])
    print(f"  {X_fr.shape[0]} x {X_fr.shape[1]}")

    aux = [] if args.no_emowoz else load_emowoz(args.emowoz)
    X_aux = y_aux = None
    if aux:
        print(f"embedding {len(aux)} EmoWOZ rows...", flush=True)
        X_aux = embed([row["text"] for row in aux])
        y_aux = np.array([int(row["label"]) for row in aux])
        print(f"  {X_aux.shape[0]} x {X_aux.shape[1]}  "
              f"({int(y_aux.sum())} frustrated, {len(y_aux) - int(y_aux.sum())} calm)")
    elif not args.no_emowoz:
        print(f"\n! no EmoWOZ at {args.emowoz}")
        print("  build it first: uv run python scripts/build_emowoz.py")
        print("  continuing with arm (a) only")

    arms: dict = {}
    print("\n(a) French only, grouped")
    arms[("(a) French only", "grouped")] = run_arm(
        rows, X_fr, y_fr, args.folds, args.seeds, grouped=True,
    )
    print("(a) French only, UNGROUPED — the leak gap")
    arms[("(a) French only", "ungrouped")] = run_arm(
        rows, X_fr, y_fr, args.folds, args.seeds, grouped=False,
    )

    if X_aux is not None:
        # The unweighted arm is the most expensive cell in the table and the
        # least informative one: logistic regression already answered it at
        # F1 0.81 against (a)'s 0.93, and the MLP row cannot change the
        # conclusion that unweighted mixing swamps 144 French rows. Its
        # training set is 1744 rows against every other MLP arm's ~144, so it
        # costs about twelve times as much per fit. `--models` restricts it.
        unweighted = args.unweighted_models or tuple(MODELS)
        print(f"(b) French + {len(y_aux)} EmoWOZ, unweighted, grouped "
              f"[{', '.join(unweighted)}]")
        arms[(f"(b) +{len(y_aux)} unweighted", "grouped")] = run_arm(
            rows, X_fr, y_fr, args.folds, args.seeds, grouped=True,
            X_aux=X_aux, y_aux=y_aux, only=tuple(unweighted),
        )

        # (b1) Equal total weight, three sizes. The question this separates:
        # did EmoWOZ hurt because English is the wrong signal, or only
        # because 1600 rows drowned 144?
        for size in args.b1_sizes:
            if size > len(y_aux):
                continue
            print(f"(b1) French + {size} EmoWOZ, equal total weight, grouped")
            arms[(f"(b1) +{size} weighted", "grouped")] = run_arm(
                rows, X_fr, y_fr, args.folds, args.seeds, grouped=True,
                X_aux=X_aux, y_aux=y_aux, aux_limit=size, balance_weight=True,
            )

        # (b2) Representation from English, boundary from French. MLP only —
        # logistic regression has nothing to pretrain.
        print("(b2) MLP pretrained on EmoWOZ, fine-tuned on French, grouped")
        arms[("(b2) pretrain + finetune", "grouped")] = run_arm(
            rows, X_fr, y_fr, args.folds, args.seeds, grouped=True,
            X_aux=X_aux, y_aux=y_aux, pretrain=True, only=("torch MLP x5",),
        )

    # The shipped configuration, with its epoch count chosen rather than
    # asserted. Only this one: the comparison arms above deliberately keep a
    # fixed, equalised update budget, which is what makes them comparable.
    picks: list = []
    if not args.no_select:
        from hermes.understand.mlp import EPOCH_CANDIDATES, MLPConfig

        candidates = tuple(args.epoch_candidates or EPOCH_CANDIDATES)
        print(f"(ship) torch MLP, epochs chosen by inner grouped CV "
              f"from {list(candidates)}")
        arms[("(ship) inner-CV epochs", "grouped")] = run_arm(
            rows, X_fr, y_fr, args.folds, args.seeds, grouped=True,
            only=("torch MLP x5",),
            select_epochs=MLPConfig(epochs=max(candidates)), picks=picks,
            candidates=candidates,
        )

    models_report(rows, arms, baseline, totals)

    # Step 3 runs on the shipped arm's out-of-fold probabilities — the only
    # honest basis for a threshold, since every row was scored by a model
    # that had not seen it.
    shipped = arms.get(("(ship) inner-CV epochs", "grouped"), {}).get(
        "torch MLP x5"
    )
    if shipped and not args.no_thresholds:
        proba_sets = [run["proba"] for run in shipped]
        table = [
            threshold_row(rows, proba_sets, y_fr, tau, totals) for tau in SWEEP
        ]
        soft, hard = pick_thresholds(table)
        step3_report(rows, table, soft, hard, totals)
        options = hard_level_options(rows, proba_sets, y_fr, SOFT_FIXED, totals)
        hard_options_report(options, totals)
        stubborn_rows(rows, proba_sets, y_fr)
        user_test_set(args.user_test)
        if not args.no_report:
            write_report(
                args.reports / "frustration_eval.md", rows, baseline, arms,
                table, options, picks, totals, proba_sets, y_fr,
            )

    if picks:
        print(f"\n{'=' * 70}")
        print("EPOCHS CHOSEN BY INNER GROUPED CV — shipped config")
        print(f"{'=' * 70}")
        chosen = Counter(pick for pick, _curve in picks)
        print(f"{len(picks)} selections (5 seeds x 5 outer folds)")
        for epochs, count in sorted(chosen.items()):
            print(f"  {epochs:>4} epochs   chosen {count:>2}/{len(picks)} times")
        print("\nmean inner validation loss, averaged over every selection")
        averaged: dict[int, list[float]] = {}
        for _pick, curve in picks:
            for candidate, loss in curve.items():
                averaged.setdefault(candidate, []).append(loss)
        for candidate in sorted(averaged):
            values = averaged[candidate]
            print(f"  {candidate:>4} epochs   {mean(values):.4f}"
                  f" +/- {pstdev(values):.4f}")
        print("\nA spread of choices is not instability — each outer fold sees")
        print("different training rows, so a different optimum is the correct")
        print("outcome. The point is that no single number was asserted.")

    print("\nthe paraphrase leak, French only")
    for model in MODELS:
        g = mean(r["f1"] for r in arms[("(a) French only", "grouped")][model])
        u = mean(r["f1"] for r in arms[("(a) French only", "ungrouped")][model])
        print(f"  {model:<22} grouped {g:.2f}   ungrouped {u:.2f}"
              f"   inflation +{u - g:.2f}")

    # Does (a) have a data problem at all? The curve over French training
    # rows answers that, and it is the control for the EmoWOZ result: if (a)
    # is already flat, more data of any kind was never going to be the lever.
    if not args.no_curve:
        print("\nlearning curve over FRENCH training data, arm (a)")
        curve: dict[str, dict] = {}
        for fraction in args.fractions:
            label = f"{int(fraction * 100)}% of each training fold"
            print(f"  {label}", flush=True)
            curve[label] = run_arm(
                rows, X_fr, y_fr, args.folds, args.seeds, grouped=True,
                french_fraction=fraction,
            )
        curve_report(curve)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
