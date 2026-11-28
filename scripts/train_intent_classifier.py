"""Train the intent classifier, and report how it compares with the alternatives.

Fits a class-balanced logistic regression on the sentence embeddings of the
labelled examples and writes it to `models/`. Also cross-validates kNN and an
MLP on the identical embeddings, which is the comparison the evaluation plan
asks for — and the comparison is a result in itself, not just a formality.

Measured on the 209 French examples:

    kNN k=5                68.9% accuracy, 0.650 macro F1, 79.5% medical recall
    kNN k=1                72.7%           0.703           74.4%
    logistic regression    78.0%           0.759           87.2%
    logistic regression +  78.9%           0.793           92.3%   <- shipped
      class_weight=balanced
    MLP 256                80.4%           0.785           84.6%

The MLP has the best raw accuracy and is still the wrong choice: it loses on
macro F1 and, more importantly, on medical recall. Accuracy rewards getting the
big class right, and `info` is 42% of the French corpus. Balanced logistic
regression is what ships.

    python scripts/train_intent_classifier.py
    python scripts/train_intent_classifier.py --lang all --compare
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hermes.infra import embeddings  # noqa: E402
from hermes.config import get_settings  # noqa: E402

#: The label whose recall matters more than the average.
SAFETY_LABEL = "medical_question"


def load_rows(path: Path, lang: str | None) -> list[dict[str, str]]:
    """Read the labelled examples, optionally for one language."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = [
            r for r in csv.DictReader(handle)
            if (r.get("text") or "").strip() and (r.get("intent") or "").strip()
        ]
    return [r for r in rows if lang is None or r.get("lang") == lang]


def compare(X: np.ndarray, y: np.ndarray, seed: int = 7) -> None:
    """Cross-validate every candidate and print the table."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    candidates = {
        "kNN k=5 (cosine)": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "kNN k=1 (cosine)": KNeighborsClassifier(n_neighbors=1, metric="cosine"),
        "logistic regression": LogisticRegression(max_iter=3000, C=10.0),
        "logistic regression, balanced": build_model(),
        "MLP 256": MLPClassifier(hidden_layer_sizes=(256,), max_iter=1200,
                                 random_state=seed),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    header = (f"{'method':<32}{'accuracy':>10}{'macro F1':>10}"
              f"{'medical recall':>16}")
    print(f"\n{header}")
    print("-" * len(header))
    for name, model in candidates.items():
        predicted = cross_val_predict(model, X, y, cv=cv)
        safety = y == SAFETY_LABEL
        recall = (
            (predicted[safety] == SAFETY_LABEL).mean() if safety.any() else float("nan")
        )
        print(
            f"{name:<32}{accuracy_score(y, predicted):>10.1%}"
            f"{f1_score(y, predicted, average='macro', zero_division=0):>10.3f}"
            f"{recall:>16.1%}"
        )
    print("-" * len(header))


def build_model():
    """Return the classifier that ships.

    ``class_weight='balanced'`` is not cosmetic: `info` is 87 of the 209 French
    examples, and without the weighting the model buys accuracy by guessing it,
    which costs recall on `medical_question` — the one class where a miss is a
    safety failure rather than an annoyance.
    """
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the corpus, the language, and whether to run the comparison."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--intents", type=Path,
                        default=settings.paths.raw / "intents.csv")
    parser.add_argument("--lang", default="fr",
                        help="language to train on; 'all' for every row")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the model; default: models/")
    parser.add_argument("--compare", action="store_true",
                        help="cross-validate the alternatives too")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Fit the classifier, report it, and save it."""
    args = parse_args(argv)
    settings = get_settings()
    import joblib
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    lang = None if args.lang == "all" else args.lang
    rows = load_rows(args.intents, lang)
    if not rows:
        print(f"no examples for lang={args.lang}", file=sys.stderr)
        return 1

    print(f"{len(rows)} examples, lang={args.lang}")
    for label in sorted({r["intent"] for r in rows}):
        print(f"  {label:<24}{sum(1 for r in rows if r['intent'] == label):>5}")

    X = embeddings.encode([r["text"] for r in rows])
    y = np.array([r["intent"] for r in rows])

    if args.compare:
        compare(X, y)

    model = build_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    predicted = cross_val_predict(model, X, y, cv=cv)

    print("\ncross-validated report for the model that ships")
    print(classification_report(y, predicted, zero_division=0))

    safety = y == SAFETY_LABEL
    if safety.any():
        leaked = [
            (r["text"], p) for r, p, m in zip(rows, predicted, safety)
            if m and p != SAFETY_LABEL
        ]
        print(f"{SAFETY_LABEL} leaks: {len(leaked)} of {int(safety.sum())} "
              f"(recall {(predicted[safety] == SAFETY_LABEL).mean():.1%})")
        for text, got in leaked:
            print(f"  -> {got:<22} {text[:62]}")
        print("\nThe deterministic rules in escalation.py run before this and "
              "catch what it misses; neither layer is trusted alone.")

    model.fit(X, y)
    target = args.out or (settings.paths.models / settings.intent_model_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "classes": list(model.classes_),
            "embedding_model": settings.embedding_model,
            "embedding_dim": int(X.shape[1]),
            "lang": args.lang,
            "n_examples": len(rows),
        },
        target,
    )
    print(f"\nwritten: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
