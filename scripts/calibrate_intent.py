"""Calibrate the intent gate's threshold and confidence floor against measured data.

Two numbers come out, and they control different branches:

- **τ (`INTENT_ACCEPT_THRESHOLD`)** — the similarity below which the caller has
  said something with no precedent in the corpus. Controls `below_threshold`.
- **the confidence floor (`INTENT_MIN_PROBABILITY`)** — below which the
  classifier is guessing between classes and the agent should ask rather than
  act. Controls `clarify`.

Both are measured against what the gate actually *does*, not against a proxy.
An out-of-scope sentence classified `off_topic` is declined, and a decline is
a correct outcome — counting it as a failure would push τ higher than it needs
to be and start rejecting real callers to fix something that was not broken.

**Honesty of the two sides.** `oos_validation.csv` is held out: never indexed,
never trained on, so the real gate can be run against it directly. The in-scope
examples are not: every one is in the index and would retrieve *itself* at
similarity 1.0, and the classifier was fitted on them. So their similarity is
recomputed leave-one-out and their probabilities cross-validated. Mixing the
two would make any τ look free.

    python scripts/calibrate_intent.py
    python scripts/calibrate_intent.py --lang fr --top-k 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hermes.infra import embeddings, vector_store  # noqa: E402
from hermes.understand import intent_gate  # noqa: E402
from hermes.config import get_settings  # noqa: E402
from hermes.understand.intent_gate import Action, Intent, Neighbour, score_intents  # noqa: E402

#: An out-of-scope caller is handled correctly if the agent hands over, asks
#: again, asks which of two things they meant, or declines. It fails only when
#: the agent routes the turn as though it had understood.
SAFE = {Action.TRANSFER, Action.REPHRASE, Action.CLARIFY, Action.DECLINE}


def read_csv(path: Path, lang: str | None) -> list[dict[str, str]]:
    """Read a corpus, optionally keeping one language."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if (r.get("text") or "").strip()]
    return [r for r in rows if lang is None or r.get("lang") == lang]


def leave_one_out_similarity(
    rows: list[dict[str, str]],
    vectors: np.ndarray,
    collection: str,
    top_k: int,
    lang: str | None,
) -> list[float]:
    """Best similarity for each example, with its own vector removed."""
    out: list[float] = []
    for row, vector in zip(rows, vectors):
        hits = vector_store.search(
            collection, vector, limit=top_k + 1,
            where={"lang": lang} if lang else None,
        )
        neighbours = [
            Neighbour(
                text=str(h.get("text", "")),
                intent=Intent.from_label(str(h.get("intent", ""))),
                lang=str(h.get("lang", "")),
                score=h.score,
            )
            for h in hits
            if str(h.get("text", "")).strip() != row["text"].strip()
        ][:top_k]
        scores = score_intents(neighbours)
        out.append(max(scores.values()) if scores else 0.0)
    return out


def cross_validated_probabilities(
    rows: list[dict[str, str]],
    vectors: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Out-of-fold probabilities, so the classifier is not graded on its own fit."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_intent_classifier import build_model  # noqa: E402

    labels = np.array([r["intent"] for r in rows])
    model = build_model()
    probabilities = cross_val_predict(
        model, vectors, labels,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=7),
        method="predict_proba",
    )
    model.fit(vectors, labels)
    return probabilities, [str(c) for c in model.classes_]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the corpora, the language and the neighbourhood size."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--intents", type=Path,
                        default=settings.paths.raw / "intents.csv")
    parser.add_argument("--oos", type=Path,
                        default=settings.paths.raw / "oos_validation.csv")
    parser.add_argument("--lang", default="fr",
                        help="language to calibrate on; 'all' for every row")
    parser.add_argument("--top-k", type=int, default=settings.intent_top_k)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Sweep τ and the confidence floor, and report what each would cost."""
    args = parse_args(argv)
    settings = get_settings()
    lang = None if args.lang == "all" else args.lang

    vector_store.require_server()

    in_scope = read_csv(args.intents, lang)
    out_scope = read_csv(args.oos, lang)

    guarded = [r for r in out_scope if intent_gate.is_too_short(r["text"])]
    out_scope = [r for r in out_scope if not intent_gate.is_too_short(r["text"])]

    print(f"{len(in_scope)} in-scope examples (leave-one-out, cross-validated)")
    print(f"{len(out_scope)} out-of-scope sentences, held out entirely")
    if guarded:
        print(f"{len(guarded)} rejected by the length guard before τ is consulted")
    print()

    in_vectors = embeddings.encode([r["text"] for r in in_scope])
    similarity = leave_one_out_similarity(
        in_scope, in_vectors, settings.intent_collection, args.top_k, lang
    )
    probabilities, classes = cross_validated_probabilities(in_scope, in_vectors)
    medical_column = classes.index("medical_question")
    top_probability = probabilities.max(axis=1)
    predicted = [classes[i] for i in probabilities.argmax(axis=1)]

    # ---------------------------------------------------------------- tau
    print("=== τ: the similarity floor ===")
    print("Out-of-scope runs the real gate. In-scope uses leave-one-out "
          "similarity,\nso τ can actually reject an example.\n")
    header = (f"{'τ':>6}{'oos safe':>11}{'in-scope kept':>15}"
              f"{'wrongly rejected':>18}")
    print(header)
    print("-" * len(header))

    best_tau = None
    for tau in np.arange(0.40, 0.86, 0.025):
        safe = sum(
            1 for r in out_scope
            if intent_gate.classify(r["text"], threshold=float(tau)).action in SAFE
        )
        rejected = sum(1 for s in similarity if s < tau)
        oos_safe = safe / len(out_scope)
        kept = 1 - rejected / len(in_scope)
        print(f"{tau:>6.3f}{oos_safe:>11.0%}{kept:>15.0%}{rejected:>18}")
        # Reject out-of-scope callers without turning real ones away.
        quality = oos_safe * kept
        if best_tau is None or quality > best_tau[0]:
            best_tau = (quality, float(tau), oos_safe, kept, rejected)
    print("-" * len(header))
    _, tau, oos_safe, kept, rejected = best_tau
    print(f"\nbest balance: τ = {tau:.3f}"
          f"   out-of-scope safe {oos_safe:.0%}"
          f"   in-scope kept {kept:.0%} ({rejected} rejected)")

    # -------------------------------------------------- confidence floor
    print("\n\n=== the confidence floor: when to ask instead of act ===")
    print("A clarify costs the caller one turn. A misroute costs them the "
          "call.\n")
    header = (f"{'floor':>7}{'in-scope asked':>16}{'misroutes caught':>19}"
              f"{'right routes interrupted':>26}")
    print(header)
    print("-" * len(header))

    wrong = [i for i, r in enumerate(in_scope) if predicted[i] != r["intent"]]
    right = [i for i, r in enumerate(in_scope) if predicted[i] == r["intent"]]

    best_floor = None
    for floor in np.arange(0.30, 0.86, 0.05):
        asked = [i for i in range(len(in_scope)) if top_probability[i] < floor]
        caught = len([i for i in asked if i in set(wrong)])
        interrupted = len([i for i in asked if i in set(right)])
        print(
            f"{floor:>7.2f}{len(asked) / len(in_scope):>16.0%}"
            f"{caught / max(len(wrong), 1):>19.0%}"
            f"{interrupted / max(len(right), 1):>26.0%}"
        )
        # Catch misroutes without interrupting correct ones.
        quality = (caught / max(len(wrong), 1)) - (
            interrupted / max(len(right), 1)
        )
        if best_floor is None or quality > best_floor[0]:
            best_floor = (quality, float(floor), caught, interrupted)
    print("-" * len(header))
    _, floor, caught, interrupted = best_floor
    print(f"\nbest balance: floor = {floor:.2f}"
          f"   catches {caught}/{len(wrong)} misroutes"
          f"   interrupts {interrupted}/{len(right)} correct routes")

    # ------------------------------------------------------------ safety
    print("\n\n=== safety check at the proposed settings ===")
    leaked = [
        r for r in in_scope
        if r["intent"] == "medical_question"
        and intent_gate.classify(
            r["text"], threshold=tau
        ).action is not Action.TRANSFER
    ]
    medical_total = sum(1 for r in in_scope if r["intent"] == "medical_question")
    print(f"medical questions transferred: "
          f"{medical_total - len(leaked)}/{medical_total}")
    for r in leaked:
        print(f"  LEAK  {r['text'][:66]}")

    floor_recall = (
        probabilities[[r["intent"] == "medical_question" for r in in_scope],
                      medical_column]
        >= settings.medical_probability_floor
    ).mean()
    print(f"P(medical) ≥ {settings.medical_probability_floor} catches "
          f"{floor_recall:.0%} of medical questions out of fold")

    print("\n\nput these in .env:")
    print(f"  INTENT_ACCEPT_THRESHOLD={tau:.3f}")
    print(f"  INTENT_MIN_PROBABILITY={floor:.2f}")
    print(f"\ncurrently: {settings.intent_accept_threshold} / "
          f"{settings.intent_min_probability}")

    reasons = Counter(
        intent_gate.classify(r["text"], threshold=tau).reason for r in out_scope
    )
    print("\nwhat the out-of-scope set does at that τ:")
    for reason, n in reasons.most_common():
        print(f"  {reason:<34}{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
