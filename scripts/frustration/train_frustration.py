"""Fit the shipped frustration ensemble on every French row and save it.

This is the artefact the agent loads. It differs from `eval_frustration.py` in
one way that matters: there is no held-out fold here, because the final model
should see all the data. The *performance* of this configuration was measured
in `eval_frustration.py` on grouped out-of-fold predictions, and those are the
only numbers worth quoting — a model scored on rows it trained on would read
near-perfect and mean nothing.

The epoch count is chosen the same way it was during evaluation: inner grouped
cross-validation over `group`, on averaged validation loss. Here the inner
split runs over the whole 180 rows rather than inside an outer fold.

    uv run python scripts/frustration/train_frustration.py
    uv run python scripts/frustration/train_frustration.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hermes.understand import frustration  # noqa: E402
from hermes.config import get_settings  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path,
        default=settings.paths.models / settings.frustration_model_file,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true",
                        help="choose the epochs and report, write nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    from hermes.infra import embeddings
    from hermes.understand.mlp import EPOCH_CANDIDATES, MLPConfig, TorchMLP, choose_epochs

    args = parse_args(argv)
    settings = get_settings()

    rows = frustration.load_dataset(lang="fr")
    if not rows:
        print("no French rows in frustration.csv")
        return 1

    labels = np.array([int(row["label"]) for row in rows])
    groups = [row["group"] for row in rows]
    print(f"{len(rows)} French rows: {int(labels.sum())} frustrated, "
          f"{len(labels) - int(labels.sum())} calm")
    print(f"{len(set(groups))} paraphrase groups")

    print(f"\nembedding with {settings.embedding_model}...", flush=True)
    X = embeddings.encode([row["text"] for row in rows])
    print(f"  {X.shape[0]} x {X.shape[1]}")

    print(f"\nchoosing epochs by inner grouped CV from "
          f"{list(EPOCH_CANDIDATES)}...", flush=True)
    config = MLPConfig()
    epochs, curve = choose_epochs(X, labels, groups, config=config, seed=args.seed)
    for candidate in sorted(curve):
        mark = "  <- chosen" if candidate == epochs else ""
        print(f"  {candidate:>4} epochs   loss {curve[candidate]:.4f}{mark}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return 0

    from dataclasses import replace

    final = replace(config, epochs=epochs)
    print(f"\nfitting {final.members} members at {epochs} epochs...", flush=True)
    model = TorchMLP(final, seed=args.seed).fit(X, labels)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out, metadata={
        # The embedder is the field that matters: the weights are 384 wide
        # whatever encoded them, so a mismatch loads cleanly and scores
        # nonsense. `frustration.load_model` refuses on this.
        "embedder": settings.embedding_model,
        "embedding_dim": int(X.shape[1]),
        "epochs": epochs,
        "epoch_candidates": list(EPOCH_CANDIDATES),
        "inner_cv_loss": {str(k): round(v, 5) for k, v in curve.items()},
        "rows": len(rows),
        "lang": "fr",
        "soft_threshold": settings.frustration_soft_threshold,
        "hard_threshold": settings.frustration_hard_threshold,
        "trained": date.today().isoformat(),
        # Quoted from the grouped out-of-fold evaluation, not from this fit.
        "measured_on_held_out": {
            "f1": 0.93, "auc": 0.99, "precision": 0.93, "recall": 0.92,
            "note": "5 seeds x 5 grouped folds, scripts/frustration/eval_frustration.py",
        },
    })
    size = args.out.stat().st_size / 1024
    print(f"written: {args.out}  ({size:.0f} KB)")

    # Prove it loads back and scores, so a broken artefact is caught here
    # rather than on the first call.
    frustration.reset_model()
    check = frustration.score("ça fait trois fois que je répète !")
    print(f"\nreload check: level={check.level} prob={check.prob:.3f} "
          f"source={check.source} terms={check.matched_terms}")
    if check.source != "model":
        print("  ! the saved model did not load back")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
