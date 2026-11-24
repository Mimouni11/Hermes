"""Calibrate the intent gate's threshold and margin against measured data.

Two questions, one sweep:

- **τ (`INTENT_ACCEPT_THRESHOLD`)** — below what similarity is a caller saying
  something the agent has no example of? Measured against
  `oos_validation.csv`, which is 60 sentences that belong to no intent.
- **the margin (`INTENT_MARGIN`)** — how close may the top two intents be
  before the agent should ask rather than guess?

In-scope scores are measured **leave-one-out**. Every labelled example is in
the index, so searching with one returns itself at similarity 1.0; scoring
against that would say the gate is perfect and tell you nothing. The example's
own vector is dropped from its own result.

The plan asks for out-of-scope recall as high as possible, but not at any
price: a τ that refuses every caller has perfect recall and no use. The sweep
reports both sides so the trade is visible.

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

from hermes import embeddings, vector_store  # noqa: E402
from hermes.config import get_settings  # noqa: E402
from hermes.intent_gate import Intent, score_intents, Neighbour  # noqa: E402


def read_csv(path: Path, lang: str | None) -> list[dict[str, str]]:
    """Read a corpus, optionally keeping one language."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if (r.get("text") or "").strip()]
    if lang:
        rows = [r for r in rows if r.get("lang") == lang]
    return rows


def ranked_for(
    text: str,
    vector: np.ndarray,
    collection: str,
    top_k: int,
    lang: str | None,
    exclude_self: bool,
) -> list[tuple[Intent, float]]:
    """Return (intent, best score) pairs for one query, best first."""
    hits = vector_store.search(
        collection,
        vector,
        # One extra, so dropping the self-match still leaves top_k.
        limit=top_k + (1 if exclude_self else 0),
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
    ]
    if exclude_self:
        neighbours = [n for n in neighbours if n.text.strip() != text.strip()][:top_k]
    scores = score_intents(neighbours)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the corpora, the language and the neighbourhood size."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--intents", type=Path,
                        default=settings.paths.raw / "intents.csv")
    parser.add_argument("--oos", type=Path,
                        default=settings.paths.raw / "oos_validation.csv")
    parser.add_argument("--lang", default="fr",
                        help="which language to calibrate on; 'all' for every row")
    parser.add_argument("--top-k", type=int, default=settings.intent_top_k)
    parser.add_argument("--keep-short", action="store_true",
                        help="include utterances the length guard already "
                             "rejects; they are excluded by default because "
                             "they never reach the embedder")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Sweep τ and the margin, and report what each would cost."""
    args = parse_args(argv)
    settings = get_settings()
    lang = None if args.lang == "all" else args.lang

    vector_store.require_server()

    in_scope = read_csv(args.intents, lang)
    out_scope = read_csv(args.oos, lang)
    if not in_scope or not out_scope:
        print("one of the corpora is empty for this language", file=sys.stderr)
        return 1

    # The length guard runs before the embedder, so calibrating τ against
    # utterances it already rejects measures the wrong thing — and measures it
    # badly, since those are exactly the ones that score highest.
    guarded: list[dict[str, str]] = []
    if not args.keep_short:
        from hermes.intent_gate import is_too_short

        guarded = [r for r in out_scope if is_too_short(r["text"])]
        out_scope = [r for r in out_scope if not is_too_short(r["text"])]

    print(f"{len(in_scope)} in-scope examples, {len(out_scope)} out-of-scope "
          f"sentences  (lang={args.lang}, top_k={args.top_k})")
    if guarded:
        caught = Counter(r.get("oos_type", "") for r in guarded)
        print(f"{len(guarded)} rejected by the length guard before embedding: "
              + ", ".join(f"{n} {t}" for t, n in caught.most_common()))
    print("in-scope scored leave-one-out\n")

    in_vectors = embeddings.encode([r["text"] for r in in_scope])
    out_vectors = embeddings.encode([r["text"] for r in out_scope])

    scored_in = []
    for row, vector in zip(in_scope, in_vectors):
        ranked = ranked_for(row["text"], vector, settings.intent_collection,
                            args.top_k, lang, exclude_self=True)
        if not ranked:
            continue
        top, score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        scored_in.append({
            "text": row["text"],
            "want": Intent.from_label(row["intent"]),
            "got": top,
            "score": score,
            "margin": score - second,
            "correct": top is Intent.from_label(row["intent"]),
        })

    scored_out = []
    for row, vector in zip(out_scope, out_vectors):
        ranked = ranked_for(row["text"], vector, settings.intent_collection,
                            args.top_k, lang, exclude_self=False)
        score = ranked[0][1] if ranked else 0.0
        scored_out.append({"text": row["text"], "score": score,
                           "type": row.get("oos_type", "")})

    print("score distributions")
    print(f"  {'in-scope, routed right':<26}"
          f"{np.mean([r['score'] for r in scored_in if r['correct']]):>8.3f}")
    print(f"  {'in-scope, routed wrong':<26}"
          f"{np.mean([r['score'] for r in scored_in if not r['correct']]):>8.3f}")
    print(f"  {'out of scope':<26}{np.mean([r['score'] for r in scored_out]):>8.3f}"
          f"   max {max(r['score'] for r in scored_out):.3f}")

    print(f"\nrouting accuracy with no threshold: "
          f"{sum(r['correct'] for r in scored_in) / len(scored_in):.1%}")

    print("\n-- sweeping tau --")
    header = (f"{'tau':>6}{'oos recall':>12}{'in-scope kept':>15}"
              f"{'routed right':>14}{'of all in-scope':>17}")
    print(header)
    print("-" * len(header))
    best = None
    for tau in np.arange(0.30, 0.86, 0.025):
        rejected = [r for r in scored_out if r["score"] < tau]
        kept = [r for r in scored_in if r["score"] >= tau]
        right = [r for r in kept if r["correct"]]
        recall = len(rejected) / len(scored_out)
        retention = len(kept) / len(scored_in)
        precision = len(right) / len(kept) if kept else 0.0
        coverage = len(right) / len(scored_in)
        print(f"{tau:>6.3f}{recall:>12.0%}{retention:>15.0%}"
              f"{precision:>14.1%}{coverage:>17.1%}")
        # Out-of-scope recall matters most, but a gate that keeps nothing is
        # useless, so score it by recall times the share routed correctly.
        quality = recall * coverage
        if best is None or quality > best[0]:
            best = (quality, tau, recall, retention, precision, coverage)
    print("-" * len(header))

    _, tau, recall, retention, precision, coverage = best
    print(f"\nbest balance at tau = {tau:.3f}")
    print(f"  out-of-scope recall      {recall:.0%}")
    print(f"  in-scope kept            {retention:.0%}")
    print(f"  routed right when kept   {precision:.1%}")
    print(f"  of all in-scope, right   {coverage:.1%}")

    print("\n-- margin between the top two intents, in-scope --")
    margins_right = [r["margin"] for r in scored_in if r["correct"]]
    margins_wrong = [r["margin"] for r in scored_in if not r["correct"]]
    print(f"  routed right: median {np.median(margins_right):.3f}")
    print(f"  routed wrong: median {np.median(margins_wrong):.3f}")
    print(f"\n{'margin':>8}{'would ask':>12}{'wrong caught':>15}{'right lost':>13}")
    print("-" * 48)
    for margin in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20):
        asked = [r for r in scored_in if r["margin"] < margin]
        caught = [r for r in asked if not r["correct"]]
        lost = [r for r in asked if r["correct"]]
        print(f"{margin:>8.2f}{len(asked) / len(scored_in):>12.0%}"
              f"{len(caught) / max(len(margins_wrong), 1):>15.0%}"
              f"{len(lost) / max(len(margins_right), 1):>13.0%}")
    print("-" * 48)
    print("'wrong caught' is the share of misroutes a clarifying question would")
    print("have intercepted; 'right lost' is correct routes it would interrupt.")

    print("\n-- where in-scope goes wrong --")
    confusion = Counter(
        (r["want"].value, r["got"].value) for r in scored_in if not r["correct"]
    )
    for (want, got), n in confusion.most_common(10):
        print(f"  {n:>3}x  {want:<22} -> {got}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
