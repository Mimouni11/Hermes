"""Embed the labelled intent examples into Qdrant.

Reads `data/raw/intents.csv`, embeds every example in every language, and
writes one point per sentence to the `hermes_intents` collection. The intent
gate then routes a caller by nearest neighbours rather than by a centroid: a
centroid averages "je veux annuler" together with "je dois déplacer mon
rendez-vous de jeudi", and the average is near neither.

Supersedes `build_intent_centroids.py`, which wrote one averaged vector per
intent to a file. Both the method and the destination changed; see docs/WP3.md.

Re-running is safe: point ids are derived from the corpus id, so a rebuild
overwrites rather than duplicates. `--recreate` drops the collection first,
which is what you want after deleting rows from the CSV.

    python scripts/build_intent_index.py
    python scripts/build_intent_index.py --recreate
    python scripts/build_intent_index.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hermes.infra import embeddings, vector_store  # noqa: E402
from hermes.config import get_settings  # noqa: E402

#: Payload fields copied from the CSV onto every point.
CARRIED = ["id", "text", "intent", "sub_intent", "lang", "exam_id"]


def load_rows(path: Path) -> list[dict[str, str]]:
    """Read the intent corpus, keeping rows that have both text and a label."""
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            text = (row.get("text") or "").strip()
            intent = (row.get("intent") or "").strip()
            if not text or not intent:
                continue
            rows.append({key: (row.get(key) or "").strip() for key in CARRIED})
    return rows


def report(rows: list[dict[str, str]]) -> None:
    """Print how the examples are spread across intents and languages."""
    languages = sorted({r["lang"] for r in rows})
    counts: Counter = Counter((r["intent"], r["lang"]) for r in rows)

    header = f"{'intent':<22}" + "".join(f"{lang:>9}" for lang in languages) + f"{'total':>8}"
    print(f"\n{header}")
    print("-" * len(header))
    for intent in sorted({r["intent"] for r in rows}):
        line = f"{intent:<22}"
        total = 0
        for lang in languages:
            n = counts[(intent, lang)]
            total += n
            line += f"{n:>9}"
        print(line + f"{total:>8}")
    print("-" * len(header))
    line = f"{'total':<22}"
    for lang in languages:
        line += f"{sum(1 for r in rows if r['lang'] == lang):>9}"
    print(line + f"{len(rows):>8}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the corpus path, the collection and the rebuild flags."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--intents", type=Path,
                        default=settings.paths.raw / "intents.csv",
                        help="the labelled intent corpus")
    parser.add_argument("--collection", default=settings.intent_collection,
                        help="the Qdrant collection to write to")
    parser.add_argument("--recreate", action="store_true",
                        help="drop the collection first, so deleted rows go too")
    parser.add_argument("--dry-run", action="store_true",
                        help="read and report, embed and write nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Embed the corpus and write it to Qdrant, reporting the spread."""
    args = parse_args(argv)

    if not args.intents.exists():
        print(f"intent corpus not found: {args.intents}", file=sys.stderr)
        return 1

    rows = load_rows(args.intents)
    if not rows:
        print(f"no usable rows in {args.intents}", file=sys.stderr)
        return 1

    print(f"{len(rows)} labelled examples in {args.intents.name}")
    report(rows)

    if args.dry_run:
        print("\n--dry-run: nothing embedded, nothing written")
        return 0

    vector_store.require_server()
    dim = embeddings.check_dimension()
    print(f"\nembedding with {get_settings().embedding_model} ({dim}d)...")

    vectors = embeddings.encode([r["text"] for r in rows])
    print(f"encoded {vectors.shape[0]} sentences")

    vector_store.ensure_collection(args.collection, dim, recreate=args.recreate)
    written = vector_store.upsert(
        args.collection, vectors, rows, keys=[r["id"] for r in rows]
    )

    total = vector_store.count(args.collection)
    print(f"wrote {written} points to {args.collection!r}, which now holds {total}")
    if total != len(rows) and not args.recreate:
        print(
            f"note: the collection holds {total} points against {len(rows)} rows "
            "in the CSV — rows deleted from the corpus are still indexed. "
            "Re-run with --recreate to clear them."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
