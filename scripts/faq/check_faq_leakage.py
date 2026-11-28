"""Check that no FAQ phrasing duplicates a test query.

The FAQ's question variants are what the retrieval indexes; `faq_test.csv` is
what it is scored against. If a variant and a test query are the same sentence,
that query stops being a test — the index contains the answer verbatim and the
retrieval only has to find an exact match.

This is the kind of leak that inflates a number quietly and is invisible in the
result, so it is worth a script rather than a glance. Run it whenever either
file changes.

    python scripts/faq/check_faq_leakage.py
    python scripts/faq/check_faq_leakage.py --threshold 85
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hermes.config import get_settings  # noqa: E402

#: Above this RapidFuzz ratio two sentences are treated as the same question.
DEFAULT_THRESHOLD = 88.0

VARIANT_SEPARATOR = "|"


def fold(text: str) -> str:
    """Lower-case, strip accents and punctuation, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    kept = [c if (c.isalnum() or c.isspace()) else " " for c in stripped.lower()]
    return " ".join("".join(kept).split())


def faq_phrasings(path: Path) -> list[tuple[str, str, str]]:
    """Return (faq_id, where, text) for every French phrasing in the FAQ."""
    out: list[tuple[str, str, str]] = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            faq_id = (row.get("faq_id") or "").strip()
            if not faq_id:
                continue
            primary = (row.get("question_fr") or "").strip()
            if primary:
                out.append((faq_id, "question_fr", primary))
            for variant in (row.get("question_fr_variants") or "").split(
                VARIANT_SEPARATOR
            ):
                variant = variant.strip()
                if variant:
                    out.append((faq_id, "variant", variant))
    return out


def test_queries(path: Path, lang: str = "fr") -> list[tuple[str, str, str]]:
    """Return (id, faq_id, text) for the test queries in one language."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return [
            (
                (row.get("id") or "").strip(),
                (row.get("faq_id") or "").strip(),
                (row.get("text") or "").strip(),
            )
            for row in csv.DictReader(handle)
            if (row.get("text") or "").strip() and row.get("lang") == lang
        ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the two paths and the similarity threshold."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--faq", type=Path, default=settings.paths.raw / "faq.csv")
    parser.add_argument("--test", type=Path,
                        default=settings.paths.raw / "faq_test.csv")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="RapidFuzz ratio above which two questions are "
                             "treated as the same")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Report the phrasing counts and any overlap with the test set."""
    args = parse_args(argv)
    from rapidfuzz import fuzz

    phrasings = faq_phrasings(args.faq)
    queries = test_queries(args.test)

    entries = sorted({faq_id for faq_id, _, _ in phrasings})
    variants = [p for p in phrasings if p[1] == "variant"]

    print(f"{len(entries)} FAQ entries")
    print(f"{len(phrasings)} French phrasings indexed "
          f"({len(entries)} primary + {len(variants)} variants)")
    print(f"average {len(phrasings) / len(entries):.1f} phrasings per entry")

    thin = [
        faq_id for faq_id in entries
        if sum(1 for f, _, _ in phrasings if f == faq_id) < 3
    ]
    if thin:
        print(f"\nfewer than 3 phrasings: {', '.join(thin)}")

    print(f"\nchecking {len(phrasings)} phrasings against "
          f"{len(queries)} French test queries at ratio >= {args.threshold:g}")

    collisions = []
    for query_id, want, query in queries:
        folded_query = fold(query)
        for faq_id, where, phrasing in phrasings:
            ratio = fuzz.ratio(folded_query, fold(phrasing))
            if ratio >= args.threshold:
                collisions.append((query_id, want, query, faq_id, where,
                                   phrasing, ratio))

    if not collisions:
        print("\nno overlap: every test query is distinct from every phrasing")
        return 0

    print(f"\n{len(collisions)} overlaps — these test queries are no longer a test:")
    for query_id, want, query, faq_id, where, phrasing, ratio in collisions:
        flag = "SAME ENTRY" if want == faq_id else "different entry"
        print(f"\n  [{query_id}] {query}")
        print(f"      {ratio:.0f}% match with {faq_id}.{where}: {phrasing}")
        print(f"      ({flag})")
    print("\nReword the FAQ variant, not the test query — the test set is the "
          "fixed thing.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
