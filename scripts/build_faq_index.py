"""Embed the FAQ into Qdrant for retrieval.

Reads `data/raw/faq.csv` and writes one point **per question variant** to the
`hermes_faq` collection. The CSV holds 24 entries, each with a French, Darija
and Arabizi phrasing of the same question, so the collection holds up to 72
points pointing at 24 answers.

One point per variant, not per entry, because a Darija caller has to match a
Darija question vector. Embedding only the French phrasing would make every
Darija question a cross-lingual match, which is the harder problem and the one
with the worse scores.

The answer travels in the payload rather than being looked up afterwards: at 24
entries there is no reason for a second round trip.

    python scripts/build_faq_index.py
    python scripts/build_faq_index.py --recreate
    python scripts/build_faq_index.py --dry-run
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

#: Which CSV column holds the question in which language.
QUESTION_COLUMNS = {
    "question_fr": "fr",
    "question_tn": "tn",
    "question_tn_latn": "tn_latn",
}

#: Extra phrasings of the same question, pipe-separated, indexed alongside the
#: primary one. Each is its own point: a caller's wording only has to be close
#: to *one* of them, which is what lifts the similarity score rather than just
#: the ranking.
VARIANT_COLUMNS = {"question_fr_variants": "fr"}

VARIANT_SEPARATOR = "|"

#: Answer columns carried on every point, whatever the question's language —
#: the agent may answer in French to a Darija question, which is what the plan
#: asks for while there is no Darija voice.
ANSWER_COLUMNS = ["answer_fr", "answer_tn"]

#: Which answer a question variant is paired with when the answer is embedded
#: too. Darija and Arabizi questions pair with the Darija answer where there is
#: one, so the embedded text stays in one language.
ANSWER_FOR_LANG = {"fr": "answer_fr", "tn": "answer_tn", "tn_latn": "answer_tn"}


def embedded_text(record: dict[str, str], mode: str) -> str:
    """Return the text that actually goes into the vector.

    ``question`` matches a caller's phrasing against the way the question is
    written, which is the usual choice: an answer is written for a reader and
    carries vocabulary that pulls in false matches — an answer listing "8h à
    18h" and "samedi" can hijack any query mentioning a day or a time.

    ``question+answer`` gives each entry more surface to match against, which
    helps when the questions are short and the test queries paraphrase them
    loosely. Which one wins is a measurement, not a principle; see docs/WP3.md.
    """
    question = record["question"]
    if mode == "question":
        return question

    column = ANSWER_FOR_LANG.get(record["lang"], "answer_fr")
    answer = (record.get(column) or record.get("answer_fr") or "").strip()
    return f"{question} {answer}".strip()


def load_points(path: Path) -> list[dict[str, str]]:
    """Expand the FAQ into one record per question variant."""
    records: list[dict[str, str]] = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            faq_id = (row.get("faq_id") or "").strip()
            if not faq_id:
                continue
            answers = {
                column: (row.get(column) or "").strip()
                for column in ANSWER_COLUMNS
            }
            for column, lang in QUESTION_COLUMNS.items():
                question = (row.get(column) or "").strip()
                if not question:
                    continue
                records.append(
                    {
                        "faq_id": faq_id,
                        "question": question,
                        "lang": lang,
                        "phrasing": "primary",
                        **answers,
                    }
                )

            for column, lang in VARIANT_COLUMNS.items():
                for variant in (row.get(column) or "").split(VARIANT_SEPARATOR):
                    variant = variant.strip()
                    if not variant:
                        continue
                    records.append(
                        {
                            "faq_id": faq_id,
                            "question": variant,
                            "lang": lang,
                            "phrasing": "variant",
                            **answers,
                        }
                    )
    return records


def report(records: list[dict[str, str]]) -> None:
    """Print how many question variants exist per language."""
    counts = Counter(r["lang"] for r in records)
    entries = len({r["faq_id"] for r in records})

    print(f"\n{entries} FAQ entries, {len(records)} question variants")
    header = f"{'lang':<10}{'variants':>10}{'coverage':>11}"
    print(header)
    print("-" * len(header))
    for lang in sorted(counts):
        print(f"{lang:<10}{counts[lang]:>10}{counts[lang] / entries:>10.0%}")
    print("-" * len(header))

    missing = [
        lang for lang in QUESTION_COLUMNS.values() if counts.get(lang, 0) < entries
    ]
    if missing:
        print(
            f"not every entry has a {', '.join(missing)} phrasing — a caller in "
            "that language can only match the entries that do"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the FAQ path, the collection and the rebuild flags."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--faq", type=Path,
                        default=settings.paths.raw / "faq.csv",
                        help="the FAQ question/answer CSV")
    parser.add_argument("--collection", default=settings.faq_collection,
                        help="the Qdrant collection to write to")
    parser.add_argument("--recreate", action="store_true",
                        help="drop the collection first, so deleted rows go too")
    parser.add_argument("--embed", choices=["question", "question+answer"],
                        default="question",
                        help="what goes into the vector (the payload is "
                             "unchanged either way)")
    parser.add_argument("--dry-run", action="store_true",
                        help="read and report, embed and write nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Embed every question variant and write it to Qdrant."""
    args = parse_args(argv)

    if not args.faq.exists():
        print(f"FAQ not found: {args.faq}", file=sys.stderr)
        return 1

    records = load_points(args.faq)
    if not records:
        print(f"no usable rows in {args.faq}", file=sys.stderr)
        return 1

    report(records)

    if args.dry_run:
        print("\n--dry-run: nothing embedded, nothing written")
        return 0

    vector_store.require_server()
    dim = embeddings.check_dimension()
    print(f"\nembedding {args.embed} with {get_settings().embedding_model} ({dim}d)...")

    texts = [embedded_text(r, args.embed) for r in records]
    vectors = embeddings.encode(texts)
    print(f"encoded {vectors.shape[0]} questions")

    # Record what went into the vector, so a collection can be told apart from
    # one built the other way without re-reading the CSV.
    for record, text in zip(records, texts):
        record["embedded"] = args.embed
        record["embedded_text"] = text

    # The key must include the phrasing's position: an entry now has several
    # French phrasings, and keying on (faq_id, lang) alone would make them
    # overwrite each other, leaving one point per entry and silently undoing
    # the whole reason for adding variants.
    keys = []
    seen: dict[tuple[str, str], int] = {}
    for record in records:
        pair = (record["faq_id"], record["lang"])
        index = seen.get(pair, 0)
        seen[pair] = index + 1
        keys.append(f"{record['faq_id']}:{record['lang']}:{index}")

    vector_store.ensure_collection(args.collection, dim, recreate=args.recreate)
    written = vector_store.upsert(args.collection, vectors, records, keys=keys)

    total = vector_store.count(args.collection)
    print(f"wrote {written} points to {args.collection!r}, which now holds {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
