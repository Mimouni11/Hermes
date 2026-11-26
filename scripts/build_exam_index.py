"""Build the exam gazetteer: 173 exams grouped into base exams with slots.

Writes `data/processed/exam_index.json`. Nothing here calls a model or a
server — it is a pure transform of `exams.csv` and `confusables.csv`, and it
runs in well under a second, so it is cheap to re-run after any catalogue edit.

What the index holds, and why it is shaped this way:

**Bases, not exams.** Each base is a group of exams differing only in their
slots. Measured: 173 exams collapse to 132 bases — 91 single exams, 39
droite/gauche pairs, 2 ±injection pairs. Those 39 pairs cannot be separated by
any string metric and should not be, because the caller did not say which side;
see the module docstring of `hermes.exam_bot`.

**Phrases from `keywords_fr`, not `name_fr`.** The colloquial form outnumbers
the formal one for every modality that has both: `echo` appears on 194 strings
against `echographie`'s 84, `radio` on 168 against `radiographie`'s 92. And the
formal name can be unrecognisable — `EX076` is "Radiographie du grill costal"
and nobody asks for that; its keyword "radio des côtes" is what a caller says,
and the two are 48.9 similar.

**French only.** `--langs` defaults to `fr`. Indexing `keywords_tn_latn` would
add five near-duplicate entries per exam (`irm ras`, `IRM mta3 mo5`) and let a
French utterance win against an Arabizi keyword, which distorts every margin in
the ranking. The flag exists so Darija is a rebuild rather than a rewrite.

    uv run python scripts/build_exam_index.py
    uv run python scripts/build_exam_index.py --langs fr tn --out other.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hermes.config import get_settings  # noqa: E402
from hermes.exam_bot import (  # noqa: E402
    base_key,
    base_phrase,
    match_phrase,
    modalities_in,
    pretty_label,
)

#: Which catalogue column holds each language's spoken forms.
KEYWORD_COLUMNS: dict[str, str] = {
    "fr": "keywords_fr",
    "tn": "keywords_tn",
    "tn_latn": "keywords_tn_latn",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a corpus CSV. `utf-8-sig` because Excel writes a BOM."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def exam_phrases(row: dict[str, str], langs: list[str]) -> list[str]:
    """Every spoken form for one exam, base-folded and deduplicated.

    `name_fr` is included even though it is often not how anyone speaks: it
    costs one phrase per exam and it is the only form a caller reading from a
    prescription will use.
    """
    strings = [row.get("name_fr", "")]
    for lang in langs:
        column = KEYWORD_COLUMNS[lang]
        strings.extend(k.strip() for k in (row.get(column) or "").split("|"))

    seen: dict[str, None] = {}
    for raw in strings:
        # `match_phrase`, not `base_phrase`: the filler words come off here, at
        # build time, so the stored phrases are already in the form the scorer
        # compares and neither side pays for it per call.
        phrase = match_phrase(raw)
        if phrase and phrase not in seen:
            seen[phrase] = None
    return list(seen)


def build(
    exams: list[dict[str, str]],
    confusables: list[dict[str, str]],
    langs: list[str],
) -> dict[str, Any]:
    """Group the catalogue into bases and assemble the index payload.

    Rejects a duplicate `id` outright. The catalogue is keyed by id into a
    dict, so a repeated id silently overwrites the earlier row and one exam
    disappears from the index while the build still reports 173 exams.
    """
    seen: dict[str, int] = {}
    for position, row in enumerate(exams, start=1):
        exam_id = (row.get("id") or "").strip()
        if not exam_id:
            raise ValueError(f"row {position} of the catalogue has no id")
        if exam_id in seen:
            raise ValueError(
                f"duplicate exam id {exam_id!r} at rows {seen[exam_id]} "
                f"and {position} — one row would overwrite the other"
            )
        seen[exam_id] = position

    members: dict[str, list[dict[str, str]]] = defaultdict(list)
    phrases: dict[str, dict[str, None]] = defaultdict(dict)
    modalities: dict[str, set[str]] = defaultdict(set)
    display: dict[str, str] = {}
    label: dict[str, str] = {}

    for row in exams:
        key = base_key(row["modality"], row["name_fr"])
        members[key].append(row)

        for phrase in exam_phrases(row, langs):
            phrases[key][phrase] = None
            # A base accepts the modalities its own phrases mention, not only
            # the one in its column. This is what keeps the four dentaire
            # exams reachable: EX172's keywords are "radio d'une dent" and
            # "radio de la dent", so a caller saying "radio" must not have it
            # filtered out by its `dentaire` column.
            modalities[key] |= modalities_in(phrase)

        modalities[key].add(row["modality"])
        display.setdefault(key, base_phrase(row["name_fr"]))
        # The spoken form, built from the first member's name with the slot
        # words taken out: "IRM de l'épaule", not "irm de l epaule".
        label.setdefault(key, pretty_label(row["name_fr"]))

    bases: list[dict[str, Any]] = []
    for key, rows in members.items():
        bases.append({
            "key": key,
            "modality": rows[0]["modality"],
            "display": display[key],
            "label": label[key],
            "phrases": sorted(phrases[key]),
            "exam_ids": sorted(r["id"] for r in rows),
            "modalities": sorted(modalities[key]),
            "laterality_options": sorted({r["laterality"] for r in rows}),
            "contrast_options": sorted({r["contrast"] for r in rows}),
        })
    bases.sort(key=lambda b: b["key"])

    catalogue = {
        row["id"]: {
            "name": row["name_fr"],
            "modality": row["modality"],
            "laterality": row["laterality"],
            "contrast": row["contrast"],
            "region": row.get("region_fr", ""),
            "duration_min": int(row.get("duration_min") or 0),
            "escalate": row.get("escalate", "0").strip() == "1",
            "escalate_reason": (row.get("escalate_reason") or "").strip(),
        }
        for row in exams
    }

    known = set(catalogue)
    pairs = [
        {
            "a": row["exam_a"],
            "b": row["exam_b"],
            "type": row.get("type", ""),
            "label": row.get("label", ""),
        }
        for row in confusables
        if row.get("exam_a") in known and row.get("exam_b") in known
    ]

    return {
        "langs": langs,
        "exams": catalogue,
        "bases": bases,
        "confusables": pairs,
    }


def report(payload: dict[str, Any]) -> list[str]:
    """Print what was built and return the integrity problems found.

    The checks used to print a `!!` line and let the build succeed, which
    defeats their purpose — the whole reason they exist is that a hole in the
    index surfaces only as an unexplained miss, and nobody re-reads the output
    of a command that exited 0. The caller now refuses to publish on any
    non-empty return.
    """
    bases = payload["bases"]
    exams = payload["exams"]

    single = [b for b in bases if len(b["exam_ids"]) == 1]
    side_split = [
        b for b in bases
        if len({o for o in b["laterality_options"] if o in ("droite", "gauche")}) > 1
    ]
    contrast_split = [b for b in bases if len(set(b["contrast_options"])) > 1]
    escalate = [i for i, row in exams.items() if row["escalate"]]
    phrase_count = sum(len(b["phrases"]) for b in bases)

    print(f"  {len(exams)} exams -> {len(bases)} bases, {phrase_count} phrases")
    print(f"    {len(single)} bases hold one exam")
    print(f"    {len(side_split)} need a side before they resolve")
    print(f"    {len(contrast_split)} need an injection answer")
    print(f"    {len(payload['confusables'])} confusable pairs")
    print(f"    {len(escalate)} exams transfer: {', '.join(sorted(escalate))}")

    problems: list[str] = []

    # Every exam must be reachable through exactly one base, or the ranking has
    # a hole that no amount of threshold tuning will show up as anything other
    # than an unexplained miss.
    placed = [i for b in bases for i in b["exam_ids"]]
    missing = set(exams) - set(placed)
    if missing:
        problems.append(f"{len(missing)} exams are in no base: {sorted(missing)}")
    duplicated = {i for i in placed if placed.count(i) > 1}
    if duplicated:
        problems.append(f"exams in more than one base: {sorted(duplicated)}")

    empty = [b["key"] for b in bases if not b["phrases"]]
    if empty:
        problems.append(f"bases with no phrases: {empty}")

    for problem in problems:
        print(f"  !! {problem}")
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the catalogue paths, the languages to index and the output path."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exams", type=Path,
                        default=settings.paths.raw / "exams.csv")
    parser.add_argument("--confusables", type=Path,
                        default=settings.paths.raw / "confusables.csv")
    parser.add_argument("--langs", nargs="+", default=["fr"],
                        choices=sorted(KEYWORD_COLUMNS),
                        help="which keyword columns to index (default: fr)")
    parser.add_argument("--out", type=Path,
                        default=(settings.paths.processed
                                 / settings.exam_index_file))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the exam index and write it."""
    args = parse_args(argv)

    if not args.exams.exists():
        print(f"no catalogue at {args.exams}")
        return 1

    exams = read_csv(args.exams)
    confusables = read_csv(args.confusables) if args.confusables.exists() else []
    if not args.confusables.exists():
        print(f"  no confusables at {args.confusables} — continuing without")

    print(f"indexing {', '.join(args.langs)} from {args.exams.name}")
    payload = build(exams, confusables, list(args.langs))
    problems = report(payload)
    if problems:
        print(f"\nrefusing to write {args.out.name}: "
              f"{len(problems)} integrity problem(s) above.")
        print("The existing index is left untouched.")
        return 1

    # Written to a sibling temp file and moved into place, so a reader never
    # sees half a document. `load_index()` parses JSON, and a truncated file
    # raises JSONDecodeError, which the Gradio handler does not catch — a
    # rebuild while the bench is open would have surfaced as a stack trace.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging = args.out.with_suffix(args.out.suffix + ".partial")
    staging.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    staging.replace(args.out)
    size_kb = args.out.stat().st_size / 1024
    print(f"\nwritten: {args.out}  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
