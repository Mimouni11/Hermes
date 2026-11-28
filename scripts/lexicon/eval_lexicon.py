"""Measure the repair layer: does it fix what it should and leave the rest alone?

Four numbers, and the two that matter most are the ones about *not* acting:

- **fix rate** — corrupted domain terms correctly repaired
- **false-repair rate** — correct words that got changed anyway
- **safety-flag recall** — mangled safety terms that raised a flag
- **false-flag rate** — clean sentences that raised one

Three sources of test material:

1. the real transcripts from `reports/stt_results.csv`, which are what the
   model actually produced on this machine;
2. synthetic corruptions of sentences containing domain and safety terms —
   typos, phonetic swaps, and the real Whisper errors recorded in the lexicon;
3. near-miss sentences that must survive untouched ("brioche", "brosse à
   dents", "écho" as a plain echo).

Writes `reports/lexicon_eval.csv`.

    python scripts/lexicon/eval_lexicon.py
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hermes.listen import lexicon  # noqa: E402
from hermes.config import get_settings  # noqa: E402

RESULT_COLUMNS = [
    "case_id", "kind", "term", "input", "expected", "repaired",
    "edits", "flags", "outcome",
]

#: Sentences built around a term, so corruption happens in context.
TEMPLATES = [
    "Je voudrais une {term} la semaine prochaine.",
    "Le médecin a prescrit une {term}.",
    "C'est pour une {term}, c'est possible quand ?",
]

SAFETY_TEMPLATES = [
    "J'ai {term} dans le bras, je peux faire une IRM ?",
    "Le médecin a parlé de {term} avant l'examen.",
    "Avec {term}, est-ce que l'IRM est possible ?",
]

#: Sentences that must come through untouched and unflagged.
NEAR_MISSES = [
    "Je voudrais une brioche et un café.",
    "Des brioches pour le petit-déjeuner.",
    "Où est ma brosse à dents ?",
    "J'ai acheté une brosse neuve.",
    "Il y a de l'écho dans la salle.",
    "Le plan du centre est à l'entrée.",
    "Je cherche un plan de la ville.",
    "C'est un code postal, pas un code d'accès.",
    "La ceinture de sécurité est obligatoire.",
    "J'ai vidé mon sac dans la voiture.",
    "Mon énergie est au plus bas aujourd'hui.",
    "Je prends le métro tous les matins.",
    "Vous avez un parking pour les voitures ?",
    "Quels sont vos horaires d'ouverture ?",
    "Je dois annuler mon rendez-vous de jeudi.",
    "Vous êtes ouverts le samedi matin ?",
    "Je peux payer par carte bancaire ?",
    "Comment récupérer mes résultats ?",
    "Il faut une ordonnance du médecin ?",
    "Je viens accompagné de ma mère.",
]


@dataclass
class Case:
    """One evaluation case."""

    case_id: str
    kind: str
    term: str
    text: str
    expected: str


def corrupt(word: str, rng: random.Random) -> str:
    """Mangle a word the way a transcriber would: letters, not meaning."""
    swaps = [("ch", "c"), ("é", "e"), ("è", "e"), ("ph", "f"), ("qu", "k"),
             ("s", "ss"), ("c", "k"), ("in", "ain"), ("au", "o")]
    choice = rng.randrange(3)

    if choice == 0:
        for old, new in rng.sample(swaps, len(swaps)):
            if old in word.lower():
                return word.lower().replace(old, new, 1)
    if choice == 1 and len(word) > 4:
        cut = rng.randrange(1, len(word) - 1)
        return word[:cut] + word[cut + 1:]
    if len(word) > 3:
        cut = rng.randrange(1, len(word) - 1)
        return word[:cut] + word[cut] + word[cut:]
    return word


def build_cases(book: lexicon.Lexicon, rng: random.Random, root: Path) -> list[Case]:
    """Assemble the three sources of test material."""
    cases: list[Case] = []

    # 1. Real transcripts, where the reference says what should have been said.
    results = root / "reports" / "stt_results.csv"
    if results.exists():
        seen: set[str] = set()
        with open(results, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                text = row["hypothesis"].strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                cases.append(
                    Case(f"real-{len(cases):03d}", "real", "", text, row["reference"])
                )

    # 2. Synthetic corruptions, domain and safety.
    domain = [t for t in book.domain_terms if len(t) > 6][:20]
    for term in domain:
        for template in TEMPLATES[:2]:
            broken = corrupt(term, rng)
            if broken.lower() == term.lower():
                continue
            cases.append(
                Case(
                    f"dom-{len(cases):03d}", "corrupt_domain", term,
                    template.format(term=broken), template.format(term=term),
                )
            )

    for term, variants in book.safety.items():
        pool = list(variants) + [corrupt(term, rng)]
        for variant in pool[:2]:
            if variant.lower() == term.lower():
                continue
            cases.append(
                Case(
                    f"saf-{len(cases):03d}", "corrupt_safety", term,
                    SAFETY_TEMPLATES[0].format(term=variant), term,
                )
            )

    # 3. Near misses that must survive.
    for text in NEAR_MISSES:
        cases.append(Case(f"near-{len(cases):03d}", "near_miss", "", text, text))

    return cases


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the output path and the seed."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        default=settings.paths.reports / "lexicon_eval.csv")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run every case, write the table, print the four metrics."""
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    book = lexicon.load_lexicon()
    rng = random.Random(args.seed)

    cases = build_cases(book, rng, root)
    print(f"{len(cases)} cases")
    for kind in sorted({c.kind for c in cases}):
        print(f"  {kind:<18}{sum(1 for c in cases if c.kind == kind)}")

    rows: list[dict[str, object]] = []
    fixed = missed_fix = false_repair = 0
    flagged = missed_flag = false_flag = 0

    for case in cases:
        result = lexicon.repair(case.text)
        outcome = "ok"

        if case.kind == "corrupt_domain":
            # The repaired sentence should contain the real term again.
            if re.search(re.escape(case.term), result.repaired, re.IGNORECASE):
                fixed += 1
                outcome = "fixed"
            else:
                missed_fix += 1
                outcome = "not_fixed"

        elif case.kind == "corrupt_safety":
            if case.term in result.flagged_terms:
                flagged += 1
                outcome = "flagged"
            else:
                missed_flag += 1
                outcome = "not_flagged"

        elif case.kind == "near_miss":
            if result.repaired != case.text:
                false_repair += 1
                outcome = "FALSE_REPAIR"
            elif result.safety_flags:
                false_flag += 1
                outcome = "FALSE_FLAG"

        elif case.kind == "real":
            if result.changed:
                outcome = "repaired"
            if result.safety_flags:
                outcome += "+flagged"

        rows.append({
            "case_id": case.case_id,
            "kind": case.kind,
            "term": case.term,
            "input": case.text,
            "expected": case.expected,
            "repaired": result.repaired,
            "edits": " | ".join(str(e) for e in result.edits),
            "flags": " | ".join(str(f) for f in result.safety_flags),
            "outcome": outcome,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    near = sum(1 for c in cases if c.kind == "near_miss")
    domain_total = fixed + missed_fix
    safety_total = flagged + missed_flag

    print("\n" + "=" * 58)
    print(f"{'metric':<30}{'value':>12}{'n':>10}")
    print("-" * 58)
    print(f"{'fix rate (domain)':<30}"
          f"{fixed / max(domain_total, 1):>12.1%}{domain_total:>10}")
    print(f"{'false-repair rate':<30}"
          f"{false_repair / max(near, 1):>12.1%}{near:>10}")
    print(f"{'safety-flag recall':<30}"
          f"{flagged / max(safety_total, 1):>12.1%}{safety_total:>10}")
    print(f"{'false-flag rate':<30}"
          f"{false_flag / max(near, 1):>12.1%}{near:>10}")
    print("-" * 58)

    real = [r for r in rows if r["kind"] == "real"]
    touched = [r for r in real if r["outcome"] != "ok"]
    print(f"\nreal transcripts: {len(touched)} of {len(real)} touched")
    for row in touched[:10]:
        print(f"  [{row['outcome']}] {str(row['input'])[:60]}")
        if row["edits"]:
            print(f"      {row['edits']}")
        if row["flags"]:
            print(f"      {row['flags']}")

    problems = [r for r in rows if str(r["outcome"]).startswith("FALSE")]
    if problems:
        print(f"\n{len(problems)} false positives:")
        for row in problems:
            print(f"  [{row['outcome']}] {row['input']}")
            print(f"      {row['edits']}{row['flags']}")

    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
