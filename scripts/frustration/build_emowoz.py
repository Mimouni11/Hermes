"""Turn EmoWOZ into auxiliary training rows for the frustration detector.

EmoWOZ (Feng et al., LREC 2022) is 11,434 task-oriented dialogues with every
**user** turn labelled for emotion. It is released under **CC-BY-NC-4.0**, so
it may be used here — a school project — but never commercially, and the paper
must be cited. The underlying MultiWOZ dialogues carry their own licence,
which the EmoWOZ card does not restate.

Two gaps this data has to cross, and they are worth naming because the
measurement is the only thing that settles them:

- **Language.** EmoWOZ is English, this agent is French. The bet is the
  embedder: `paraphrase-multilingual-MiniLM-L12-v2` places both languages in
  one space, so English annoyance should land near French annoyance.
- **Domain.** EmoWOZ books restaurants, hotels and trains, not medical
  imaging. That cuts the other way in our favour: being annoyed *at a booking
  system* is exactly the label, and the nouns are irrelevant.

Nothing here touches the French test folds. These rows are training material
only, which `eval_frustration.py` enforces.

    uv run python scripts/frustration/build_emowoz.py
    uv run python scripts/frustration/build_emowoz.py --target 1500 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hermes.config import get_settings  # noqa: E402

ZENODO = "https://zenodo.org/record/6506504/files"
FILES = {
    "multiwoz": "emowoz-multiwoz.json",
    "dialmage": "emowoz-dialmage.json",
    "split": "data-split.json",
}

#: EmoWOZ's label ids, mapped onto this project's binary label and its `kind`
#: taxonomy. The mapping is the point of the exercise, so it is written out
#: rather than inferred:
#:
#:   dissatisfied and abusive are annoyance *elicited by the system* — the
#:   exact definition of label 1 here.
#:
#:   fearful/sad/disappointed is distress about something outside the system's
#:   control, which is this project's `worry`: calm towards us, and a
#:   different escalation question.
#:
#:   apologetic is the user walking something back, which is `correction` —
#:   the hard negative the French lexicon already trips over.
MAPPING: dict[int, tuple[int, str]] = {
    0: (0, "neutral"),
    1: (0, "worry"),
    2: (1, "frustrated"),
    3: (0, "correction"),
    4: (1, "frustrated"),
    5: (0, "positive"),
    6: (0, "positive"),
}

EMOTION_NAMES = {
    0: "neutral", 1: "fearful/sad", 2: "dissatisfied", 3: "apologetic",
    4: "abusive", 5: "excited", 6: "satisfied",
}


#: Measured with HEAD on the Zenodo record:
#:
#:     emowoz-multiwoz.json   159.6 MB
#:     emowoz-dialmage.json    18.1 MB
#:     data-split.json          0.3 MB
#:
#: The first version of this function called `response.read()` once, which
#: tried to hold 160 MB in memory and died at 3.6 MB with `IncompleteRead`.
#: Zenodo advertises `Accept-Ranges: bytes` and a range request returns 206,
#: so the fix is to stream in chunks and resume from whatever is on disk.
CHUNK = 1 << 20
ATTEMPTS = 5


def fetch(destination: Path, name: str, attempts: int = ATTEMPTS) -> Path:
    """Download one Zenodo file, resuming a partial attempt if there is one."""
    import time

    target = destination / name
    if target.exists() and target.stat().st_size:
        print(f"  {name}: already present ({target.stat().st_size / 1e6:.1f} MB)")
        return target

    url = f"{ZENODO}/{name}"
    # The partial file is kept between attempts on purpose — deleting it is
    # what would turn a flaky connection into an unwinnable one.
    staging = target.with_suffix(target.suffix + ".partial")

    for attempt in range(1, attempts + 1):
        have = staging.stat().st_size if staging.exists() else 0
        headers = {"User-Agent": "cli-client"}
        if have:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                resuming = response.status == 206
                if have and not resuming:
                    # The server ignored the range, so start over rather than
                    # append the whole file onto a partial one.
                    print(f"  {name}: server ignored resume, restarting")
                    have = 0
                total = int(response.headers.get("Content-Length") or 0) + have
                mode = "ab" if resuming and have else "wb"
                shown = -1
                with open(staging, mode) as out:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        have += len(chunk)
                        if total:
                            percent = int(100 * have / total)
                            if percent >= shown + 5:
                                shown = percent
                                print(f"    {name}: {percent:>3}% "
                                      f"({have / 1e6:.1f}/{total / 1e6:.1f} MB)",
                                      flush=True)
            if total and have < total:
                raise OSError(f"short read: {have} of {total} bytes")
            break
        except Exception as exc:  # noqa: BLE001 — any transport failure retries
            kept = staging.stat().st_size if staging.exists() else 0
            print(f"  {name}: attempt {attempt}/{attempts} failed "
                  f"({type(exc).__name__}), {kept / 1e6:.1f} MB kept")
            if attempt == attempts:
                print(f"  {name}: giving up. Re-run to resume from "
                      f"{kept / 1e6:.1f} MB.")
                raise
            time.sleep(2 * attempt)

    staging.replace(target)
    print(f"  {name}: {target.stat().st_size / 1e6:.1f} MB")
    return target


def user_turns(dialogues: dict, ids: list[str]) -> list[dict[str, object]]:
    """Every labelled user turn of the given dialogues.

    Turns alternate and the user speaks first, so user turns sit at even
    indices. The gold emotion is the fourth annotation entry — the loading
    script reads `emotion[3]["emotion"]`, and this follows it rather than
    inventing an aggregation.
    """
    rows: list[dict[str, object]] = []
    for dialogue_id in ids:
        dialogue = dialogues.get(dialogue_id)
        if not dialogue:
            continue
        for index, turn in enumerate(dialogue["log"]):
            if index % 2 != 0:
                continue  # system turn
            try:
                emotion = int(turn["emotion"][3]["emotion"])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            text = (turn.get("text") or "").strip()
            if not text or emotion not in MAPPING:
                continue
            label, kind = MAPPING[emotion]
            rows.append({
                "id": f"{dialogue_id}#{index}",
                "text": text,
                "label": label,
                "lang": "en",
                "kind": kind,
                "group": dialogue_id,
                "emotion": emotion,
            })
    return rows


def sample(
    rows: list[dict[str, object]], target: int, seed: int,
) -> list[dict[str, object]]:
    """Sample to roughly balanced, keeping every frustrated row.

    `dissatisfied` is the scarce class — the EmoWOZ card notes neutral,
    satisfied and dissatisfied are over 95 % of labels, and the negatives
    dominate. So the positives set the budget and the negatives are drawn down
    to match, spread evenly across the four calm kinds so `worry` and
    `correction` are not crowded out by `neutral`. Those two are the hard
    negatives this data is being added for.
    """
    rng = random.Random(seed)
    positive = [r for r in rows if r["label"] == 1]
    rng.shuffle(positive)
    keep_positive = positive[: target // 2]

    by_kind: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row["label"] == 0:
            by_kind.setdefault(str(row["kind"]), []).append(row)

    budget = len(keep_positive)
    share = max(1, budget // max(1, len(by_kind)))
    keep_negative: list[dict[str, object]] = []
    for kind in sorted(by_kind):
        pool = by_kind[kind]
        rng.shuffle(pool)
        keep_negative.extend(pool[:share])

    # If a scarce kind could not fill its share, top up from neutral so the
    # two classes stay balanced overall.
    shortfall = budget - len(keep_negative)
    if shortfall > 0:
        taken = {id(r) for r in keep_negative}
        spare = [r for r in by_kind.get("neutral", []) if id(r) not in taken]
        keep_negative.extend(spare[:shortfall])

    out = keep_positive + keep_negative
    rng.shuffle(out)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path,
                        default=settings.paths.raw / "emowoz")
    parser.add_argument("--out", type=Path,
                        default=settings.paths.processed / "emowoz_train.csv")
    parser.add_argument("--target", type=int, default=1600,
                        help="roughly how many rows to keep (half positive)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--splits", nargs="+", default=["train"],
                        help="which EmoWOZ splits to draw from")
    parser.add_argument("--sources", nargs="+", default=["multiwoz", "dialmage"],
                        choices=["multiwoz", "dialmage"],
                        help="multiwoz is 159.6 MB, dialmage 18.1 MB")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)

    print("EmoWOZ (CC-BY-NC-4.0, Feng et al. LREC 2022)")
    needed = {"split": FILES["split"]}
    needed.update({key: FILES[key] for key in args.sources})
    paths = {key: fetch(args.cache, name) for key, name in needed.items()}

    dialogues: dict = {}
    per_source: dict[str, set[str]] = {}
    for key in args.sources:
        block = json.loads(paths[key].read_text(encoding="utf-8"))
        per_source[key] = set(block)
        dialogues.update(block)
    data_split = json.loads(paths["split"].read_text(encoding="utf-8"))

    ids: list[str] = []
    for split in args.splits:
        block = data_split[split]
        for key in args.sources:
            ids.extend(block.get(key, []))
    ids = sorted(set(ids))
    print(f"\n{len(dialogues)} dialogues loaded from "
          f"{', '.join(args.sources)}, "
          f"{len(ids)} in split(s) {', '.join(args.splits)}")

    rows = user_turns(dialogues, ids)
    print(f"{len(rows)} labelled user turns\n")

    print("raw emotion distribution")
    counts = Counter(int(r["emotion"]) for r in rows)
    for emotion, count in sorted(counts.items()):
        label, kind = MAPPING[emotion]
        print(f"  {emotion} {EMOTION_NAMES[emotion]:<14} {count:>6}"
              f"   -> label {label} kind {kind}")

    # Per source, because the two are not interchangeable. DialMAGE came from
    # a human evaluation of a weaker dialogue policy, so it plausibly holds a
    # higher share of genuine annoyance than the scripted MultiWOZ
    # collection — plausibly, not measurably, until this prints.
    if len(args.sources) > 1:
        print("\nfrustrated share by source")
        for key, keys in per_source.items():
            subset = [r for r in rows if str(r["group"]) in keys]
            if not subset:
                continue
            positive = sum(1 for r in subset if r["label"] == 1)
            print(f"  {key:<10} {positive:>5}/{len(subset):<6} turns "
                  f"({positive / len(subset):.1%} frustrated)")

    kept = sample(rows, args.target, args.seed)
    print(f"\nsampled {len(kept)} rows: "
          f"{sum(r['label'] == 1 for r in kept)} frustrated, "
          f"{sum(r['label'] == 0 for r in kept)} calm")
    print("  by kind: " + ", ".join(
        f"{kind} {count}" for kind, count in
        sorted(Counter(str(r["kind"]) for r in kept).items())
    ))
    print(f"  distinct dialogues (groups): "
          f"{len({str(r['group']) for r in kept})}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["id", "text", "label", "lang", "kind", "group"],
        )
        writer.writeheader()
        for row in kept:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
