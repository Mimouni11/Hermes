"""Pick the French sentences for the test audio set and write the manifest.

Chooses 20 French sentences from `data/raw/intents.csv` with a fixed seed — 8 to
be synthesised by `make_tts_clips.py` and 12 for a human to record — and writes
`data/audio/manifest.csv`, which every later audio script reads.

The selection is constrained, not merely random:

- all six intents appear, and the eight TTS clips cover all six between them,
  so a modality of recording never hides an intent from the evaluation,
- at least :data:`MIN_HARD_WORDS` sentences contain a hard medical word (see
  :data:`HARD_MEDICAL_WORDS`) — "densitométrie" and "mammographie" are where a
  French ASR model actually fails, so they must be in the set, not left to luck.

The same seed always gives the same 20 sentences and the same split, so a
recording made today still matches the manifest regenerated next week.

    python scripts/pick_audio_sentences.py
    python scripts/pick_audio_sentences.py --seed 11 --total 20 --tts 8
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INTENTS_CSV = _PROJECT_ROOT / "data" / "raw" / "intents.csv"
DEFAULT_AUDIO_DIR = _PROJECT_ROOT / "data" / "audio"

#: Only French for now; the Tunisian Arabic sets come later.
LANGUAGE = "fr"

DEFAULT_SEED = 7
DEFAULT_TOTAL = 20
DEFAULT_TTS = 8

#: How many of the chosen sentences must contain a hard medical word.
MIN_HARD_WORDS = 3

#: The columns of `data/audio/manifest.csv`, in order.
MANIFEST_COLUMNS = [
    "clip_id",
    "text",
    "lang",
    "intent",
    "exam_id",
    "source",
    "noise_snr_db",
    "file",
]

#: Terms a French speech model gets wrong: long, rare, or borrowed. Matched
#: without accents or case, so "densitometrie" counts as "densitométrie".
HARD_MEDICAL_WORDS = [
    "densitométrie",
    "ostéodensitométrie",
    "ostéoporose",
    "mammographie",
    "échographie",
    "hystérosalpingographie",
    "cholangiographie",
    "arthroscanner",
    "coloscanner",
    "entéroscanner",
    "uroscanner",
    "télérachis",
    "sénologique",
    "cone beam",
    "doppler",
    "gadolinium",
    "créatinine",
    "pacemaker",
    "claustrophobie",
    "produit de contraste",
    "iode",
    "rachis",
]

#: Where each kind of clip lives, relative to `data/audio`.
SOURCE_DIRS = {"tts": "tts", "recorded": "recorded", "noisy": "noisy"}


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

def fold(text: str) -> str:
    """Lower-case ``text`` and strip its diacritics, for tolerant matching."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def hard_words_in(text: str) -> list[str]:
    """Return the hard medical words this sentence contains."""
    folded = fold(text)
    return [word for word in HARD_MEDICAL_WORDS if fold(word) in folded]


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Sentence:
    """One candidate row of the intent corpus."""

    id: str
    text: str
    intent: str
    exam_id: str
    lang: str

    @property
    def hard(self) -> bool:
        """Whether this sentence contains a hard medical word."""
        return bool(hard_words_in(self.text))


@dataclass(frozen=True)
class Clip:
    """One row of the manifest: a sentence plus where its audio will live."""

    clip_id: str
    sentence: Sentence
    source: str
    file: str
    noise_snr_db: int | None = None

    def as_row(self) -> dict[str, object]:
        """Render this clip as a manifest row."""
        return {
            "clip_id": self.clip_id,
            "text": self.sentence.text,
            "lang": self.sentence.lang,
            "intent": self.sentence.intent,
            "exam_id": self.sentence.exam_id,
            "source": self.source,
            "noise_snr_db": "" if self.noise_snr_db is None else self.noise_snr_db,
            "file": self.file,
        }


# --------------------------------------------------------------------------
# Reading the corpus
# --------------------------------------------------------------------------

def load_sentences(path: Path, language: str = LANGUAGE) -> list[Sentence]:
    """Read the intent corpus and return its sentences in one language."""
    sentences: list[Sentence] = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("lang") or "").strip() != language:
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            sentences.append(
                Sentence(
                    id=(row.get("id") or "").strip(),
                    text=text,
                    intent=(row.get("intent") or "").strip(),
                    exam_id=(row.get("exam_id") or "").strip(),
                    lang=language,
                )
            )
    return sentences


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def _by_intent(sentences: list[Sentence]) -> dict[str, list[Sentence]]:
    grouped: dict[str, list[Sentence]] = {}
    for sentence in sentences:
        grouped.setdefault(sentence.intent, []).append(sentence)
    for pool in grouped.values():
        pool.sort(key=lambda s: s.id)
    return grouped


def select(
    sentences: list[Sentence],
    rng: random.Random,
    total: int = DEFAULT_TOTAL,
    min_hard: int = MIN_HARD_WORDS,
) -> list[Sentence]:
    """Choose ``total`` sentences covering every intent and enough hard words.

    Three passes, each deterministic for a given seed: one sentence per intent
    first, then hard medical words until the quota is met, then a random fill.
    Pools are sorted by id before any shuffle so the result does not depend on
    the order rows happen to sit in the CSV.
    """
    grouped = _by_intent(sentences)
    intents = sorted(grouped)
    if total < len(intents):
        raise ValueError(
            f"cannot cover {len(intents)} intents with only {total} sentences"
        )

    chosen: list[Sentence] = []
    taken: set[str] = set()

    for intent in intents:
        pick = rng.choice(grouped[intent])
        chosen.append(pick)
        taken.add(pick.id)

    hard_pool = sorted(
        (s for s in sentences if s.hard and s.id not in taken), key=lambda s: s.id
    )
    rng.shuffle(hard_pool)
    while sum(1 for s in chosen if s.hard) < min_hard:
        if not hard_pool:
            raise ValueError(
                f"only {sum(1 for s in chosen if s.hard)} sentences with a hard "
                f"medical word were available, {min_hard} required"
            )
        pick = hard_pool.pop()
        chosen.append(pick)
        taken.add(pick.id)

    # Fill the rest evenly: always take from the intent that has the fewest
    # sentences so far, so a 20-clip set is roughly three per intent instead of
    # seven of one and one of another.
    remaining: dict[str, list[Sentence]] = {}
    for sentence in sentences:
        if sentence.id not in taken:
            remaining.setdefault(sentence.intent, []).append(sentence)
    for pool in remaining.values():
        pool.sort(key=lambda s: s.id)
        rng.shuffle(pool)

    while len(chosen) < total:
        available = [intent for intent in intents if remaining.get(intent)]
        if not available:
            raise ValueError(
                f"only {len(chosen)} sentences available, {total} requested"
            )
        counts = {
            intent: sum(1 for s in chosen if s.intent == intent)
            for intent in available
        }
        thinnest = min(available, key=lambda intent: (counts[intent], intent))
        pick = remaining[thinnest].pop()
        chosen.append(pick)
        taken.add(pick.id)

    return chosen[:total]


def split_sources(
    chosen: list[Sentence],
    rng: random.Random,
    n_tts: int = DEFAULT_TTS,
) -> list[Clip]:
    """Split the chosen sentences into TTS clips and clips to record.

    The sentences are dealt out one intent at a time, so the first ``n_tts`` of
    the resulting order hold every intent that fits — the TTS half is never all
    one intent, whatever the seed.
    """
    buckets = _by_intent(chosen)
    for pool in buckets.values():
        rng.shuffle(pool)

    order: list[Sentence] = []
    intents = sorted(buckets)
    while any(buckets[intent] for intent in intents):
        for intent in intents:
            if buckets[intent]:
                order.append(buckets[intent].pop())

    clips: list[Clip] = []
    for index, sentence in enumerate(order, start=1):
        source = "tts" if index <= n_tts else "recorded"
        clip_id = f"{sentence.lang}-{index:03d}"
        clips.append(
            Clip(
                clip_id=clip_id,
                sentence=sentence,
                source=source,
                file=f"data/audio/{SOURCE_DIRS[source]}/{clip_id}.wav",
            )
        )
    return clips


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------

def write_manifest(clips: list[Clip], path: Path) -> None:
    """Write the manifest, creating `data/audio` and its subdirectories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for directory in SOURCE_DIRS.values():
        (path.parent / directory).mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for clip in clips:
            writer.writerow(clip.as_row())


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def report_coverage(clips: list[Clip]) -> None:
    """Print how the set covers intents and hard medical words."""
    print(f"\n{len(clips)} sentences picked — coverage:")
    header = f"{'intent':<22}{'total':>7}{'tts':>6}{'record':>8}{'hard':>6}"
    print(header)
    print("-" * len(header))
    for intent in sorted({c.sentence.intent for c in clips}):
        group = [c for c in clips if c.sentence.intent == intent]
        print(
            f"{intent:<22}{len(group):>7}"
            f"{sum(1 for c in group if c.source == 'tts'):>6}"
            f"{sum(1 for c in group if c.source == 'recorded'):>8}"
            f"{sum(1 for c in group if c.sentence.hard):>6}"
        )
    print("-" * len(header))

    hard = [c for c in clips if c.sentence.hard]
    print(f"{len(hard)} sentences with a hard medical word (minimum {MIN_HARD_WORDS}):")
    for clip in hard:
        words = ", ".join(hard_words_in(clip.sentence.text))
        print(f"  {clip.clip_id}  [{words}]")


def print_recording_sheet(clips: list[Clip]) -> None:
    """Print the sentences a human has to record, with the file name for each."""
    to_record = [c for c in clips if c.source == "recorded"]
    print(f"\n{'=' * 78}")
    print(f"RECORD THESE {len(to_record)} SENTENCES — 16 kHz mono WAV, one file each")
    print(f"{'=' * 78}")
    print("Read each line out loud at a normal pace, then save it under the name")
    print("given above it. Leave about half a second of silence at both ends.\n")
    for clip in to_record:
        words = hard_words_in(clip.sentence.text)
        marker = f"   (careful: {', '.join(words)})" if words else ""
        print(f"{clip.file}{marker}")
        print(f"    [{clip.sentence.intent}]  {clip.sentence.text}\n")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the corpus path, output directory, seed and split sizes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--intents", type=Path, default=DEFAULT_INTENTS_CSV,
                        help="the intent corpus CSV to pick from")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR,
                        help="where manifest.csv and the clip folders go")
    parser.add_argument("--lang", type=str, default=LANGUAGE,
                        help="which language to pick (fr for now)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="fixed seed, so the same sentences come back")
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL,
                        help="how many sentences in total")
    parser.add_argument("--tts", type=int, default=DEFAULT_TTS,
                        help="how many of them are synthesised rather than recorded")
    parser.add_argument("--min-hard", type=int, default=MIN_HARD_WORDS,
                        help="minimum sentences containing a hard medical word")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Pick the sentences, write the manifest and print the recording sheet."""
    args = parse_args(argv)

    if not args.intents.exists():
        print(f"intent corpus not found: {args.intents}", file=sys.stderr)
        return 1
    if args.tts > args.total:
        print(f"--tts {args.tts} exceeds --total {args.total}", file=sys.stderr)
        return 1

    sentences = load_sentences(args.intents, args.lang)
    print(f"{len(sentences)} {args.lang!r} sentences in {args.intents.name}")
    if len(sentences) < args.total:
        print(
            f"only {len(sentences)} available, {args.total} requested",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(args.seed)
    try:
        chosen = select(sentences, rng, total=args.total, min_hard=args.min_hard)
    except ValueError as exc:
        print(f"selection failed: {exc}", file=sys.stderr)
        return 1

    clips = split_sources(chosen, rng, n_tts=args.tts)

    manifest = args.audio_dir / "manifest.csv"
    write_manifest(clips, manifest)

    report_coverage(clips)
    print(f"\nmanifest written: {manifest}")
    print(f"  {sum(1 for c in clips if c.source == 'tts')} to synthesise  "
          f"(python scripts/make_tts_clips.py)")
    print(f"  {sum(1 for c in clips if c.source == 'recorded')} to record  (below)")
    print_recording_sheet(clips)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
