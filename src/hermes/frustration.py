"""Notice when a caller is getting annoyed with the system, before they say so.

Two things live here, and the first one is not a model.

A **French lexicon baseline**, written from the label's definition — annoyance
*with the system*, not worry about an exam or a complaint about something else.
It exists to be beaten. A classifier that cannot beat a word list has not
earned the 384 dimensions it costs, and without the baseline there is no way to
tell whether the embeddings are contributing anything.

And the trained classifier over sentence embeddings, which arrives at step 2.

**French only.** `data/raw/frustration.csv` holds 476 rows, of which 180 are
`lang == "fr"`. The Darija, Arabizi and mixed rows are not used anywhere in
WP5: measured on the `mix` rows, 53 of 56 carry their frustration in Darija
("yezzi", "fadit mel robot", "ma tefhem chay"), so scoring them in French
would measure the embedder's Darija coverage rather than this feature.

Weights live in `models/` and are gitignored. Scoring degrades to a neutral
result when they are missing rather than raising.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

#: Text-derived signals, kept only as an **ablation** for step 2 — shipped
#: only if they beat the embedding alone on the grouped split.
#:
#: The turn-level signals the first draft of this module declared —
#: `turn_index`, `repeated_question`, `failed_matches`, `audio_retries` —
#: are gone. They are not model features. Four of the six could not be
#: computed from this corpus at all (476 standalone sentences, no dialogue),
#: and three of them need the dialogue state WP6 owns. They belong to the WP6
#: escalation policy, which combines this score with its own counters.
@dataclass(frozen=True)
class TurnFeatures:
    """Signals derivable from the utterance itself, and nothing else."""

    words: int
    exclamations: int


@dataclass(frozen=True)
class FrustrationScore:
    """How frustrated the caller sounds on this turn."""

    score: float
    label: str
    is_frustrated: bool
    contributing: tuple[str, ...]


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def fold(text: str) -> str:
    """Lowercase, strip accents, flatten apostrophes, collapse whitespace.

    Applied to the terms and the utterance alike, so "répète" matches
    "repete" and "j'en ai marre" matches "j en ai marre".
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    spaced = re.sub(r"[^0-9a-z]+", " ", stripped.lower())
    return " ".join(spaced.split())


# --------------------------------------------------------------------------
# The lexicon
# --------------------------------------------------------------------------

#: Written from the label's definition — "frustrated **with the system**" —
#: and deliberately **not** from looking at which rows a model gets wrong.
#: Growing a word list by reading its own errors is how a baseline stops being
#: a baseline: it starts encoding the test set, the comparison in step 2
#: becomes unfair in its favour, and the leak is invisible because it lives in
#: a human's head rather than in a split.
#:
#: Grouped by the way callers express the feeling, which is also how the
#: per-kind error analysis reads.
FRUSTRATION_TERMS: dict[str, tuple[str, ...]] = {
    # Having to say the same thing again is the most common complaint, and the
    # only one the system causes directly.
    "repetition": (
        "je repete",
        "je le repete",
        "repete",
        "repeter",
        "je l ai deja dit",
        "deja dit",
        "je vous ai dit",
        "ca fait deux fois",
        "ca fait trois fois",
        "combien de fois",
        "encore une fois",
    ),
    # Being misunderstood.
    "not understood": (
        "vous ne comprenez",
        "vous comprenez rien",
        "comprenez pas",
        "comprend pas",
        "vous me comprenez pas",
        "ne me comprend",
        "ca ne veut rien dire",
    ),
    # The system judged useless.
    "useless": (
        "n importe quoi",
        "ca ne marche pas",
        "ca marche pas",
        "ne fonctionne pas",
        "fonctionne pas",
        "c est nul",
        "nul",
        "inutile",
        "ca sert a rien",
    ),
    # Time.
    "wasted time": (
        "perdre mon temps",
        "perds mon temps",
        "perdu mon temps",
        "fait perdre mon temps",
        "une perte de temps",
        "j attends depuis",
    ),
    # Exasperation.
    "fed up": (
        "j en ai marre",
        "ras le bol",
        "ca suffit",
        "j en ai assez",
        "insupportable",
        "inadmissible",
        "scandaleux",
        "c est penible",
    ),
    # Asking to leave the bot. The strongest single signal, and the one the
    # feature exists to anticipate rather than wait for.
    "wants a human": (
        "passez moi quelqu un",
        "passez moi un",
        "parler a quelqu un",
        "parler a une personne",
        "une vraie personne",
        "un vrai agent",
        "un humain",
        "une standardiste",
        "l operateur",
        "operateur",
        "un conseiller",
    ),
    # Addressing the machine as a machine is itself a complaint — but only in
    # the first person. "la machine IRM" is an exam, so the bare word is not
    # in the list.
    "talking to a robot": (
        "votre robot",
        "ce robot",
        "un robot",
        "le robot",
        "votre machine",
        "cette machine",
        "repondeur",
        "automate",
    ),
}

#: Flattened, longest first, so "je l ai deja dit" is reported rather than
#: the "deja dit" inside it.
_TERMS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        ((term, family) for family, terms in FRUSTRATION_TERMS.items() for term in terms),
        key=lambda pair: -len(pair[0]),
    )
)


def lexicon_hits(text: str) -> tuple[str, ...]:
    """Which frustration terms appear in this utterance, longest match first.

    Overlapping matches are reported once: a hit consumes its span, so
    "je l ai deja dit" does not also report "deja dit".
    """
    folded = f" {fold(text)} "
    if not folded.strip():
        return ()

    found: list[str] = []
    claimed: list[tuple[int, int]] = []
    for term, _family in _TERMS:
        start = folded.find(f" {term} ")
        if start < 0:
            continue
        span = (start, start + len(term) + 2)
        if any(span[0] < end and start_ < span[1] for start_, end in claimed):
            continue
        claimed.append(span)
        found.append(term)
    return tuple(found)


def lexicon_families(text: str) -> tuple[str, ...]:
    """Which families of frustration the utterance expresses."""
    lookup = dict(_TERMS)
    return tuple(dict.fromkeys(lookup[term] for term in lexicon_hits(text)))


def lexicon_flag(text: str) -> bool:
    """The baseline's prediction: any term at all means frustrated."""
    return bool(lexicon_hits(text))


# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------

def load_dataset(path: str | Path | None = None, lang: str = "fr") -> list[dict[str, str]]:
    """Read the labelled frustration rows for one language.

    Defaults to French, which is WP5's whole scope.
    """
    import csv

    from hermes.config import get_settings

    source = Path(path) if path else get_settings().paths.raw / "frustration.csv"
    with open(source, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if not lang or row["lang"] == lang]


def turn_features(text: str) -> TurnFeatures:
    """The two text-derived signals, for the step-2 ablation."""
    return TurnFeatures(
        words=len(fold(text).split()),
        exclamations=(text or "").count("!"),
    )
