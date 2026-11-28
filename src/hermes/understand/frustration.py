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


#: What `score()` can conclude. `none` is the normal case.
LEVELS = ("none", "soft", "hard")


@dataclass(frozen=True)
class FrustrationScore:
    """How frustrated the caller sounds on this turn.

    `level` is the only field the orchestrator should branch on. `prob` is
    kept for the bench and the logs, and `matched_terms` is kept because a
    flag a human cannot explain is a flag nobody will trust.

    Two levels, two actions, decided in WP6 and not here:
    `soft` → apologise and simplify, `hard` → hand to a human.
    """

    prob: float
    level: str
    #: Where the verdict came from: "model", "model+lexicon",
    #: "unsupported_lang", "no_model", "empty", or "error:<Type>". Never
    #: silently absent.
    source: str
    lexicon_hits: tuple[str, ...] = ()

    @property
    def is_frustrated(self) -> bool:
        """Either level. Kept because `escalation` and `agent` annotate on it."""
        return self.level in ("soft", "hard")

    @property
    def needs_human(self) -> bool:
        """Whether *this signal alone* calls for a handover.

        One of three paths to a transfer, not the only one: two consecutive
        soft hits also transfer, and so does the WP6/WP7 repeat counter. A
        `none` here does not mean the call stays with the bot.
        """
        return self.level == "hard"


def neutral(source: str, prob: float = 0.0) -> FrustrationScore:
    """The no-opinion result. Scoring degrades to this rather than raising."""
    return FrustrationScore(prob=prob, level="none", source=source)


def level_for(
    probability: float,
    has_lexicon_hit: bool,
    soft: float,
    hard: float,
) -> str:
    """The shipped decision rule, in one place so the sweep and the module agree.

        soft  if  prob >= soft
        hard  if  prob >= hard  OR  (lexicon hit AND prob >= soft)

    The lexicon arm exists because the model's misses are not marginal. It
    scores "Bon, je répète, le genou DROIT !!" at 0.116 — explicit, twice
    exclaimed, and all but invisible to the embedding — while the word list
    catches it outright at a measured precision of one false alarm in 180
    rows. Requiring `prob >= soft` on that arm keeps the word list from
    acting alone on a sentence the model actively disagrees with.
    """
    if probability >= hard or (has_lexicon_hit and probability >= soft):
        return "hard"
    if probability >= soft:
        return "soft"
    return "none"


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
    #
    # **First person only.** The bare infinitive and the imperative are not
    # complaints at all — "vous pouvez répéter s'il vous plaît ?" is a polite
    # request and is labelled `correction`. Step 1 measured that as the word
    # list's single false alarm in 180 rows, and left both forms in the same
    # family regardless; keeping them cost the hybrid its entire false-alarm
    # budget, because the row it wrongly flags is a `correction` and the hard
    # threshold is already spending that budget to clear "Non non, c'est bon".
    #
    # So the family is now defined by *who* is repeating. A caller repeating
    # themselves is frustrated with the system. A caller asking the system to
    # repeat is being helpful.
    "repetition": (
        "je repete",
        "je le repete",
        "je me repete",
        "je l ai deja dit",
        "je vous l ai deja dit",
        "deja dit",
        "je vous ai dit",
        "ca fait deux fois",
        "ca fait trois fois",
        "combien de fois",
        "je redis",
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
        "j en peux plus",
        "je n en peux plus",
        "insupportable",
        "inadmissible",
        "inacceptable",
        "scandaleux",
        "c est penible",
        "c est lourd",
        "ca devient penible",
    ),
    # Anger named directly. Added as a family rather than as the two words
    # that happened to be missed on a probe: the definition of label 1 is
    # annoyance *with the system*, and saying so outright is the most
    # literal expression of it that the corpus barely contains.
    "angry": (
        "ca m enerve",
        "ca m agace",
        "m enerve",
        "m agace",
        "enervant",
        "agacant",
        "ca m irrite",
        "je suis enerve",
        "je suis enervee",
        "je suis en colere",
        "vous m enervez",
    ),
    # Threatening to leave. The last thing a caller says before hanging up,
    # and the clearest signal that a handover is already overdue.
    "leaving": (
        "je raccroche",
        "je vais raccrocher",
        "laissez tomber",
        "j abandonne",
        "tant pis",
        "je rappellerai",
        "j irai ailleurs",
        "je vais me plaindre",
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


# --------------------------------------------------------------------------
# The trained model
# --------------------------------------------------------------------------

#: Loaded once per process. `None` means not yet attempted; a failed attempt
#: is remembered as a reason string so a missing file is reported once rather
#: than retried on every turn of every call.
_MODEL: object | None = None
_META: dict | None = None
_FAILURE: str | None = None


def load_model(path: Path | None = None, reload: bool = False):
    """Load the ensemble, or return None with the reason recorded.

    **Refuses a model trained with a different embedder.** The weights are
    384 numbers wide either way, so a mismatched encoder loads cleanly and
    scores nonsense — plausible-looking probabilities from the wrong vector
    space. That is a worse failure than an exception, so it is checked.
    """
    global _MODEL, _META, _FAILURE

    if _MODEL is not None and not reload:
        return _MODEL
    if _FAILURE is not None and not reload:
        return None

    from hermes.config import get_settings

    settings = get_settings()
    source = Path(path) if path else settings.paths.models / settings.frustration_model_file

    if not source.is_file():
        _FAILURE = "no_model"
        return None

    try:
        from hermes.understand.mlp import TorchMLP

        model, meta = TorchMLP.load(source, width=settings.embedding_dim)
    except Exception as exc:  # noqa: BLE001 — a bad artefact must not kill a call
        _FAILURE = f"error:{type(exc).__name__}"
        return None

    trained_with = str(meta.get("embedder", ""))
    if trained_with and trained_with != settings.embedding_model:
        _FAILURE = "embedder_mismatch"
        print(f"  ! frustration model was trained with {trained_with!r}, "
              f"config says {settings.embedding_model!r} — refusing to load")
        return None

    _MODEL, _META, _FAILURE = model, meta, None
    return _MODEL


def model_metadata() -> dict:
    """What the loaded artefact says about itself."""
    load_model()
    return dict(_META or {})


def reset_model() -> None:
    """Forget the cached model. For tests and for a retrain mid-session."""
    global _MODEL, _META, _FAILURE
    _MODEL, _META, _FAILURE = None, None, None


# --------------------------------------------------------------------------
# Scoring one turn
# --------------------------------------------------------------------------

def detect(text: str, lang: str = "fr") -> FrustrationScore:
    """Score one caller turn. CPU only, model loaded once, never raises.

    French only. Anything else returns level `none` with source
    `unsupported_lang` rather than a number, because the model was trained
    and calibrated on 180 French rows and its thresholds mean nothing
    outside that. Measured reason for the scope, in `docs/WP5.md`: 53 of the
    56 `mix` corpus rows carry their frustration in Darija, so a French
    reading of them has almost nothing to work from.
    """
    if not (text or "").strip():
        return neutral("empty")
    if (lang or "").lower() != "fr":
        return neutral("unsupported_lang")

    terms = lexicon_hits(text)
    model = load_model()
    if model is None:
        return FrustrationScore(
            prob=0.0, level="none", source=_FAILURE or "no_model",
            lexicon_hits=terms,
        )

    try:
        from hermes.infra import embeddings

        vector = embeddings.encode_one(text).reshape(1, -1)
        probability = float(model.predict_proba(vector)[0, 1])
    except Exception as exc:  # noqa: BLE001 — a scoring failure is not fatal
        return FrustrationScore(
            prob=0.0, level="none", source=f"error:{type(exc).__name__}",
            lexicon_hits=terms,
        )

    from hermes.config import get_settings

    settings = get_settings()
    level = level_for(
        probability,
        bool(terms),
        settings.frustration_soft_threshold,
        settings.frustration_hard_threshold,
    )
    lifted = (
        level == "hard" and probability < settings.frustration_hard_threshold
    )
    return FrustrationScore(
        prob=probability,
        level=level,
        source="model+lexicon" if lifted else "model",
        lexicon_hits=terms,
    )


#: The name this module used before the level logic moved in. Kept so an
#: older caller does not break; `detect` is the one to use.
score = detect
