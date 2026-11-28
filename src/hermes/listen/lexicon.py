"""Repair the domain words Whisper mangles, without touching the sentence.

Whisper on CPU writes "broches" as "brosses", "échographie" as "écographie"
and "à l'iode" as "alleraute". Three things downstream break on that: the
safety rules stop matching, the classifier embeds a nonsense word, and FAQ
retrieval loses the term it needed.

Fuzzy-matching whole sentences would fix the words and destroy the meaning —
"broches **dans le bras**" is an implant and "**retirer** mes bijoux" is
preparation, and those differ only in structure. So the fuzziness lives at the
word level and the sentence is rebuilt exactly as it was, punctuation and all.

**The two lexicons are treated differently, because the risks are not
symmetric.**

- *Domain* terms (exam names, modalities, body parts): a false repair sends a
  caller to the wrong exam. Conservative — a token is replaced only when it is
  not a real French word and is very close to a catalogue term.
- *Safety* terms (implants, allergies, pregnancy, contrast): a false flag costs
  one transfer; a miss can let an implant question reach the booking branch.
  Aggressive, and **never rewrites** — it only raises a flag. A real French
  word can be flagged, which is the whole point: "brosses" is French, and in
  "des brosses en métal dans le bras" it is almost certainly "broches".

Nothing here is ever used to score WER. The word error rate measures the
transcriber; measuring it on repaired text would measure this module instead
and quietly flatter the numbers.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from hermes.config import get_settings

#: Tokens are words plus the characters French glues onto them.
_TOKEN = re.compile(r"[0-9A-Za-zÀ-ÿ]+(?:['’][0-9A-Za-zÀ-ÿ]+)*", re.UNICODE)

#: Elisions stripped before matching: "l'iode" should match "iode".
_ELISION = re.compile(r"^(?:l|d|j|m|t|s|n|c|qu)['’]", re.IGNORECASE)


@dataclass(frozen=True)
class Edit:
    """One token the repair rewrote."""

    position: int
    original: str
    replacement: str
    score: float
    rule: str

    def __str__(self) -> str:
        return f"{self.original!r} -> {self.replacement!r} ({self.rule} {self.score:.2f})"


@dataclass(frozen=True)
class SafetyFlag:
    """A safety term the transcript probably contains, however it was spelled."""

    term: str
    matched: str
    position: int
    score: float
    source: str

    def __str__(self) -> str:
        return f"{self.term} ~ {self.matched!r} ({self.source} {self.score:.2f})"


@dataclass(frozen=True)
class RepairResult:
    """What the repair did to one transcript."""

    raw: str
    repaired: str
    edits: tuple[Edit, ...] = ()
    safety_flags: tuple[SafetyFlag, ...] = ()

    @property
    def changed(self) -> bool:
        """Whether any token was rewritten."""
        return bool(self.edits)

    @property
    def flagged_terms(self) -> tuple[str, ...]:
        """The distinct safety terms raised, in order."""
        seen: list[str] = []
        for flag in self.safety_flags:
            if flag.term not in seen:
                seen.append(flag.term)
        return tuple(seen)


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

def fold(text: str) -> str:
    """Lower-case and strip diacritics, for matching only."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def strip_elision(token: str) -> str:
    """Drop a leading elided article: "l'iode" -> "iode"."""
    return _ELISION.sub("", token)


@lru_cache(maxsize=8192)
def is_french_word(token: str, minimum: float | None = None) -> bool:
    """Whether a token is a real French word common enough to leave alone.

    Frequency rather than mere attestation: "écographie" exists in the corpus
    wordfreq was built from, but at Zipf 1.20 against "échographie"'s 3.25 it
    is a misspelling that got written down, not a word a caller said.
    """
    settings = get_settings()
    floor = settings.french_word_min_zipf if minimum is None else minimum
    try:
        from wordfreq import zipf_frequency
    except ImportError:  # pragma: no cover - wordfreq is a dependency
        # Without the frequency list, refuse to repair anything rather than
        # repair wrongly: every token looks like a real word.
        return True
    return zipf_frequency(token, "fr") >= floor


# --------------------------------------------------------------------------
# The lexicon file
# --------------------------------------------------------------------------

@dataclass
class Lexicon:
    """The built vocabulary: canonical terms and their known variants."""

    domain: dict[str, list[str]] = field(default_factory=dict)
    safety: dict[str, list[str]] = field(default_factory=dict)
    #: Exact multi-word repairs, applied before token matching.
    phrases: dict[str, str] = field(default_factory=dict)
    variant_sources: dict[str, str] = field(default_factory=dict)

    @property
    def domain_terms(self) -> list[str]:
        return list(self.domain)

    @property
    def safety_terms(self) -> list[str]:
        return list(self.safety)

    def variants_of(self, term: str) -> list[str]:
        return self.domain.get(term, self.safety.get(term, []))

    def source_of(self, variant: str) -> str:
        """Where a known variant came from: whisper, llm, or generated."""
        return self.variant_sources.get(fold(variant), "lexicon")


_LEXICON: dict[str, Lexicon] = {}


def load_lexicon(path: str | Path | None = None) -> Lexicon:
    """Load and cache the built lexicon."""
    settings = get_settings()
    target = Path(path) if path else settings.paths.processed / settings.lexicon_file
    key = str(target)

    if key not in _LEXICON:
        if not target.exists():
            raise RuntimeError(
                f"no lexicon at {target}. Build it with:\n"
                "    uv run python scripts/build_lexicon.py"
            )
        data = json.loads(target.read_text(encoding="utf-8"))
        _LEXICON[key] = Lexicon(
            domain={k: list(v) for k, v in data.get("domain", {}).items()},
            safety={k: list(v) for k, v in data.get("safety", {}).items()},
            phrases=dict(data.get("phrases", {})),
            variant_sources={
                fold(k): v for k, v in data.get("variant_sources", {}).items()
            },
        )
    return _LEXICON[key]


def reset_lexicon() -> None:
    """Drop the cached lexicon, so a rebuild is picked up."""
    _LEXICON.clear()
    is_french_word.cache_clear()


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def _best_match(
    token: str,
    terms: dict[str, list[str]],
    minimum: float,
) -> tuple[str, float, str] | None:
    """Return (canonical term, score, rule) for the closest entry, or None.

    An exact hit on a *known* variant beats fuzzy matching: those were observed
    coming out of Whisper, so they are evidence rather than a guess.
    """
    from rapidfuzz import fuzz

    folded = fold(strip_elision(token))
    if not folded:
        return None

    for term, variants in terms.items():
        if folded == fold(term):
            return term, 1.0, "exact"
        for variant in variants:
            if folded == fold(variant):
                return term, 1.0, "known_variant"

    best: tuple[str, float, str] | None = None
    for term, variants in terms.items():
        for candidate in (term, *variants):
            score = fuzz.ratio(folded, fold(candidate)) / 100.0
            if score >= minimum and (best is None or score > best[1]):
                best = (term, score, "fuzzy")
    return best


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------

#: Words that make a sentence about medical imaging. A loose safety match is
#: only believed when one of these is present: "brosses" beside "métal" and
#: "bras" is almost certainly "broches", while "brioche" beside "café" is
#: breakfast.
#:
#: This context gate is not a refinement, it is what makes safety flagging
#: possible at all. Measured on the decisive pairs, no single similarity
#: threshold works — "brosses ~ broches" scores 0.71 while "brioches ~
#: broches" scores 0.93, so the character signal is inverted exactly where it
#: matters.
CONTEXT_PATTERNS: tuple[str, ...] = (
    r"\bm[ée]tal\w*\b",
    r"\b(?:irm|scanner|radio|échographie|écographie|mammographie|examen)\b",
    r"\bdans\s+(?:la|le|les|mon|ma|mes)\s+"
    r"(?:bras|tête|jambe|corps|genou|épaule|dos|hanche|crâne|cou|sein)\b",
    r"\b(?:opér\w+|chirurg\w+|pose\w*)\b",
    # "centre" is deliberately absent: it appears in almost every sentence
    # about this clinic, so it made the context gate fire everywhere — and it
    # then fuzzy-matched "enceinte" at 0.71 in "Le plan du centre".
    r"\b(?:rendez[\s-]vous|médecin|radiologue|docteur)\b",
    r"\b(?:injection|contraste|produit)\b",
    r"\b(?:enceinte|grossesse|allait\w*)\b",
)


def has_imaging_context(text: str) -> bool:
    """Whether the sentence is about the centre's business at all."""
    folded = fold(text)
    return any(re.search(pattern, folded) for pattern in CONTEXT_PATTERNS)


def _flaggable(token: str, rule: str) -> bool:
    """Whether a fuzzy match on this token is worth believing.

    An everyday word is not anyone's mangled safety term. "plan" matched
    "implant" at 0.73 and "centre" matched "enceinte" at 0.71, both inside an
    ordinary sentence about the building. A known variant is evidence and is
    always believed; a fuzzy match on a common word is not.
    """
    if rule != "fuzzy":
        return True
    settings = get_settings()
    try:
        from wordfreq import zipf_frequency
    except ImportError:  # pragma: no cover - wordfreq is a dependency
        return True
    bare = fold(strip_elision(token))
    return zipf_frequency(bare, "fr") < settings.safety_flag_max_zipf


def repair(
    text: str,
    lexicon: Lexicon | None = None,
    repair_ratio: float | None = None,
    flag_ratio: float | None = None,
    context_ratio: float | None = None,
) -> RepairResult:
    """Repair domain words and flag safety words, keeping the sentence intact.

    Returns the original text, the repaired text, every edit made and every
    safety term suspected. The sentence is rebuilt from the original string
    with only the matched spans substituted, so punctuation, spacing and case
    around the token survive untouched.
    """
    settings = get_settings()
    book = lexicon or load_lexicon()
    repair_floor = settings.repair_min_ratio if repair_ratio is None else repair_ratio
    flag_floor = settings.safety_flag_min_ratio if flag_ratio is None else flag_ratio
    context_floor = (
        settings.safety_flag_context_ratio if context_ratio is None else context_ratio
    )

    raw = text or ""
    if not raw.strip():
        return RepairResult(raw=raw, repaired=raw)

    edits: list[Edit] = []

    # Phrases first. A phrase can repair what a token never could: "jeun" is
    # heard as "jeu", and "jeu" is too common a word for the guard to let it
    # be rewritten — but "à jeu" in a call to an imaging centre is not a game.
    # Done before tokenising so the token pass sees the corrected sentence.
    working = _apply_phrases(raw, book.phrases, edits)

    # The looser bar applies only when the sentence is about imaging at all.
    in_context = has_imaging_context(working)
    effective_flag_floor = min(flag_floor, context_floor) if in_context else flag_floor

    flags: list[SafetyFlag] = []
    pieces: list[str] = []
    cursor = 0

    for match in _TOKEN.finditer(working):
        token = match.group(0)
        start, end = match.span()

        # --- safety: flag, never rewrite -------------------------------
        hit = _best_match(token, book.safety, effective_flag_floor)
        if hit is not None and _flaggable(token, hit[2]):
            term, score, rule = hit
            source = book.source_of(token) if rule == "known_variant" else rule
            if rule == "fuzzy" and score < flag_floor:
                source = "fuzzy+context"
            flags.append(
                SafetyFlag(
                    term=term,
                    matched=token,
                    position=start,
                    score=score,
                    source=source,
                )
            )

        # --- domain: rewrite, but only when it is safe to --------------
        replacement = None
        domain_hit = _best_match(token, book.domain, repair_floor)
        if domain_hit is not None:
            term, score, rule = domain_hit
            bare = strip_elision(token)
            already_right = fold(bare) == fold(term)
            # A real French word is left alone however close it looks: that is
            # what stops "brioche" becoming "broche".
            allowed = rule == "known_variant" or not is_french_word(fold(bare))
            if not already_right and allowed:
                prefix = token[: len(token) - len(bare)]
                replacement = prefix + _match_case(bare, term)
                edits.append(
                    Edit(
                        position=start,
                        original=token,
                        replacement=replacement,
                        score=score,
                        rule=rule,
                    )
                )

        if replacement is not None:
            pieces.append(working[cursor:start])
            pieces.append(replacement)
            cursor = end

    pieces.append(working[cursor:])

    flags.extend(
        _multiword_flags(working, book, effective_flag_floor, flag_floor)
    )

    return RepairResult(
        raw=raw,
        repaired="".join(pieces),
        edits=tuple(edits),
        safety_flags=tuple(flags),
    )


#: Longest safety term, in words. Windows up to this size are checked.
_MAX_TERM_WORDS = 3


def _multiword_flags(
    text: str,
    book: Lexicon,
    effective_floor: float,
    standalone_floor: float,
) -> list[SafetyFlag]:
    """Flag safety terms that span several words.

    Token-by-token matching structurally cannot see these: "cardiake" alone is
    nowhere near "stimulateur cardiaque", so the whole term was invisible.
    Measured — "stimulateur cardiake" and "produit de contrasste" were the
    only two safety cases the token pass missed, and both are two words.
    """
    from rapidfuzz import fuzz

    multi = {
        term: variants
        for term, variants in book.safety.items()
        if len(term.split()) > 1
    }
    if not multi:
        return []

    spans = [(m.group(0), m.start()) for m in _TOKEN.finditer(text)]
    found: list[SafetyFlag] = []
    claimed: set[int] = set()

    for size in range(_MAX_TERM_WORDS, 1, -1):
        for index in range(len(spans) - size + 1):
            if any(i in claimed for i in range(index, index + size)):
                continue
            window = " ".join(w for w, _ in spans[index : index + size])
            folded = fold(window)

            for term, variants in multi.items():
                best = max(
                    fuzz.ratio(folded, fold(candidate)) / 100.0
                    for candidate in (term, *variants)
                )
                if best < effective_floor:
                    continue
                source = "fuzzy" if best >= standalone_floor else "fuzzy+context"
                found.append(
                    SafetyFlag(
                        term=term,
                        matched=window,
                        position=spans[index][1],
                        score=best,
                        source=source,
                    )
                )
                claimed.update(range(index, index + size))
                break
    return found


def _apply_phrases(
    text: str,
    phrases: dict[str, str],
    edits: list[Edit],
) -> str:
    """Apply exact multi-word repairs, longest first.

    Longest first so "être à jeu" wins over "à jeu" and the edit recorded is
    the one that actually fired. Matching ignores case and accents, but the
    replacement is written as the lexicon spells it — the whole point is to
    restore the accents Whisper dropped.
    """
    if not phrases:
        return text

    working = text
    for phrase in sorted(phrases, key=len, reverse=True):
        replacement = phrases[phrase]
        # Build an accent-insensitive pattern from the folded phrase: each
        # letter matches itself or any accented form of itself.
        pattern = r"\b" + r"\s+".join(
            "".join(_ACCENT_CLASS.get(c, re.escape(c)) for c in word)
            for word in fold(phrase).split()
        ) + r"\b"

        for match in list(re.finditer(pattern, working, flags=re.IGNORECASE)):
            found = match.group(0)
            if fold(found) == fold(replacement):
                continue
            edits.append(
                Edit(
                    position=match.start(),
                    original=found,
                    replacement=_match_case(found, replacement),
                    score=1.0,
                    rule="phrase",
                )
            )
            working = (
                working[: match.start()]
                + _match_case(found, replacement)
                + working[match.end():]
            )
            break  # positions shifted; the next pass re-scans
    return working


#: Letters that French accents the same letter, so a folded phrase pattern
#: still matches the accented text it came from.
_ACCENT_CLASS: dict[str, str] = {
    "a": "[aàâä]", "e": "[eéèêë]", "i": "[iîï]", "o": "[oôö]",
    "u": "[uùûü]", "c": "[cç]",
}


def _match_case(original: str, replacement: str) -> str:
    """Carry the original token's capitalisation onto the replacement."""
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def repaired_text(text: str) -> str:
    """Shorthand for callers that only want the corrected sentence."""
    return repair(text).repaired


def safety_terms_in(text: str) -> tuple[str, ...]:
    """Shorthand for callers that only want the flagged terms."""
    return repair(text).flagged_terms


# --------------------------------------------------------------------------
# What the agent asks when a flag is raised
# --------------------------------------------------------------------------

#: How to name each safety term when confirming it aloud. A flag is a
#: suspicion, not a fact: the transcript said "brosses", and asking is how the
#: agent finds out whether that was "broches" without assuming either way.
CONFIRMATIONS: dict[str, str] = {
    "broches": "des broches, c'est-à-dire un implant métallique",
    "clips": "des clips métalliques",
    "implant": "un implant",
    "prothèse": "une prothèse",
    "pacemaker": "un pacemaker",
    "stimulateur cardiaque": "un stimulateur cardiaque",
    "iode": "l'iode",
    "produit de contraste": "le produit de contraste",
    "anticoagulant": "un anticoagulant",
    "allergie": "une allergie",
    "enceinte": "une grossesse",
}


def confirmation_question(flags: tuple[SafetyFlag, ...]) -> str:
    """Return the line that checks a flag aloud, or "" when there is nothing.

    Spoken before acting, never instead of transferring: if the caller
    confirms, the turn goes to a human anyway. The question exists so the
    transfer is explained rather than abrupt, and so a mishearing does not
    silently become a fact.
    """
    if not flags:
        return ""
    term = flags[0].term
    named = CONFIRMATIONS.get(term, term)
    # "mentionné" rather than "parlé de": the named forms carry their own
    # article ("des broches"), and "parlé de des broches" is what the other
    # phrasing produces.
    return f"Pour être sûr de bien comprendre : vous avez mentionné {named} ?"
