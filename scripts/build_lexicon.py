"""Build the repair lexicon: domain and safety terms, with their known variants.

Writes `data/processed/lexicon.json`. Two vocabularies, built differently
because they are used differently:

- **domain** — exam names, modalities and body regions from `exams.csv`. Used
  to *rewrite* a mangled token, so it is conservative.
- **safety** — implants, allergies, pregnancy, contrast. Used to *flag*, never
  to rewrite, so it can afford to be generous.

Variants come from two places, and each is recorded with its source so the
report can tell evidence from conjecture:

1. **whisper** — the term is spoken by Piper inside a short French sentence,
   transcribed by Whisper, and whatever comes back is recorded. These are
   observed failures on this exact model and machine, not guesses.
2. **llm** — one Gemini call per batch, at build time only, asking for likely
   French mistranscriptions. Cached to the lexicon; nothing calls an LLM at
   run time.

    python scripts/build_lexicon.py
    python scripts/build_lexicon.py --no-whisper --no-llm   # terms only
    python scripts/build_lexicon.py --sentences 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hermes.config import get_settings  # noqa: E402

#: The safety vocabulary. Mirrors hermes.stt.SAFETY_VOCABULARY.
SAFETY_TERMS: list[str] = [
    "broches",
    "prothèse",
    "stimulateur cardiaque",
    "pacemaker",
    "implant",
    "clips",
    "iode",
    "produit de contraste",
    "anticoagulant",
    "allergie",
    "enceinte",
    # Added after the end-to-end measurement: "Est-ce que je peux arrêter mon
    # traitemen avant le scanner ?" was the single turn in 85 that reached the
    # booking branch. The regex needs "traitement" spelled correctly; the
    # lexicon does not.
    "traitement",
    "médicament",
]

#: Multi-word repairs, applied as exact phrase lookups before anything else.
#:
#: These exist for the case the token-level repair cannot reach: "jeun" is
#: heard as "jeu", and "jeu" is a common French word (Zipf 5.49) that the
#: guard protecting "brioche" must also protect. The phrase is the way out —
#: "à jeu" in a call to an imaging centre is never about a game, while "jeu"
#: alone very well might be.
#:
#: "Faut-il être à jeun ?" is among the most asked questions on this line, and
#: without this it was being misrouted to medical_question.
PHRASE_REPAIRS: dict[str, str] = {
    "à jeu": "à jeun",
    "a jeu": "à jeun",
    "être à jeu": "être à jeun",
    "etre a jeu": "être à jeun",
    "rester à jeu": "rester à jeun",
    "venir à jeu": "venir à jeun",
    "à jeune": "à jeun",
    "a jeune": "à jeun",
    "au jeun": "à jeun",
}

#: Modality names, always in the domain lexicon whatever the catalogue holds.
CORE_DOMAIN: list[str] = [
    "IRM",
    "scanner",
    "échographie",
    "radiographie",
    "mammographie",
    "densitométrie",
    "arthroscanner",
    "uroscanner",
    "entéroscanner",
    "coloscanner",
    "télérachis",
    "rachis",
    "doppler",
    "gadolinium",
]

#: Carrier sentences. Short, so the transcription failure is the term itself
#: rather than the model losing the thread of a long sentence.
CARRIERS: list[str] = [
    "J'ai {term}, est-ce que c'est possible ?",
    "Le médecin a parlé de {term}.",
    "C'est pour {term}.",
]

#: Words too short or too common to be worth matching against.
_MIN_LENGTH = 5
_STOP = {
    "avec", "sans", "pour", "dans", "droit", "droite", "gauche", "simple",
    "complet", "controle", "contrôle", "bilateral", "bilatéral", "des", "les",
}


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


# --------------------------------------------------------------------------
# The domain vocabulary
# --------------------------------------------------------------------------

def domain_terms(exams_csv: Path) -> list[str]:
    """Mine exam names, modalities and regions from the catalogue."""
    terms: dict[str, None] = {t: None for t in CORE_DOMAIN}

    if not exams_csv.exists():
        return list(terms)

    with open(exams_csv, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for field_name in ("modality", "region_fr", "name_fr"):
                for word in re.split(r"[^\wÀ-ÿ]+", row.get(field_name) or ""):
                    folded = fold(word)
                    if len(folded) < _MIN_LENGTH or folded in _STOP:
                        continue
                    if folded.isdigit():
                        continue
                    if not any(fold(t) == folded for t in terms):
                        terms[word] = None
    return list(terms)


# --------------------------------------------------------------------------
# Variants observed from Whisper
# --------------------------------------------------------------------------

def whisper_variants(terms: list[str], sentences: int) -> dict[str, set[str]]:
    """Speak each term, transcribe it, and record what came back.

    The only step here that is evidence rather than conjecture: these are the
    words this model produces for these terms on this machine.
    """
    from hermes import stt, tts

    found: dict[str, set[str]] = {term: set() for term in terms}
    engine = stt.load_model()

    for index, term in enumerate(terms, start=1):
        print(f"  [{index:>3}/{len(terms)}] {term}", flush=True)
        for carrier in CARRIERS[:sentences]:
            sentence = carrier.format(term=term)
            try:
                samples, rate = tts.synthesise_samples(sentence)
                heard = engine.transcribe_full(samples, rate).text
            except Exception as exc:  # noqa: BLE001 — one term failing is not fatal
                print(f"        skipped: {type(exc).__name__}: {exc}")
                continue

            for word in re.findall(r"[0-9A-Za-zÀ-ÿ']+", heard):
                variant = _acceptable_variant(word, term)
                if variant:
                    found[term].add(variant)
                    print(f"        heard {variant!r}")
    return found


#: Above this Zipf frequency a word is too common to be anyone's
#: mistranscription. Measured need: the first build recorded "plan" as a
#: variant of "implant" at 72% similarity, which would have flagged an implant
#: every time a caller said "un plan".
_MAX_VARIANT_ZIPF = 4.0

#: Raised from 72 after that build: 72 admitted "plan", and every genuine
#: variant observed ("broche", "déchographie", "télérachie") clears 78.
_MIN_VARIANT_RATIO = 78


def _acceptable_variant(word: str, term: str) -> str | None:
    """Return the variant to record for this heard word, or None.

    Three filters, each earning its place from the first build's output:
    carrier bleed ("Géanticoagulant", "J'escaner"), everyday words ("plan"),
    and anything not actually close to the term.
    """
    from rapidfuzz import fuzz

    bare = re.sub(r"^(?:l|d|j|m|t|s|n|qu)['’]", "", word)
    if len(bare) < 4 or "'" in bare or "’" in bare:
        return None
    if fold(bare) == fold(term):
        return None
    if fuzz.ratio(fold(bare), fold(term)) < _MIN_VARIANT_RATIO:
        return None

    try:
        from wordfreq import zipf_frequency

        if zipf_frequency(bare.lower(), "fr") >= _MAX_VARIANT_ZIPF:
            return None
    except ImportError:  # pragma: no cover - wordfreq is a dependency
        pass
    return bare.lower()


# --------------------------------------------------------------------------
# Variants suggested by an LLM
# --------------------------------------------------------------------------

LLM_PROMPT = """Tu aides à construire un correcteur pour un agent vocal d'un \
centre d'imagerie médicale en Tunisie. La transcription est faite par Whisper \
(modèle "small") en français.

Pour chaque terme ci-dessous, donne les erreurs de transcription les plus \
probables : des mots français réels ou des suites de lettres que Whisper \
écrirait à la place parce qu'ils se prononcent presque pareil.

Exemples réels observés : "broches" -> "brosses", "échographie" -> \
"écographie", "à l'iode" -> "alleraute".

Réponds UNIQUEMENT en JSON, sans texte autour :
{{"terme": ["variante1", "variante2"], ...}}

Termes : {terms}
"""


def llm_variants(terms: list[str]) -> dict[str, set[str]]:
    """Ask Gemini for likely mistranscriptions. Build time only, then cached.

    Uses `google-genai`, the current SDK. The older `google-generativeai` is
    deprecated — its own repository is now named
    `deprecated-generative-ai-python` — and this is the project's only LLM
    call so far, so it is the cheapest possible moment to switch.
    """
    from hermes.config import require_gemini_key

    settings = get_settings()
    try:
        key = require_gemini_key(settings)
    except RuntimeError as exc:
        print(f"  no LLM variants: {exc}")
        return {}

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("  no LLM variants: google-genai is not installed.\n"
              "    uv add google-genai")
        return {}

    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=LLM_PROMPT.format(terms=", ".join(terms)),
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        body = (response.text or "").strip()
        body = re.sub(r"^```(?:json)?|```$", "", body, flags=re.MULTILINE).strip()
        data = json.loads(body)
    except Exception as exc:  # noqa: BLE001 — the build works without it
        print(f"  no LLM variants: {type(exc).__name__}: {exc}")
        return {}

    out: dict[str, set[str]] = {}
    for term in terms:
        suggested = data.get(term) or data.get(term.lower()) or []
        kept = {
            str(v).strip().lower()
            for v in suggested
            if isinstance(v, str) and len(str(v).strip()) >= 4
            and fold(str(v)) != fold(term)
        }
        if kept:
            out[term] = kept
            print(f"  {term}: {', '.join(sorted(kept))}")
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the catalogue path, the output, and which variant sources to use."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exams", type=Path,
                        default=settings.paths.raw / "exams.csv")
    parser.add_argument("--out", type=Path,
                        default=settings.paths.processed / settings.lexicon_file)
    parser.add_argument("--sentences", type=int, default=2,
                        help="carrier sentences per term for the Whisper pass")
    parser.add_argument("--no-whisper", action="store_true",
                        help="skip the spoken round-trip (much faster)")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip the Gemini call")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the lexicon and write it."""
    args = parse_args(argv)

    domain = domain_terms(args.exams)
    print(f"{len(domain)} domain terms, {len(SAFETY_TERMS)} safety terms")

    variants: dict[str, set[str]] = {t: set() for t in domain + SAFETY_TERMS}
    sources: dict[str, str] = {}

    # Only the safety terms and the modalities get the expensive treatment:
    # they are the ones whose mangling is dangerous rather than untidy.
    probed = SAFETY_TERMS + CORE_DOMAIN

    if not args.no_whisper:
        print(f"\nspeaking {len(probed)} terms through Piper and Whisper...")
        for term, heard in whisper_variants(probed, args.sentences).items():
            variants.setdefault(term, set()).update(heard)
            for variant in heard:
                sources[variant] = "whisper"

    if not args.no_llm:
        print("\nasking Gemini for likely mistranscriptions...")
        for term, suggested in llm_variants(probed).items():
            fresh = suggested - variants.get(term, set())
            variants.setdefault(term, set()).update(fresh)
            for variant in fresh:
                sources.setdefault(variant, "llm")

    payload = {
        "domain": {t: sorted(variants.get(t, set())) for t in domain},
        "safety": {t: sorted(variants.get(t, set())) for t in SAFETY_TERMS},
        "phrases": dict(PHRASE_REPAIRS),
        "variant_sources": sources,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total = sum(len(v) for v in variants.values())
    by_source: dict[str, int] = {}
    for source in sources.values():
        by_source[source] = by_source.get(source, 0) + 1
    print(f"\nwritten: {args.out}")
    print(f"  {len(domain)} domain terms, {len(SAFETY_TERMS)} safety terms")
    print(f"  {total} variants ("
          + ", ".join(f"{n} {s}" for s, n in sorted(by_source.items())) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
