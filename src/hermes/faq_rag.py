"""Retrieval over the centre's FAQ, so answers come from the centre's own text.

The FAQ lives as question/answer rows in `data/raw/faq.csv`. At build time
every phrasing of every question is embedded into the `hermes_faq` collection
by `scripts/build_faq_index.py`; at call time the caller's question is embedded
and the nearest entries come back. The retrieved passages are grounding for
:mod:`hermes.agent` — the agent rephrases them, it does not invent around them.

Returns nothing when no entry clears the acceptance threshold, which is the
signal to escalate rather than to guess. That direction is deliberate: a
confidently wrong answer about fasting or preparation is worse than a transfer.

Measured on the 31 French test queries — 24 answerable, 7 unanswerable by
design — against an index of 120 French phrasings:

    top-1 on answerable        75.0%
    top-3 on answerable        87.5%
    at tau 0.625               13 of 24 answered, 86.7% precision,
                               100% of the unanswerable refused

Half the answerable questions are refused. That is a retrieval ceiling, not a
threshold to tune: adding four paraphrases per entry lifted every score,
including the wrong ones, and left top-1 unchanged at 75%.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes.config import get_settings


@dataclass(frozen=True)
class FaqEntry:
    """One row of the centre's FAQ."""

    faq_id: str
    question: str
    answer: str
    lang: str = "fr"
    #: The Darija answer, when the row has one.
    answer_tn: str | None = None
    #: Whether this point was the entry's primary phrasing or a paraphrase.
    phrasing: str | None = None


@dataclass(frozen=True)
class Retrieved:
    """One retrieved FAQ entry with its similarity to the caller's question."""

    entry: FaqEntry
    score: float

    @property
    def faq_id(self) -> str:
        """Shorthand for the entry's id."""
        return self.entry.faq_id


# --------------------------------------------------------------------------
# Reading the source
# --------------------------------------------------------------------------

def load_faq(path: str | Path | None = None) -> list[dict[str, str]]:
    """Read the raw FAQ CSV.

    Only needed to inspect or rebuild the index — retrieval reads Qdrant, not
    this file.
    """
    import csv

    settings = get_settings()
    target = Path(path) if path else settings.paths.raw / "faq.csv"
    with open(target, encoding="utf-8-sig", newline="") as handle:
        return [r for r in csv.DictReader(handle) if (r.get("faq_id") or "").strip()]


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def _entry_from_payload(payload: dict[str, Any], lang: str) -> FaqEntry:
    """Build an entry from a Qdrant payload, answering in the right language."""
    answer_fr = str(payload.get("answer_fr") or "")
    answer_tn = str(payload.get("answer_tn") or "") or None

    # French is the agent's spoken language: there is no free Darija TTS voice,
    # so a Darija question gets a French answer until there is one.
    answer = answer_fr or (answer_tn or "")

    return FaqEntry(
        faq_id=str(payload.get("faq_id") or ""),
        question=str(payload.get("question") or ""),
        answer=answer,
        lang=str(payload.get("lang") or lang),
        answer_tn=answer_tn,
        phrasing=str(payload.get("phrasing") or "") or None,
    )


def search(
    question: str,
    k: int | None = None,
    lang: str | None = "fr",
) -> list[Retrieved]:
    """Return the ``k`` closest FAQ entries to ``question``, best first.

    Results are deduplicated by ``faq_id``: an entry has several phrasings
    indexed, and three hits on three paraphrases of the same answer is one
    result, not three. The best-scoring phrasing wins.

    ``lang`` filters to one language's phrasings. It is worth doing — French
    queries were otherwise matching Darija variants, and filtering lifted
    top-3 from 64.5% to 67.7%.
    """
    from hermes import embeddings, vector_store

    settings = get_settings()
    wanted = k or settings.faq_top_k
    if not (question or "").strip():
        return []

    # Over-fetch, because several phrasings of one entry collapse into one
    # result and would otherwise leave fewer than k.
    hits = vector_store.search(
        settings.faq_collection,
        embeddings.encode_one(question),
        limit=max(wanted * 4, 12),
        where={"lang": lang} if lang else None,
    )

    best: dict[str, Retrieved] = {}
    for hit in hits:
        entry = _entry_from_payload(hit.payload, lang or "fr")
        if not entry.faq_id:
            continue
        seen = best.get(entry.faq_id)
        if seen is None or hit.score > seen.score:
            best[entry.faq_id] = Retrieved(entry=entry, score=hit.score)

    return sorted(best.values(), key=lambda r: r.score, reverse=True)[:wanted]


def best_match(
    question: str,
    threshold: float | None = None,
    lang: str | None = "fr",
) -> Retrieved | None:
    """Return the single best FAQ entry, or ``None`` if none is close enough.

    ``None`` is a real answer: it means hand the call over. The threshold is
    calibrated in `config.py` and is set on the safe side.
    """
    settings = get_settings()
    floor = settings.faq_accept_threshold if threshold is None else threshold

    hits = search(question, k=1, lang=lang)
    if not hits or hits[0].score < floor:
        return None
    return hits[0]


def answer(question: str, lang: str | None = "fr") -> str | None:
    """Return the FAQ answer to speak, or ``None`` when there is not one."""
    hit = best_match(question, lang=lang)
    return hit.entry.answer if hit else None


#: What the model is asked to do with a retrieved answer.
#:
#: The third rule exists because of a measured failure. A caller asked "Vous
#: êtes ouverts le samedi matin ?" and the `weekend` entry matched — but that
#: entry's answer is written as a reply to "Vous êtes ouverts le dimanche ?",
#: so it opens with "Non". Repeated faithfully, the agent answered "no" to a
#: question whose answer was yes. The passage holds the *facts*; its yes/no
#: belongs to whichever question it was written for.
PHRASING_SYSTEM = (
    "Tu es l'agent téléphonique d'un centre d'imagerie médicale. On te donne "
    "la réponse officielle du centre.\n"
    "1. Réponds à la question RÉELLEMENT posée, en UNE phrase naturelle en "
    "français.\n"
    "2. N'utilise que les faits présents dans la réponse officielle. "
    "N'invente rien, ne devine rien, ne donne jamais de conseil médical.\n"
    "3. La réponse officielle a pu être écrite pour une question légèrement "
    "différente : garde ses faits, mais n'en reprends pas le « oui » ou le "
    "« non » s'il ne correspond pas à la question posée.\n"
    "4. Si les faits ne répondent pas à la question, dis simplement que tu "
    "n'as pas cette information."
)

PHRASING_PROMPT = """Question du patient : {question}

Extraits de la FAQ officielle du centre :
{context}

Réponse à la question posée (une phrase) :"""


_CONTENT_HASH: dict[str, str] = {}


def content_hash() -> str:
    """A hash of the FAQ's answers, so an edit invalidates cached replies.

    Without it, correcting an answer in `faq.csv` would leave every cached
    reply still quoting the old text — and the cache is meant to be a replayed
    answer, not a stale one.
    """
    import hashlib

    if "hash" not in _CONTENT_HASH:
        material = "␟".join(
            f"{row.get('faq_id', '')}={row.get('answer_fr', '')}"
            for row in sorted(load_faq(), key=lambda r: r.get("faq_id", ""))
        )
        _CONTENT_HASH["hash"] = hashlib.sha256(
            material.encode("utf-8")
        ).hexdigest()[:12]
    return _CONTENT_HASH["hash"]


def normalise_question(question: str) -> str:
    """Collapse a question to what actually determines the answer.

    Case, spacing and trailing punctuation do not change the reply, so they
    should not split the cache. Accents are kept — they distinguish real
    words here, the same reason the WER normaliser keeps them.
    """
    import unicodedata

    text = unicodedata.normalize("NFC", question or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" ?!.,;:")


def answer_cache_key(question: str, hits: list[Retrieved]) -> str:
    """Cache key for one FAQ answer.

    Three parts, each for a reason: the **normalised question**, so phrasing
    noise does not split the cache; the **top-3 faq_ids**, because the same
    question against different passages is a different task; and the **FAQ
    content hash**, so editing an answer retires every reply that quoted it.
    """
    import hashlib

    material = "␟".join([
        normalise_question(question),
        ",".join(hit.faq_id for hit in hits),
        content_hash(),
    ])
    return "faq-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def fallback_is_safe(hits: list[Retrieved]) -> bool:
    """Whether the stored answer may be spoken without the model.

    The model is the relevance check — it is what catches a passage that
    scores well and answers a different question. On a timeout the check is
    exactly what is missing, so the retrieval has to carry the decision alone,
    and it is only trusted when it is both confident and unambiguous.

    Calibrated for zero wrong answers on `faq_test.csv`.
    """
    if not hits:
        return False
    settings = get_settings()
    best = hits[0].score
    runner = hits[1].score if len(hits) > 1 else 0.0
    return (
        best >= settings.faq_fallback_threshold
        and (best - runner) >= settings.faq_fallback_margin
    )


def spoken_answer(
    question: str,
    lang: str | None = "fr",
    timeout: float | None = None,
) -> tuple[str | None, str]:
    """Return the answer to speak and where it came from.

    The model sees the **top k** passages, not only the best one. Measured:
    top-1 retrieval is 83.3% and top-3 is 91.7%, so eight points of accuracy
    sit in passages that were retrieved and then thrown away. Asking about
    Friday afternoon matched `weekend` (0.871) ahead of `horaires` (0.676) —
    and `horaires` is the entry that actually answers it.

    The threshold still applies to the *best* hit, so a weak retrieval still
    hands over rather than fishing through near misses.

    The stored best answer remains the fallback, so a timeout, a 503 or a
    missing SDK degrade to the centre's own wording rather than to silence.

    Returns ``(text, source)``; source is "llm", "cache", "timeout",
    "error:…" or "no_match".
    """
    from hermes import llm

    hit = best_match(question, lang=lang)
    if hit is None:
        return None, "no_match"

    settings = get_settings()
    passages = search(question, k=settings.faq_top_k, lang=lang)

    text, source = llm.complete_or(
        fallback=hit.entry.answer,
        prompt=PHRASING_PROMPT.format(
            question=question, context=format_context(passages)
        ),
        system=PHRASING_SYSTEM,
        temperature=0.2,
        timeout=settings.llm_timeout_faq_s if timeout is None else timeout,
        key=answer_cache_key(question, passages),
    )

    # The fallback fired, so nothing checked whether the passage answers the
    # question. Speak it only from a retrieval confident enough to stand
    # alone; otherwise hand over, which is always a safe thing to do.
    if source not in ("llm", "cache") and not fallback_is_safe(passages):
        return None, f"{source}+transfer"

    return text, source


# --------------------------------------------------------------------------
# Grounding for the agent
# --------------------------------------------------------------------------

def format_context(hits: list[Retrieved]) -> str:
    """Render retrieved entries as the grounding block for the LLM prompt.

    Scores are included so the model can see which passage is the strong one.
    It is told to rephrase these and nothing else; an empty block means it has
    nothing to work from and should say so.
    """
    if not hits:
        return "(aucune entrée pertinente dans la FAQ)"

    lines = []
    for index, hit in enumerate(hits, start=1):
        lines.append(
            f"[{index}] ({hit.score:.2f}) Q: {hit.entry.question}\n"
            f"    R: {hit.entry.answer}"
        )
    return "\n".join(lines)


def is_answerable(question: str, lang: str | None = "fr") -> bool:
    """Whether the FAQ has something close enough to answer with."""
    return best_match(question, lang=lang) is not None
