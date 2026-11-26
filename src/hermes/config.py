"""Single source of truth for settings, filesystem paths and thresholds.

Reads `.env` once at import time (via python-dotenv) and exposes everything the
other modules need: API keys, model names, directory layout and the numeric
cut-offs used by the audio check, the intent gate, the exam matcher and the
escalation rules. No other module should call ``os.getenv`` or build paths by
hand.

Every threshold has a default here and an optional `.env` override, so the
project runs with no `.env` at all — which is what the tests rely on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    """Absolute paths to every directory and file the project reads or writes."""

    root: Path
    data: Path
    raw: Path
    audio: Path
    processed: Path
    models: Path
    reports: Path
    figures: Path
    agenda_db: Path
    llm_cache: Path
    call_log: Path
    #: Synthesised lines, keyed by hash. The agent repeats its prompts and the
    #: fixed safety script constantly, so each is spoken into a file once.
    tts_cache: Path


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    """Runtime configuration, assembled from `.env` and the defaults below."""

    gemini_api_key: str | None
    gemini_model: str
    ollama_model: str
    use_local_llm: bool

    stt_backend: str
    stt_model_size: str
    #: Where faster-whisper runs. "cpu" by default — see DEFAULTS for why.
    stt_device: str
    #: The quantisation faster-whisper loads with.
    stt_compute_type: str
    tts_backend: str
    #: A Piper .onnx voice. Empty means "first one found in models/piper".
    tts_voice: str | None
    #: The rate the agent's replies are written at. Higher than the 16 kHz the
    #: STT side uses: nothing transcribes these, and the listening test does.
    tts_sample_rate: int
    language: str

    embedding_model: str
    #: Dimensions the embedding model emits. Checked against the collection on
    #: startup, because a mismatch is otherwise a silent retrieval failure.
    embedding_dim: int

    # --- the vector store ---
    qdrant_host: str
    qdrant_port: int
    qdrant_api_key: str | None
    intent_collection: str
    faq_collection: str
    #: How many neighbours the intent vote looks at.
    intent_top_k: int
    #: How many FAQ entries are retrieved before the threshold is applied.
    faq_top_k: int

    min_snr_db: float
    #: Below this transcription confidence the turn is not acted on. Measured,
    #: not guessed: see the calibration note in DEFAULTS.
    min_confidence: float
    intent_accept_threshold: float
    intent_margin: float
    #: Guards below which an utterance is too short to classify at all.
    min_utterance_words: int
    min_utterance_letters: int
    #: P(medical) above which a turn is medical whatever else scores higher.
    medical_probability_floor: float
    #: FAQ score above which the centre's own answer beats a soft medical call.
    faq_override_threshold: float
    #: Below this top probability the gate asks rather than routes.
    intent_min_probability: float
    #: Filename of the fitted intent classifier inside models/.
    intent_model_file: str

    # --- the lexicon repair layer ---
    #: Filename of the built lexicon inside data/processed/.
    lexicon_file: str
    #: Similarity a token needs before a domain term may replace it.
    repair_min_ratio: float
    #: Zipf frequency above which a token counts as a real French word.
    french_word_min_zipf: float
    #: Similarity a token needs to raise a safety flag on its own.
    safety_flag_min_ratio: float
    #: Lower bar, usable only when the sentence is about imaging.
    safety_flag_context_ratio: float
    #: Above this frequency a word is too ordinary to be a fuzzy safety match.
    safety_flag_max_zipf: float
    #: Hard budget for one LLM call before the stored answer is used instead.
    llm_timeout_s: float
    #: Retrieval score the stored answer needs to be spoken without the model.
    faq_fallback_threshold: float
    #: And how far top-1 must lead top-2 on that path.
    faq_fallback_margin: float
    faq_accept_threshold: float
    exam_fuzzy_threshold: int
    frustration_escalate_threshold: float
    max_turns_before_escalation: int

    # --- the audio front end ---
    #: Level `audio_quality.normalize` brings every clip to.
    target_rms_dbfs: float
    #: The peak limiter's ceiling, so normalising never clips.
    peak_ceiling: float
    #: Below this fraction of speech-like frames a clip is mostly silence.
    min_speech_ratio: float
    #: Above this fraction of full-scale samples a clip is distorted.
    max_clipping_ratio: float
    #: Shorter than this and there is nothing to transcribe.
    min_duration_s: float
    #: SNR estimates are capped here; a clean clip's noise floor is ~0.
    max_snr_db: float

    paths: Paths


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

DEFAULTS: dict[str, object] = {
    # Chosen by calling models.list on this key and benchmarking what came
    # back, not from the plan — which named gemini-2.0-flash, now a 404.
    #
    #   gemini-3.8-flash        503 on every attempt, with backoff
    #   gemini-3.1-flash-lite   503 on every attempt
    #   gemini-3.5-flash-lite   3/3 calls, 3/3 valid JSON
    #
    # The newer Flash models are listed but not actually served to this free
    # tier. Latency on the one that works ranged 0.89 s to 33.4 s, against a
    # 5-second per-turn budget with roughly 1.5 s left after STT and TTS — so
    # caching every response is load-bearing, not an optimisation, and Ollama
    # stays the offline fallback the plan specifies.
    "GEMINI_MODEL": "gemini-3.5-flash-lite",
    "OLLAMA_MODEL": "mistral",
    "USE_LOCAL_LLM": False,
    "STT_BACKEND": "faster-whisper",
    # "small", not "base". Head to head on the same 35 clips: small 0.173 mean
    # WER against base's 0.310, and base collapses on exactly the vocabulary
    # this call line runs on — "IRM rachis lombaire" came back as "irrer un
    # hockey un bœur". small costs 0.64 xRT on CPU, still realtime, so base's
    # speed advantage buys nothing. (The later 45-clip run, which added the
    # 0 and -5 dB tiers, puts small at 0.256; it ran small only, so it is not
    # a comparison with base.)
    "STT_MODEL_SIZE": "small",
    # CPU int8 by default rather than probing for a GPU. The demo runs on a
    # school laptop; int8 is what makes "small" realtime there, and pinning it
    # keeps the measured latency reproducible. Set STT_DEVICE=auto to let the
    # loader pick CUDA when a GPU is present.
    "STT_DEVICE": "cpu",
    "STT_COMPUTE_TYPE": "int8",
    "TTS_BACKEND": "piper",
    "TTS_SAMPLE_RATE": 22050,
    "LANGUAGE": "fr",
    # Multilingual on purpose: the corpora carry French, Darija in Arabic
    # script, Arabizi and code-switched sentences, and one model has to embed
    # all four into the same space.
    "EMBEDDING_MODEL": (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    "EMBEDDING_DIM": 384,
    # Qdrant runs in Docker; see docker-compose.yml.
    "QDRANT_HOST": "localhost",
    "QDRANT_PORT": 6333,
    "INTENT_COLLECTION": "hermes_intents",
    "FAQ_COLLECTION": "hermes_faq",
    "INTENT_TOP_K": 5,
    "FAQ_TOP_K": 3,
    # Calibrated by scripts/calibrate_gate.py against measured WER, not picked.
    # At 11.2 dB the gate rejects 13 of 45 clips whose mean WER is 0.511 and
    # keeps 32 whose mean WER is 0.152 — the cut that isolates the unusable
    # group without throwing away turns the agent could still have handled.
    "MIN_SNR_DB": 11.2,
    # The same exercise on Whisper's own confidence: 0.62 rejects 12 clips at
    # mean WER 0.523 and keeps 33 at 0.159. This one gates *after* decoding, so
    # it catches what the SNR estimate cannot — a clean recording of a mumble.
    "MIN_CONFIDENCE": 0.62,
    # Calibrated end to end on 222 French examples (leave-one-out similarity,
    # since every one is in the index) against 28 held-out out-of-scope
    # sentences. The finding is that τ is a weak lever: out-of-scope safety is
    # flat at 75% from 0.40 to 0.525 and only reaches 89% at 0.65, by which
    # point 72 of 222 real callers are being told to rephrase.
    #
    # At this value just 2 of the 28 out-of-scope sentences are caught by the
    # threshold. The rest are handled by off_topic declining (11), the medical
    # rules (3) and the clarify floor (3) — which is where the safety actually
    # comes from. Raising τ buys little and costs callers.
    "INTENT_ACCEPT_THRESHOLD": 0.50,
    "INTENT_MARGIN": 0.08,
    # An utterance shorter than this never reaches the embedder. Measured: on
    # the out-of-scope set, "oui", "hmm", "euh..." and "allô ?" all match
    # "Opérateur !" at ~0.78, higher than almost any genuine in-scope example.
    # Short strings embed near short strings whatever they mean, so a
    # similarity threshold cannot separate them and must not be asked to.
    "MIN_UTTERANCE_WORDS": 2,
    "MIN_UTTERANCE_LETTERS": 5,
    # Asymmetric on purpose. If the classifier gives medical_question this much
    # probability, the turn is treated as medical even when another intent
    # scores higher. A false alarm costs one transfer; a miss costs safety, and
    # the measured leaks were "J'ai un pacemaker, je peux faire l'IRM ?" going
    # to new_appointment.
    "MEDICAL_PROBABILITY_FLOOR": 0.25,
    # A FAQ match this strong may answer a question the classifier called
    # medical. Deliberately above FAQ_ACCEPT_THRESHOLD: reading the centre's
    # own written answer is quoting policy, not practising medicine, but only
    # when the match is unambiguous. It never overrides the absolute patterns
    # in escalation.py — an implant question is not answerable from a document
    # however well it scores.
    "FAQ_OVERRIDE_THRESHOLD": 0.70,
    # Below this top probability the classifier is guessing between classes, so
    # the agent asks instead of acting. Cross-validated on the French corpus:
    #
    #   floor   misroutes caught   correct routes interrupted
    #   0.50          35%                  12%
    #   0.55          48%                  18%
    #   0.60          62%                  26%
    #
    # 0.60 maximises caught-minus-interrupted, and is still not the choice:
    # interrupting one correct call in four with "did you mean X or Y?" is a
    # lot on a phone line, and a third of the "misroutes" it catches are
    # info → medical_question, which already end in a transfer rather than a
    # wrong action. 0.55 takes most of the benefit for two thirds of the cost.
    "INTENT_MIN_PROBABILITY": 0.55,
    # Where train_intent_classifier.py writes the fitted model. Under models/,
    # which is gitignored: it is derived from the corpus and rebuilds in
    # seconds.
    "INTENT_MODEL_FILE": "intent_classifier.joblib",
    # --- the lexicon repair layer ---
    "LEXICON_FILE": "lexicon.json",
    # Domain repair is conservative: a wrong repair sends the caller to the
    # wrong exam. A token is only rewritten when it is this close to a
    # catalogue term AND is not a real French word.
    "REPAIR_MIN_RATIO": 0.88,
    # Above this Zipf frequency a token is a real French word and is left
    # alone. "brosses" is 3.11 and stays; "écographie" is 1.20 and does not.
    "FRENCH_WORD_MIN_ZIPF": 2.0,
    # Safety flagging never rewrites, only flags. Two bars, because a single
    # ratio provably cannot work here:
    #
    #   brosses  ~ broches   0.71   must flag
    #   brioche  ~ broches   0.86   must not
    #   brioches ~ broches   0.93   must not
    #
    # The signal is inverted on the decisive pair, so any single threshold
    # either misses "brosses" or flags pastry. What separates them is context:
    # "des brosses en métal dans mes bras" against "une brioche et un café".
    #
    # So: a very close match stands alone, and a looser one needs the sentence
    # to be about imaging.
    "SAFETY_FLAG_MIN_RATIO": 0.94,
    "SAFETY_FLAG_CONTEXT_RATIO": 0.70,
    # Above this Zipf frequency a word is too ordinary to be a mangled safety
    # term, and a fuzzy match on it is noise. "plan" (implant, 0.73) and
    # "centre" (enceinte, 0.71) both cleared the context bar in "Le plan du
    # centre est à l'entrée". "brosses" is 3.11 and stays flaggable, which is
    # the line this threshold has to sit on.
    "SAFETY_FLAG_MAX_ZIPF": 4.0,
    # --- the LLM answer step ---
    # Hard budget for one LLM call. Measured on this key, the same prompt to
    # the same model took anywhere from 0.89 s to 33.4 s, so the timeout is
    # not a safety margin — it is what keeps a turn inside the 5-second
    # target at all. Past it, the stored FAQ answer is spoken instead, which
    # is the centre's own text and needs no model.
    "LLM_TIMEOUT_S": 2.0,
    # When the LLM call fails, the stored answer is spoken only from a
    # retrieval this confident. Calibrated for **zero wrong answers** on the
    # fallback path, because the fallback has no relevance check behind it:
    # the model is what catches a confident-but-wrong passage, and on a
    # timeout the model is exactly what is missing.
    #
    # Pinned by one case — "Il y a un café dans le centre ?" retrieves `jeun`
    # at 0.711 with a +0.229 margin, so the margin cannot catch it and the
    # threshold must clear it. At 0.72 the fallback speaks 8 of the 20
    # answers top-1 gets right and transfers the other 12.
    "FAQ_FALLBACK_THRESHOLD": 0.72,
    # And top-1 must lead top-2 by this much. It excludes nothing at 0.72
    # today — every delivered answer leads by at least 0.072 — and is kept so
    # a near-tie cannot start being spoken when the corpus changes.
    "FAQ_FALLBACK_MARGIN": 0.05,
    # Calibrated on the 31 French rows of faq_test.csv — 24 answerable, 7
    # unanswerable by design — against 149 French phrasings (5.3 per entry).
    # At 0.625: 14 of 24 answered correctly, 93.3% precision, 86% of the
    # unanswerable refused. Top-1 on answerable is 83.3%, top-3 91.7%.
    #
    # 100% refusal would need 0.725, and would cut correct answers from 14 to
    # 8. Not taken, because the one query that slips at 0.625 is "Il y a un
    # café dans le centre ?" — the FAQ branch only ever sees `info` intents,
    # since anything medical is diverted before retrieval, so a wrong answer
    # here is a service failure rather than a safety one.
    #
    # Re-measure after any change to faq.csv. This has moved 0.575 -> 0.625
    # across two rounds of added paraphrases, because every score rises
    # together — the wrong ones faster than the right ones.
    "FAQ_ACCEPT_THRESHOLD": 0.625,
    "EXAM_FUZZY_THRESHOLD": 82,
    "FRUSTRATION_ESCALATE_THRESHOLD": 0.70,
    "MAX_TURNS_BEFORE_ESCALATION": 8,
    "TARGET_RMS_DBFS": -20.0,
    "PEAK_CEILING": 0.99,
    "MIN_SPEECH_RATIO": 0.10,
    "MAX_CLIPPING_RATIO": 0.01,
    "MIN_DURATION_S": 0.30,
    "MAX_SNR_DB": 60.0,
}


# --------------------------------------------------------------------------
# Environment readers
# --------------------------------------------------------------------------

def _raw(name: str) -> str | None:
    """Read one environment variable, treating blank as unset.

    `.env.example` ships keys with empty values, so a blank must mean "use the
    default" rather than "the empty string".
    """
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _as_str(name: str) -> str:
    return str(_raw(name) or DEFAULTS[name])


def _as_float(name: str) -> float:
    value = _raw(name)
    if value is None:
        return float(DEFAULTS[name])  # type: ignore[arg-type]
    try:
        return float(value)
    except ValueError:
        return float(DEFAULTS[name])  # type: ignore[arg-type]


def _as_int(name: str) -> int:
    value = _raw(name)
    if value is None:
        return int(DEFAULTS[name])  # type: ignore[arg-type]
    try:
        return int(float(value))
    except ValueError:
        return int(DEFAULTS[name])  # type: ignore[arg-type]


def _as_bool(name: str) -> bool:
    value = _raw(name)
    if value is None:
        return bool(DEFAULTS[name])
    return value.lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------
# Building the configuration
# --------------------------------------------------------------------------

def project_root() -> Path:
    """Return the repository root, found by walking up from this file.

    Anchored on `pyproject.toml` rather than a fixed number of parents, so the
    package still resolves its root when installed in editable mode.
    """
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    # src/hermes/config.py -> the repository root two levels up.
    return here.parents[2]


def load_env(env_file: Path | None = None) -> None:
    """Load environment variables from `.env`, without overriding real ones.

    A variable already set in the real environment wins, so CI and a shell
    export both beat the file. Missing python-dotenv is not fatal: the defaults
    above then apply.
    """
    path = env_file or (project_root() / ".env")
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a dependency
        return
    load_dotenv(path, override=False)


def build_paths(root: Path | None = None) -> Paths:
    """Build the :class:`Paths` layout and create the writable directories."""
    base = Path(root) if root is not None else project_root()
    data = base / "data"
    processed = data / "processed"
    reports = base / "reports"

    paths = Paths(
        root=base,
        data=data,
        raw=data / "raw",
        audio=data / "audio",
        processed=processed,
        models=base / "models",
        reports=reports,
        figures=reports / "figures",
        agenda_db=processed / "agenda.db",
        llm_cache=processed / "llm_cache.sqlite",
        call_log=processed / "calls.jsonl",
        tts_cache=processed / "tts_cache",
    )

    for directory in (
        paths.processed,
        paths.audio,
        paths.models,
        paths.reports,
        paths.figures,
        paths.tts_cache,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def load_settings(env_file: Path | None = None) -> Settings:
    """Read `.env` and return the fully populated :class:`Settings`."""
    load_env(env_file)
    return Settings(
        gemini_api_key=_raw("GEMINI_API_KEY"),
        gemini_model=_as_str("GEMINI_MODEL"),
        ollama_model=_as_str("OLLAMA_MODEL"),
        use_local_llm=_as_bool("USE_LOCAL_LLM"),
        stt_backend=_as_str("STT_BACKEND"),
        stt_model_size=_as_str("STT_MODEL_SIZE"),
        stt_device=_as_str("STT_DEVICE"),
        stt_compute_type=_as_str("STT_COMPUTE_TYPE"),
        tts_backend=_as_str("TTS_BACKEND"),
        tts_voice=_raw("TTS_VOICE"),
        tts_sample_rate=_as_int("TTS_SAMPLE_RATE"),
        language=_as_str("LANGUAGE"),
        embedding_model=_as_str("EMBEDDING_MODEL"),
        embedding_dim=_as_int("EMBEDDING_DIM"),
        qdrant_host=_as_str("QDRANT_HOST"),
        qdrant_port=_as_int("QDRANT_PORT"),
        qdrant_api_key=_raw("QDRANT_API_KEY"),
        intent_collection=_as_str("INTENT_COLLECTION"),
        faq_collection=_as_str("FAQ_COLLECTION"),
        intent_top_k=_as_int("INTENT_TOP_K"),
        faq_top_k=_as_int("FAQ_TOP_K"),
        min_snr_db=_as_float("MIN_SNR_DB"),
        min_confidence=_as_float("MIN_CONFIDENCE"),
        intent_accept_threshold=_as_float("INTENT_ACCEPT_THRESHOLD"),
        intent_margin=_as_float("INTENT_MARGIN"),
        min_utterance_words=_as_int("MIN_UTTERANCE_WORDS"),
        min_utterance_letters=_as_int("MIN_UTTERANCE_LETTERS"),
        medical_probability_floor=_as_float("MEDICAL_PROBABILITY_FLOOR"),
        faq_override_threshold=_as_float("FAQ_OVERRIDE_THRESHOLD"),
        intent_min_probability=_as_float("INTENT_MIN_PROBABILITY"),
        intent_model_file=_as_str("INTENT_MODEL_FILE"),
        lexicon_file=_as_str("LEXICON_FILE"),
        repair_min_ratio=_as_float("REPAIR_MIN_RATIO"),
        french_word_min_zipf=_as_float("FRENCH_WORD_MIN_ZIPF"),
        safety_flag_min_ratio=_as_float("SAFETY_FLAG_MIN_RATIO"),
        safety_flag_context_ratio=_as_float("SAFETY_FLAG_CONTEXT_RATIO"),
        safety_flag_max_zipf=_as_float("SAFETY_FLAG_MAX_ZIPF"),
        llm_timeout_s=_as_float("LLM_TIMEOUT_S"),
        faq_fallback_threshold=_as_float("FAQ_FALLBACK_THRESHOLD"),
        faq_fallback_margin=_as_float("FAQ_FALLBACK_MARGIN"),
        faq_accept_threshold=_as_float("FAQ_ACCEPT_THRESHOLD"),
        exam_fuzzy_threshold=_as_int("EXAM_FUZZY_THRESHOLD"),
        frustration_escalate_threshold=_as_float("FRUSTRATION_ESCALATE_THRESHOLD"),
        max_turns_before_escalation=_as_int("MAX_TURNS_BEFORE_ESCALATION"),
        target_rms_dbfs=_as_float("TARGET_RMS_DBFS"),
        peak_ceiling=_as_float("PEAK_CEILING"),
        min_speech_ratio=_as_float("MIN_SPEECH_RATIO"),
        max_clipping_ratio=_as_float("MAX_CLIPPING_RATIO"),
        min_duration_s=_as_float("MIN_DURATION_S"),
        max_snr_db=_as_float("MAX_SNR_DB"),
        paths=build_paths(),
    )


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings`, loading it on first use."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


def set_settings(settings: Settings) -> None:
    """Replace the cached settings. For tests, which override thresholds."""
    global _SETTINGS
    _SETTINGS = settings


def reset_settings() -> None:
    """Drop the cached settings so the next read rebuilds them."""
    global _SETTINGS
    _SETTINGS = None


def with_overrides(**changes: object) -> Settings:
    """Return a copy of the current settings with some fields replaced."""
    return replace(get_settings(), **changes)  # type: ignore[arg-type]


def require_gemini_key(settings: Settings | None = None) -> str:
    """Return the Gemini API key, raising if it is missing from the environment."""
    active = settings or get_settings()
    if not active.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or set USE_LOCAL_LLM=true to run against Ollama instead."
        )
    return active.gemini_api_key
