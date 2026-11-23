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

    min_snr_db: float
    #: Below this transcription confidence the turn is not acted on. Measured,
    #: not guessed: see the calibration note in DEFAULTS.
    min_confidence: float
    intent_accept_threshold: float
    intent_margin: float
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
    "GEMINI_MODEL": "gemini-2.0-flash",
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
    "EMBEDDING_MODEL": (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    # Calibrated by scripts/calibrate_gate.py against measured WER, not picked.
    # At 11.2 dB the gate rejects 13 of 45 clips whose mean WER is 0.511 and
    # keeps 32 whose mean WER is 0.152 — the cut that isolates the unusable
    # group without throwing away turns the agent could still have handled.
    "MIN_SNR_DB": 11.2,
    # The same exercise on Whisper's own confidence: 0.62 rejects 12 clips at
    # mean WER 0.523 and keeps 33 at 0.159. This one gates *after* decoding, so
    # it catches what the SNR estimate cannot — a clean recording of a mumble.
    "MIN_CONFIDENCE": 0.62,
    "INTENT_ACCEPT_THRESHOLD": 0.55,
    "INTENT_MARGIN": 0.08,
    "FAQ_ACCEPT_THRESHOLD": 0.60,
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
        min_snr_db=_as_float("MIN_SNR_DB"),
        min_confidence=_as_float("MIN_CONFIDENCE"),
        intent_accept_threshold=_as_float("INTENT_ACCEPT_THRESHOLD"),
        intent_margin=_as_float("INTENT_MARGIN"),
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
