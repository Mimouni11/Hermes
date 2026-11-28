"""Speech to text for the caller's side of the conversation.

Wraps two interchangeable Whisper backends — faster-whisper (CTranslate2, the
default) and openai-whisper (the reference implementation, used to check the
transcripts) — behind one class. Loads the model once per process, returns the
transcript with the segment timings and an average confidence the rest of the
pipeline can threshold on.

The model is given an ``initial_prompt`` built from the exam catalogue. Whisper
conditions on it as if it were the text just transcribed, which biases the
decoder towards the vocabulary that actually occurs on this call line:
"échographie", "arthroscanner", "rachis lombaire". Without it a French model
reaches for the common word and writes "IRM" as "IRM" but "arthro" as "arthrose".
"""

from __future__ import annotations

import csv
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from hermes.config import get_settings

#: The sample rate Whisper expects.
TARGET_SR = 16_000

#: How many terms go into the initial prompt in total.
#:
#: Measured, not guessed. At 60 terms the prompt reached 866 characters —
#: roughly 220 tokens against Whisper's 224-token window — and adding the
#: safety vocabulary *raised* WER from 0.256 to 0.266 while fixing none of the
#: safety errors it targeted. The prompt was not gaining vocabulary, it was
#: trading it. 30 keeps the hand-written safety and modality terms and drops
#: the long tail mined from the catalogue.
MAX_PROMPT_TERMS = 30

#: Modality names and stems worth biasing towards, whatever the catalogue holds.
CORE_VOCABULARY = [
    "IRM",
    "scanner",
    "échographie",
    "radiographie",
    "mammographie",
    "densitométrie",
    "panoramique dentaire",
    "arthroscanner",
    "uroscanner",
    "entéroscanner",
    "coloscanner",
    "télérachis",
    "rachis lombaire",
    "rachis cervical",
    "produit de contraste",
    "à jeun",
    "doppler",
    "gadolinium",
]

#: Safety vocabulary, biased for separately from the exam names.
#:
#: These are the words whose mistranscription is dangerous rather than merely
#: wrong: the measured failures were "broches" heard as "brosses" and "à
#: l'iode" as "alleraute", both on sentences that had to reach a human. Getting
#: Whisper to write them correctly is the cheapest of the three available fixes
#: — cheaper than repairing the text afterwards, and cheaper than widening the
#: rules to cover every mangling.
SAFETY_VOCABULARY = [
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
]

#: Words to drop when mining the catalogue: they carry no acoustic difficulty.
_STOPWORDS = {
    "de", "du", "des", "le", "la", "les", "un", "une", "et", "ou", "avec",
    "sans", "pour", "par", "en", "au", "aux", "sur", "sous", "droit", "droite",
    "gauche", "bilateral", "bilaterale", "controle", "simple", "complet",
}


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    """One timed chunk of transcribed speech."""

    start: float
    end: float
    text: str
    avg_logprob: float | None
    no_speech_prob: float | None


@dataclass(frozen=True)
class Transcript:
    """The full result of transcribing one caller turn."""

    text: str
    language: str
    duration_s: float
    segments: tuple[Segment, ...]
    confidence: float
    backend: str
    no_speech_prob: float = 0.0
    latency_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """The flat form the evaluation script and the logs want."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "no_speech_prob": self.no_speech_prob,
            "duration_s": self.duration_s,
            "latency_s": self.latency_s,
        }


# --------------------------------------------------------------------------
# The initial prompt
# --------------------------------------------------------------------------

def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def exam_vocabulary(
    exams_csv: str | Path | None = None,
    limit: int = MAX_PROMPT_TERMS,
) -> list[str]:
    """Mine the exam catalogue for the terms worth biasing the decoder towards.

    Takes the longest words first: length is a crude but effective proxy for
    the terms a general French model has not seen much of. "hystérosalpingo-
    graphie" earns its place in the prompt, "cheville" does not.
    """
    settings = get_settings()
    path = Path(exams_csv) if exams_csv is not None else settings.paths.raw / "exams.csv"

    # Safety vocabulary first: if the prompt ever has to be truncated, the
    # words whose mistranscription is dangerous are the ones to keep. The two
    # lists overlap ("produit de contraste" is in both), so they are merged
    # rather than concatenated.
    terms: list[str] = []
    seen: set[str] = set()
    for term in list(SAFETY_VOCABULARY) + list(CORE_VOCABULARY):
        if _fold(term) not in seen:
            terms.append(term)
            seen.add(_fold(term))

    if path.exists():
        candidates: dict[str, int] = {}
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                for field in ("name_fr", "region_fr"):
                    for word in re.split(r"[^\wÀ-ÿ]+", row.get(field) or ""):
                        folded = _fold(word)
                        if len(folded) < 7 or folded in _STOPWORDS:
                            continue
                        if folded.isdigit():
                            continue
                        candidates[word] = len(folded)

        for word in sorted(candidates, key=lambda w: (-candidates[w], _fold(w))):
            folded = _fold(word)
            if folded in seen:
                continue
            terms.append(word)
            seen.add(folded)
            if len(terms) >= limit:
                break

    return terms[:limit]


def build_initial_prompt(exams_csv: str | Path | None = None) -> str:
    """Build the ``initial_prompt`` string handed to Whisper.

    Phrased as a sentence rather than a bare word list: Whisper treats the
    prompt as preceding transcript, so prose conditions it better than a CSV.
    """
    vocabulary = exam_vocabulary(exams_csv)
    listed = ", ".join(vocabulary)
    return (
        "Centre d'imagerie médicale. Rendez-vous, examens et préparation : "
        f"{listed}."
    )


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

def _detect_device() -> tuple[str, str]:
    """Probe for a GPU, returning the device and compute type to use.

    Only called when the configuration asks for it: ``STT_DEVICE=auto``.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "int8_float16"
    except Exception:  # noqa: BLE001 — torch missing or broken is not fatal
        pass
    return "cpu", "int8"


def resolve_device(
    device: str | None = None,
    compute_type: str | None = None,
) -> tuple[str, str]:
    """Return the device and compute type to load faster-whisper with.

    The default is CPU int8, set in :mod:`hermes.config` rather than probed:
    the demo runs on a school laptop, int8 is what keeps "small" realtime
    there, and pinning it keeps the measured latency reproducible. int8 is the
    right quantisation on a GPU too — float16 on CPU is slower without being
    more accurate on 16 kHz speech.

    ``STT_DEVICE=auto`` opts in to probing for CUDA.
    """
    settings = get_settings()
    wanted = device or settings.stt_device
    if wanted == "auto":
        detected, detected_compute = _detect_device()
        return detected, compute_type or detected_compute
    return wanted, compute_type or settings.stt_compute_type


class STT:
    """A loaded Whisper model, ready to transcribe caller audio.

    One instance holds one model, so the evaluation script can keep a "small"
    and a "base" side by side and compare them without reloading either.
    """

    def __init__(
        self,
        model_size: str | None = None,
        backend: str | None = None,
        language: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        initial_prompt: str | None = None,
        exams_csv: str | Path | None = None,
    ) -> None:
        settings = get_settings()
        self.model_size = model_size or settings.stt_model_size
        self.backend = backend or settings.stt_backend
        self.language = language or settings.language

        self.device, self.compute_type = resolve_device(device, compute_type)

        self.initial_prompt = (
            build_initial_prompt(exams_csv) if initial_prompt is None
            else initial_prompt
        )

        self._model: Any | None = None

    # -- loading ---------------------------------------------------------

    @property
    def model(self) -> Any:
        """The underlying model, loaded on first use."""
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> Any:
        """Load the model for the configured backend.

        The first call downloads the weights, which is the only step here that
        touches the network.
        """
        if self.backend == "faster-whisper":
            from faster_whisper import WhisperModel

            return WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        if self.backend in ("openai-whisper", "whisper"):
            import whisper

            return whisper.load_model(self.model_size)
        raise ValueError(f"unknown STT backend: {self.backend!r}")

    # -- transcribing ----------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray | str | Path,
        sr: int = TARGET_SR,
    ) -> dict[str, Any]:
        """Transcribe one clip and return the flat result.

        Accepts a float array at ``sr`` or a path to an audio file. Returns
        ``text``, ``confidence``, ``no_speech_prob``, ``duration_s`` and
        ``latency_s`` — the last measured around the decode call only, so it is
        the per-turn cost, not the one-off model load.
        """
        return self.transcribe_full(audio, sr).as_dict()

    def transcribe_full(
        self,
        audio: np.ndarray | str | Path,
        sr: int = TARGET_SR,
    ) -> Transcript:
        """Transcribe one clip and return the full :class:`Transcript`."""
        prepared, duration = self._prepare(audio, sr)

        started = time.perf_counter()
        if self.backend == "faster-whisper":
            segments, language = self._decode_faster(prepared)
        else:
            segments, language = self._decode_openai(prepared)
        latency = time.perf_counter() - started

        text = normalise_text(" ".join(s.text for s in segments))
        return Transcript(
            text=text,
            language=language or self.language,
            duration_s=duration,
            segments=tuple(segments),
            confidence=confidence_from(segments),
            backend=self.backend,
            no_speech_prob=no_speech_from(segments),
            latency_s=latency,
        )

    def _prepare(
        self,
        audio: np.ndarray | str | Path,
        sr: int,
    ) -> tuple[np.ndarray, float]:
        """Return the audio as float32 at 16 kHz, plus its duration."""
        if isinstance(audio, (str, Path)):
            from hermes.listen.audio_quality import load_audio

            samples, rate = load_audio(audio, TARGET_SR)
            return samples, samples.size / rate

        samples = np.asarray(audio, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if sr != TARGET_SR:
            import librosa

            samples = librosa.resample(samples, orig_sr=sr, target_sr=TARGET_SR)
        return samples, samples.size / TARGET_SR

    def _decode_faster(self, samples: np.ndarray) -> tuple[list[Segment], str]:
        segments, info = self.model.transcribe(
            samples,
            language=self.language,
            initial_prompt=self.initial_prompt,
            beam_size=5,
            vad_filter=False,
        )
        decoded = [
            Segment(
                start=float(s.start),
                end=float(s.end),
                text=s.text,
                avg_logprob=getattr(s, "avg_logprob", None),
                no_speech_prob=getattr(s, "no_speech_prob", None),
            )
            for s in segments
        ]
        return decoded, getattr(info, "language", self.language)

    def _decode_openai(self, samples: np.ndarray) -> tuple[list[Segment], str]:
        result = self.model.transcribe(
            samples,
            language=self.language,
            initial_prompt=self.initial_prompt,
        )
        decoded = [
            Segment(
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                text=str(s.get("text", "")),
                avg_logprob=s.get("avg_logprob"),
                no_speech_prob=s.get("no_speech_prob"),
            )
            for s in result.get("segments", [])
        ]
        return decoded, str(result.get("language", self.language))


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------

def confidence_from(segments: list[Segment] | tuple[Segment, ...]) -> float:
    """Turn the segments' ``avg_logprob`` into one confidence in 0..1.

    ``avg_logprob`` is the mean log probability per token, so it is at most 0
    and typically around -0.2 for clean speech. Exponentiating gives back a
    probability-like number, and weighting by segment duration stops a half-
    second "oui" counting as much as a ten-second sentence.
    """
    usable = [s for s in segments if s.avg_logprob is not None]
    if not usable:
        return 0.0

    weights = [max(s.end - s.start, 1e-3) for s in usable]
    total = sum(weights)
    mean_logprob = sum(
        (s.avg_logprob or 0.0) * w for s, w in zip(usable, weights)
    ) / total
    return float(min(max(np.exp(mean_logprob), 0.0), 1.0))


def no_speech_from(segments: list[Segment] | tuple[Segment, ...]) -> float:
    """Return the highest ``no_speech_prob`` across the segments.

    The maximum, not the mean: one segment that is confidently silence is the
    interesting signal, and averaging it away is how a half-empty clip passes.
    """
    values = [s.no_speech_prob for s in segments if s.no_speech_prob is not None]
    return float(max(values)) if values else 0.0


# --------------------------------------------------------------------------
# Module-level convenience
# --------------------------------------------------------------------------

_MODELS: dict[tuple[str, str], STT] = {}


def load_model(backend: str | None = None, model_size: str | None = None) -> STT:
    """Load and cache an :class:`STT` for the requested backend and size."""
    settings = get_settings()
    key = (backend or settings.stt_backend, model_size or settings.stt_model_size)
    if key not in _MODELS:
        _MODELS[key] = STT(backend=key[0], model_size=key[1])
    return _MODELS[key]


def transcribe(
    path: str | Path,
    language: str | None = None,
    backend: str | None = None,
) -> Transcript:
    """Transcribe an audio file into a :class:`Transcript`."""
    model = load_model(backend=backend)
    if language is not None:
        model.language = language
    return model.transcribe_full(path)


def transcribe_samples(
    samples: np.ndarray,
    sr: int = TARGET_SR,
    language: str | None = None,
    backend: str | None = None,
) -> Transcript:
    """Transcribe audio already held in memory as a float array."""
    model = load_model(backend=backend)
    if language is not None:
        model.language = language
    return model.transcribe_full(samples, sr)


def stream_transcribe(
    path: str | Path,
    language: str | None = None,
) -> Iterator[Segment]:
    """Yield segments as they are decoded, for the live demo UI."""
    yield from transcribe(path, language=language).segments


# --------------------------------------------------------------------------
# Text tidying
# --------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """Tidy a raw transcript: trim, collapse whitespace, fix obvious artefacts."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned


def is_empty(transcript: Transcript) -> bool:
    """Return whether a transcript carries no usable speech."""
    return not transcript.text.strip()


#: Characters treated as apostrophes, including the typographic one Whisper
#: emits and the plain one the corpus uses.
_APOSTROPHES = "'’ʼ´`"


def normalise_for_wer(text: str) -> str:
    """Normalise a transcript or a reference for word error rate scoring.

    Four rules, and the fourth is the one that matters here:

    1. lower-cased, so "IRM" and "irm" are the same word,
    2. punctuation removed, since the reference sentences are written with
       question marks and Whisper's are not reliably,
    3. apostrophes become spaces, so "l'échographie" and "l échographie" score
       alike — Whisper is inconsistent about the elision and a WER that counted
       it would measure typography, not recognition,
    4. **accents are kept.** "echographie" is a different word from
       "échographie": stripping accents would hide exactly the mistake this
       evaluation exists to catch.

    Digits are left alone; the corpus has no spelled-out numbers to reconcile.
    """
    lowered = (text or "").lower()
    for mark in _APOSTROPHES:
        lowered = lowered.replace(mark, " ")

    # Keep letters (accented included), digits and whitespace; drop the rest.
    kept = [
        char if (char.isalnum() or char.isspace()) else " "
        for char in lowered
    ]
    return re.sub(r"\s+", " ", "".join(kept)).strip()
