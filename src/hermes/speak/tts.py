"""Text to speech for the agent's side of the conversation.

Two free backends behind one function: piper-tts (local, fast, the default for
the demo) and gTTS (network, used as a fallback and for voice comparison in the
report). Synthesised lines are cached on disk by hash, since the agent repeats
the same prompts and safety scripts constantly.

The work that is not synthesis is :func:`prepare_text`. A reply from the agent
carries a reference code and an appointment time, and a TTS voice reads neither
the way a person would: "K7M2PQ" comes out as a word, "14h30" as digits. The
text is expanded before it reaches the voice.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hermes.config import get_settings

#: Where Piper voices are looked for when none is configured.
VOICE_GLOB = "*.onnx"

#: gTTS and Piper both get resampled to one rate, so the cache and the player
#: never have to care which backend produced a line.
_EPS = 1e-12


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Speech:
    """One synthesised utterance."""

    path: Path
    text: str
    duration_s: float
    sample_rate: int
    backend: str
    from_cache: bool
    latency_s: float = 0.0


# --------------------------------------------------------------------------
# Making text speakable
# --------------------------------------------------------------------------

#: 0-59, which covers hours, minutes and days of the month.
_UNITS = [
    "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit",
    "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
    "dix-sept", "dix-huit", "dix-neuf",
]
_TENS = {20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante"}

MONTHS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]

#: Written forms a French voice mangles, and what to say instead. Acronyms are
#: spaced out so the voice reads letters rather than inventing a word.
ABBREVIATIONS: dict[str, str] = {
    "rdv": "rendez-vous",
    "r.d.v.": "rendez-vous",
    "dr": "docteur",
    "pr": "professeur",
    "mme": "madame",
    "mlle": "mademoiselle",
    "m.": "monsieur",
    "min": "minutes",
    "h": "heures",
    "nb": "nota bene",
    "cf": "voir",
    "etc": "et cetera",
}

#: Medical acronyms, spelled so the voice says the letters. Which of these a
#: given Piper voice actually needs is a listening question, not a guess — see
#: docs/tts.md.
ACRONYMS: dict[str, str] = {
    "IRM": "I.R.M.",
    "TDM": "T.D.M.",
    "ORL": "O.R.L.",
    "CBCT": "C.B.C.T.",
    "EEG": "E.E.G.",
    "ECG": "E.C.G.",
}

#: A reference code: six characters from the agenda's alphabet, no I/O/0/1.
REF_CODE = re.compile(r"\b(?=[A-Z0-9]{6}\b)(?=.*[A-Z])[A-HJ-NP-Z2-9]{6}\b")

_TIME = re.compile(r"\b(\d{1,2})\s*[:hH]\s*(\d{2})?\b")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def number_fr(value: int) -> str:
    """Spell a number from 0 to 59 in French."""
    if value < 0 or value > 59:
        return str(value)
    if value < 20:
        return _UNITS[value]
    tens, unit = divmod(value, 10)
    word = _TENS[tens * 10]
    if unit == 0:
        return word
    if unit == 1:
        return f"{word} et un"
    return f"{word}-{_UNITS[unit]}"


def say_time(hour: int, minute: int | None = None) -> str:
    """Render a clock time the way it is spoken: 14h30 -> quatorze heures trente."""
    hour_word = "une heure" if hour == 1 else f"{number_fr(hour)} heures"
    if not minute:
        return hour_word
    return f"{hour_word} {number_fr(minute)}"


def say_ref_code(code: str) -> str:
    """Space a reference code out so it is read letter by letter.

    "K7M2PQ" spoken as a word is useless to a caller writing it down. Each
    character is separated so the voice names it, and the group is bracketed by
    commas so there is a pause either side.
    """
    return ", ".join(code)


def prepare_text(text: str) -> str:
    """Make text speakable: expand abbreviations, dates, times and exam codes.

    Order matters. Reference codes are pulled out first, before the acronym
    pass can touch them; times are expanded before the bare "h" abbreviation
    would turn "14h30" into "14 heures 30" with the digits still unspoken.
    """
    if not text:
        return ""

    spoken = text

    # 1. Reference codes, before anything else can rewrite their letters.
    spoken = REF_CODE.sub(lambda m: say_ref_code(m.group(0)), spoken)

    # 2. ISO dates: 2026-11-26 -> vingt-six novembre. No article: the sentence
    # around it already supplies one, and "Le le vingt-six" is what prepending
    # produces.
    def _date(match: re.Match[str]) -> str:
        _, month, day = match.groups()
        index = int(month) - 1
        if not 0 <= index < 12:
            return match.group(0)
        return f"{number_fr(int(day))} {MONTHS_FR[index]}"

    spoken = _ISO_DATE.sub(_date, spoken)

    # 3. Clock times: 14h30, 14:30, 14h
    def _time(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else None
        if hour > 23 or (minute is not None and minute > 59):
            return match.group(0)
        return say_time(hour, minute)

    spoken = _TIME.sub(_time, spoken)

    # 4. Medical acronyms, whole words only.
    for written, said in ACRONYMS.items():
        spoken = re.sub(rf"\b{re.escape(written)}\b", said, spoken)

    # 5. Ordinary abbreviations, case-insensitively.
    for written, said in ABBREVIATIONS.items():
        spoken = re.sub(
            rf"(?<![\w.]){re.escape(written)}(?![\w])", said, spoken,
            flags=re.IGNORECASE,
        )

    return re.sub(r"\s+", " ", spoken).strip()


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------

def cache_key(text: str, backend: str, voice: str | None) -> str:
    """Return the stable cache filename for one utterance.

    Keyed on the *spoken* text, the backend and the voice: change any of them
    and the line is synthesised again, which is what makes the cache safe to
    leave in place while the voice is still being chosen.
    """
    material = "␟".join([text or "", backend or "", voice or ""])
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"{backend}-{digest}.wav"


def clear_cache() -> int:
    """Delete the cached audio and return how many files were removed."""
    directory = get_settings().paths.tts_cache
    if not directory.is_dir():
        return 0
    removed = 0
    for path in directory.glob("*.wav"):
        path.unlink()
        removed += 1
    return removed


# --------------------------------------------------------------------------
# Voices
# --------------------------------------------------------------------------

_VOICES: dict[tuple[str, str], Any] = {}


def find_voice_file(voice: str | None = None) -> Path | None:
    """Return the Piper voice model to use, or None if none is installed."""
    settings = get_settings()
    chosen = voice or settings.tts_voice
    if chosen:
        candidate = Path(chosen)
        return candidate if candidate.exists() else None

    directory = settings.paths.models / "piper"
    if not directory.is_dir():
        return None
    found = sorted(directory.glob(VOICE_GLOB))
    return found[0] if found else None


def load_voice(backend: str | None = None, voice: str | None = None) -> Any:
    """Load and cache a TTS voice for the requested backend.

    gTTS has nothing to load — it is a network call per line — so it returns
    None and the backend name carries the meaning.
    """
    settings = get_settings()
    chosen_backend = backend or settings.tts_backend

    if chosen_backend != "piper":
        return None

    path = find_voice_file(voice)
    if path is None:
        raise RuntimeError(
            f"no Piper voice found in {settings.paths.models / 'piper'}. "
            "Download a fr_FR voice (.onnx and .onnx.json) from "
            "https://huggingface.co/rhasspy/piper-voices, or set TTS_BACKEND=gtts."
        )

    key = (chosen_backend, str(path))
    if key not in _VOICES:
        from piper.voice import PiperVoice

        _VOICES[key] = PiperVoice.load(str(path))
    return _VOICES[key]


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------

def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype("float32")
    import librosa

    return librosa.resample(
        samples.astype("float32"), orig_sr=source_rate, target_sr=target_rate
    ).astype("float32")


def _piper_samples(text: str, voice: Any) -> tuple[np.ndarray, int]:
    """Synthesise with Piper, returning float samples at the voice's own rate."""
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        # The method was renamed between Piper releases; both write a WAV.
        synth = getattr(voice, "synthesize_wav", None) or voice.synthesize
        synth(text, handle)

    buffer.seek(0)
    with wave.open(buffer, "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())

    samples = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def _gtts_samples(text: str) -> tuple[np.ndarray, int]:
    """Synthesise with gTTS. This call leaves the machine."""
    import tempfile

    import librosa
    from gtts import gTTS

    settings = get_settings()
    with tempfile.TemporaryDirectory() as workspace:
        mp3 = Path(workspace) / "line.mp3"
        gTTS(text=text, lang=settings.language).save(str(mp3))
        samples, rate = librosa.load(str(mp3), sr=None, mono=True)
    return samples.astype("float32"), int(rate)


def synthesise_samples(
    text: str,
    backend: str | None = None,
    voice: str | None = None,
) -> tuple[np.ndarray, int]:
    """Render ``text`` straight to samples, skipping the disk.

    What the live demo uses: a microphone turn does not need a file, and this
    avoids the write and the read back. Nothing is cached on this path.
    """
    settings = get_settings()
    chosen = backend or settings.tts_backend
    spoken = prepare_text(text)
    if not spoken:
        return np.zeros(0, dtype="float32"), settings.tts_sample_rate

    if chosen == "piper":
        samples, rate = _piper_samples(spoken, load_voice(chosen, voice))
    elif chosen in ("gtts", "gTTS"):
        samples, rate = _gtts_samples(spoken)
    else:
        raise ValueError(f"unknown TTS backend: {chosen!r}")

    target = settings.tts_sample_rate
    samples = _resample(samples, rate, target)

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak
    return samples, target


def synthesise(
    text: str,
    out_path: str | Path | None = None,
    backend: str | None = None,
    voice: str | None = None,
    use_cache: bool = True,
) -> Speech:
    """Render ``text`` to a wav file and return where it landed.

    With no ``out_path`` the line goes to the cache, keyed by hash, and a
    second request for the same words returns the same file without touching
    the voice. ``use_cache=False`` forces a fresh synthesis, which is what the
    listening test wants when comparing voices.
    """
    import soundfile as sf

    settings = get_settings()
    chosen = backend or settings.tts_backend
    spoken = prepare_text(text)

    voice_file = find_voice_file(voice) if chosen == "piper" else None
    voice_id = str(voice_file) if voice_file else (voice or "")

    if out_path is None:
        target = settings.paths.tts_cache / cache_key(spoken, chosen, voice_id)
        cacheable = use_cache
    else:
        target = Path(out_path)
        cacheable = False

    if cacheable and target.exists():
        info = sf.info(target)
        return Speech(
            path=target,
            text=spoken,
            duration_s=info.frames / info.samplerate,
            sample_rate=info.samplerate,
            backend=chosen,
            from_cache=True,
        )

    started = time.perf_counter()
    samples, rate = synthesise_samples(text, backend=chosen, voice=voice)
    latency = time.perf_counter() - started

    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, samples, rate, subtype="PCM_16")

    return Speech(
        path=target,
        text=spoken,
        duration_s=samples.size / rate if rate else 0.0,
        sample_rate=rate,
        backend=chosen,
        from_cache=False,
        latency_s=latency,
    )
