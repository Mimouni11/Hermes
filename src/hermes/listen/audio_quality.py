"""Gate on the quality of the incoming clip before paying for transcription.

Estimates the signal-to-noise ratio of a caller's recording, plus the cheap
companions (clipping, silence, duration), and decides whether the turn is worth
transcribing or whether the agent should ask the caller to repeat themselves.
Backed by librosa and soundfile for loading, numpy for the arithmetic.

Three functions carry the work:

- :func:`normalize` brings a clip to a fixed RMS so the rest of the pipeline
  sees one level, whoever is calling and whatever they called from,
- :func:`estimate_snr` compares the loud frames to the quiet ones,
- :func:`check` turns that into ``"ok"`` or ``"repeat"``.

Every threshold comes from :mod:`hermes.config`; none is written here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hermes.config import get_settings

#: The sample rate the whole pipeline works at.
TARGET_SR = 16_000

#: Frame length for the energy analysis. 25 ms is the usual speech frame: long
#: enough to hold a pitch period, short enough to sit inside one phoneme.
FRAME_MS = 25

#: The quietest frames taken to be the noise floor.
NOISE_PERCENTILE = 10.0

#: The loudest frames taken to be speech.
SPEECH_PERCENTILE = 90.0

#: A frame this far above the noise floor counts as speech, for speech_ratio.
SPEECH_MARGIN_DB = 6.0

#: Guards every logarithm and division below.
_EPS = 1e-12


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioReport:
    """What the quality check found in one clip."""

    snr_db: float
    duration_s: float
    rms: float
    peak: float
    clipping_ratio: float
    speech_ratio: float
    is_silent: bool
    is_acceptable: bool
    reason: str | None

    @property
    def verdict(self) -> str:
        """``"ok"`` or ``"repeat"``, the form the pipeline branches on."""
        return "ok" if self.is_acceptable else "repeat"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load a clip as mono float32 resampled to ``target_sr``."""
    import soundfile as sf

    samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != target_sr:
        import librosa

        samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return np.ascontiguousarray(samples, dtype="float32"), sr


# --------------------------------------------------------------------------
# Level
# --------------------------------------------------------------------------

def rms(samples: np.ndarray) -> float:
    """Return the root-mean-square level of a clip."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype="float64"))))


def dbfs(level: float) -> float:
    """Convert a linear amplitude to dBFS, with a floor instead of -inf."""
    return 20.0 * float(np.log10(max(level, _EPS)))


def normalize(
    samples: np.ndarray,
    target_dbfs: float | None = None,
    peak_ceiling: float | None = None,
) -> np.ndarray:
    """Bring a clip to a fixed RMS level, limiting the peak so nothing clips.

    The gain that puts the RMS at ``target_dbfs`` can push the loudest sample
    past full scale — a clip with a few sharp transients and a quiet body is
    exactly that case. When it does, the whole clip is scaled back so the peak
    lands on ``peak_ceiling``, which leaves the RMS below target. That is the
    right trade: a consistent level is worth less than an undistorted one, and
    clipping is not recoverable.

    Silence is returned untouched, since there is no level to scale.
    """
    settings = get_settings()
    target = settings.target_rms_dbfs if target_dbfs is None else target_dbfs
    ceiling = settings.peak_ceiling if peak_ceiling is None else peak_ceiling

    audio = np.asarray(samples, dtype="float32")
    if audio.size == 0:
        return audio

    current = rms(audio)
    if current <= _EPS:
        return audio

    gain = (10.0 ** (target / 20.0)) / current
    scaled = audio * gain

    peak = float(np.max(np.abs(scaled)))
    if peak > ceiling:
        scaled = scaled * (ceiling / peak)

    return scaled.astype("float32")


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

def frame_energy(
    samples: np.ndarray,
    sr: int,
    frame_ms: int = FRAME_MS,
) -> np.ndarray:
    """Return per-frame RMS energy, used to separate speech from background."""
    audio = np.asarray(samples, dtype="float32")
    if audio.size == 0:
        return np.zeros(0, dtype="float64")

    frame = max(int(sr * frame_ms / 1000), 1)
    if audio.size < frame:
        return np.array([rms(audio)], dtype="float64")

    usable = (audio.size // frame) * frame
    frames = audio[:usable].reshape(-1, frame).astype("float64")
    return np.sqrt(np.mean(np.square(frames), axis=1))


def estimate_noise_floor(
    energies: np.ndarray,
    percentile: float = NOISE_PERCENTILE,
) -> float:
    """Estimate the background noise level from the quietest frames.

    A percentile, not the minimum: one freak silent frame should not define the
    floor for the whole clip.
    """
    if energies.size == 0:
        return 0.0
    return float(np.percentile(energies, percentile))


def estimate_snr_db(samples: np.ndarray, sr: int = TARGET_SR) -> float:
    """Estimate the signal-to-noise ratio of a clip, in decibels.

    Energy-based and reference-free: the clip is cut into frames, the loud ones
    are taken to be speech and the quiet ones the noise floor, and the ratio
    between them is the estimate. No clean reference is needed, which is the
    point — at run time there is only the caller's audio.

    This measures *level* difference, so it reads lower than the SNR a mixer
    produced from a known clean signal; what matters is that it orders clips
    correctly. The result is capped at ``max_snr_db``, because a clean clip's
    quietest frames are near zero and the ratio would otherwise run away.
    """
    settings = get_settings()
    energies = frame_energy(samples, sr)
    if energies.size == 0:
        return 0.0

    noise = estimate_noise_floor(energies)
    speech = float(np.percentile(energies, SPEECH_PERCENTILE))

    if speech <= _EPS:
        return 0.0
    if noise <= _EPS:
        return settings.max_snr_db

    snr = 20.0 * float(np.log10(speech / noise))
    return float(min(max(snr, 0.0), settings.max_snr_db))


def estimate_snr(samples: np.ndarray, sr: int = TARGET_SR) -> float:
    """Estimate a clip's SNR in dB. Alias of :func:`estimate_snr_db`."""
    return estimate_snr_db(samples, sr)


# --------------------------------------------------------------------------
# The cheap companions
# --------------------------------------------------------------------------

def clipping_ratio(samples: np.ndarray, threshold: float = 0.99) -> float:
    """Return the fraction of samples sitting at or beyond full scale."""
    audio = np.asarray(samples, dtype="float32")
    if audio.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(audio) >= threshold) / audio.size)


def speech_ratio(samples: np.ndarray, sr: int = TARGET_SR) -> float:
    """Return the fraction of frames that look like speech rather than silence.

    A frame counts as speech when it sits :data:`SPEECH_MARGIN_DB` above the
    clip's own noise floor, so the measure travels with the recording level
    instead of against a fixed threshold.
    """
    energies = frame_energy(samples, sr)
    if energies.size == 0:
        return 0.0

    noise = estimate_noise_floor(energies)
    if noise <= _EPS:
        # Digital silence as a floor: anything audible is speech.
        return float(np.count_nonzero(energies > _EPS) / energies.size)

    gate = noise * (10.0 ** (SPEECH_MARGIN_DB / 20.0))
    return float(np.count_nonzero(energies > gate) / energies.size)


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------

def check_samples(
    samples: np.ndarray,
    sr: int = TARGET_SR,
    min_snr_db: float | None = None,
) -> AudioReport:
    """Run the full quality check on audio held in memory and return the verdict.

    The order of the reasons matters: a clip is reported as silent before it is
    reported as noisy, because "I heard nothing" and "I heard noise" ask the
    caller for different things.
    """
    settings = get_settings()
    floor = settings.min_snr_db if min_snr_db is None else min_snr_db

    audio = np.asarray(samples, dtype="float32")
    duration = audio.size / sr if sr else 0.0
    level = rms(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    clipped = clipping_ratio(audio)
    speech = speech_ratio(audio, sr)
    snr = estimate_snr_db(audio, sr)
    silent = level <= _EPS or speech < settings.min_speech_ratio

    # The order is the diagnosis, from most specific to least. Noise is checked
    # before emptiness on purpose: speech_ratio needs roughly SPEECH_MARGIN_DB
    # of headroom over the noise floor to find any speech at all, so a clip
    # buried in noise looks empty to it. Reporting "I heard nothing" for a clip
    # that is actually full of noise would send the caller the wrong line.
    reason: str | None = None
    if duration < settings.min_duration_s:
        reason = "too_short"
    elif level <= _EPS:
        reason = "silent"
    elif clipped > settings.max_clipping_ratio:
        reason = "clipping"
    elif snr < floor:
        reason = "low_snr"
    elif speech < settings.min_speech_ratio:
        reason = "no_speech"

    return AudioReport(
        snr_db=snr,
        duration_s=duration,
        rms=level,
        peak=peak,
        clipping_ratio=clipped,
        speech_ratio=speech,
        is_silent=silent,
        is_acceptable=reason is None,
        reason=reason,
    )


def check_audio(path: str | Path, min_snr_db: float | None = None) -> AudioReport:
    """Run the full quality check on a file and return the verdict."""
    samples, sr = load_audio(path)
    return check_samples(samples, sr, min_snr_db=min_snr_db)


def check(
    samples: np.ndarray,
    sr: int = TARGET_SR,
    min_snr_db: float | None = None,
) -> str:
    """Return ``"ok"`` if the clip is worth transcribing, else ``"repeat"``.

    The pipeline counts the ``"repeat"`` verdicts: the first asks the caller to
    say it again, the second hands the call to a human rather than making them
    repeat a third time.
    """
    return check_samples(samples, sr, min_snr_db=min_snr_db).verdict


#: What the agent says for each rejection reason, in French.
RETRY_LINES: dict[str, str] = {
    "too_short": (
        "Je n'ai rien entendu. Pouvez-vous répéter, un peu plus longuement ?"
    ),
    "silent": "Je ne vous entends pas. Êtes-vous toujours là ?",
    "no_speech": "Je ne vous entends pas bien. Pouvez-vous répéter ?",
    "clipping": (
        "Le son est saturé. Pouvez-vous vous éloigner un peu du téléphone et "
        "répéter ?"
    ),
    "low_snr": (
        "Il y a beaucoup de bruit autour de vous. Pouvez-vous répéter, "
        "s'il vous plaît ?"
    ),
}

DEFAULT_RETRY_LINE = "Pouvez-vous répéter, s'il vous plaît ?"


def retry_prompt(report: AudioReport) -> str:
    """Return the line the agent should speak when a clip is rejected."""
    if report.is_acceptable:
        return ""
    return RETRY_LINES.get(report.reason or "", DEFAULT_RETRY_LINE)
