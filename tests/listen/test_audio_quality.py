"""Tests for the audio front end: levels, the SNR estimate and the gate.

The SNR tests are ordering tests, not absolute ones. An energy-based estimate
with no clean reference cannot recover the SNR a mixer dialled in, so asserting
"this clip measures 5 dB" would be asserting a coincidence. What has to hold is
that a clean clip scores above its own noisy copy, and that the gate's verdict
follows from the thresholds in config.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hermes.listen import audio_quality
from hermes import config
from hermes.config import get_settings

SR = 16_000


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def test_loads_audio_as_mono_16k(sine_wav: Path) -> None:
    """A wav file loads as mono float32 at the target sample rate."""
    samples, sr = audio_quality.load_audio(sine_wav)

    assert sr == SR
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert samples.size > 0


def test_stereo_and_other_rates_are_folded_down(tmp_path: Path) -> None:
    """A 48 kHz stereo file comes back as mono at 16 kHz."""
    stereo = np.stack(
        [
            np.sin(2 * np.pi * 220 * np.arange(48_000) / 48_000),
            np.sin(2 * np.pi * 330 * np.arange(48_000) / 48_000),
        ],
        axis=1,
    ).astype("float32")
    path = tmp_path / "stereo48k.wav"
    sf.write(path, stereo, 48_000, subtype="PCM_16")

    samples, sr = audio_quality.load_audio(path)

    assert sr == SR
    assert samples.ndim == 1
    assert abs(samples.size - 16_000) < 200  # one second, give or take the filter


# --------------------------------------------------------------------------
# normalize
# --------------------------------------------------------------------------

def test_normalize_hits_the_target_rms() -> None:
    """A quiet clip comes back at the configured RMS."""
    settings = get_settings()
    quiet = (np.sin(2 * np.pi * 200 * np.arange(SR) / SR) * 0.01).astype("float32")

    loud = audio_quality.normalize(quiet)

    assert audio_quality.dbfs(audio_quality.rms(loud)) == pytest.approx(
        settings.target_rms_dbfs, abs=0.1
    )


def test_normalize_raises_a_quiet_clip_and_lowers_a_loud_one() -> None:
    """Normalising works in both directions, not just upwards."""
    settings = get_settings()
    target = settings.target_rms_dbfs

    quiet = (np.sin(2 * np.pi * 200 * np.arange(SR) / SR) * 0.005).astype("float32")
    loud = (np.sin(2 * np.pi * 200 * np.arange(SR) / SR) * 0.9).astype("float32")

    assert audio_quality.dbfs(audio_quality.rms(quiet)) < target
    assert audio_quality.dbfs(audio_quality.rms(loud)) > target

    for clip in (quiet, loud):
        fixed = audio_quality.normalize(clip)
        assert audio_quality.dbfs(audio_quality.rms(fixed)) == pytest.approx(
            target, abs=0.1
        )


def test_normalize_never_clips() -> None:
    """The peak limiter holds even when the gain would push past full scale.

    A clip with one sharp transient and a quiet body needs a large gain to reach
    the target RMS — enough to send the transient well past 1.0. The limiter has
    to win that fight, and the RMS ends below target as the cost.
    """
    settings = get_settings()
    spiky = np.full(SR, 0.001, dtype="float32")
    spiky[SR // 2] = 0.95

    fixed = audio_quality.normalize(spiky)

    assert float(np.max(np.abs(fixed))) <= settings.peak_ceiling + 1e-6
    # Nothing at or beyond full scale. The limiter parks the loudest sample on
    # peak_ceiling itself, so the threshold here has to be above it.
    assert audio_quality.clipping_ratio(fixed, threshold=1.0) == 0.0
    assert audio_quality.dbfs(audio_quality.rms(fixed)) < settings.target_rms_dbfs


def test_normalize_leaves_silence_alone() -> None:
    """Silence has no level to scale, and must not become noise or NaN."""
    silence = np.zeros(SR, dtype="float32")

    fixed = audio_quality.normalize(silence)

    assert np.all(fixed == 0.0)
    assert not np.any(np.isnan(fixed))


def test_normalize_does_not_change_an_empty_clip() -> None:
    """An empty array is returned as-is rather than dividing by zero."""
    assert audio_quality.normalize(np.zeros(0, dtype="float32")).size == 0


# --------------------------------------------------------------------------
# estimate_snr
# --------------------------------------------------------------------------

def test_clean_clip_has_high_snr(sine_wav: Path) -> None:
    """A clean clip scores well above the minimum SNR."""
    settings = get_settings()
    samples, sr = audio_quality.load_audio(sine_wav)

    assert audio_quality.estimate_snr(samples, sr) > settings.min_snr_db


def test_a_clean_clip_beats_its_own_noisy_copy(sine_wav: Path, noisy_wav: Path) -> None:
    """The estimate orders a clip above its noisy version. This is the point."""
    clean, sr = audio_quality.load_audio(sine_wav)
    noisy, _ = audio_quality.load_audio(noisy_wav)

    assert audio_quality.estimate_snr(clean, sr) > audio_quality.estimate_snr(noisy, sr)


def test_snr_falls_as_noise_rises(sine_wav: Path) -> None:
    """Mixing in progressively more noise drives the estimate monotonically down."""
    clean, sr = audio_quality.load_audio(sine_wav)
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(clean.size).astype("float32")
    clean_rms = audio_quality.rms(clean)
    noise_rms = audio_quality.rms(noise)

    estimates = []
    for snr_db in (20, 10, 5):
        scaled = noise * (clean_rms / (10 ** (snr_db / 20.0)) / noise_rms)
        estimates.append(audio_quality.estimate_snr(clean + scaled, sr))

    assert estimates == sorted(estimates, reverse=True), estimates


def test_real_noisy_clip_scores_below_its_clean_source() -> None:
    """The same ordering holds on the actual test set, if it has been built.

    This is the assertion the brief asked for — a clean clip against its own
    5 dB copy — checked against `data/audio` rather than a synthetic fixture.
    Skipped when the clips have not been generated.
    """
    root = Path(__file__).resolve().parents[2]
    clean_path = root / "data" / "audio" / "tts" / "fr-003.wav"
    noisy_path = root / "data" / "audio" / "noisy" / "fr-003-snr5.wav"
    if not (clean_path.exists() and noisy_path.exists()):
        pytest.skip("test audio set not built; run the audio scripts first")

    clean, sr = audio_quality.load_audio(clean_path)
    noisy, _ = audio_quality.load_audio(noisy_path)

    assert audio_quality.estimate_snr(clean, sr) > audio_quality.estimate_snr(noisy, sr)


def test_silence_has_no_snr() -> None:
    """An all-zero clip scores 0 dB rather than dividing by zero."""
    assert audio_quality.estimate_snr(np.zeros(SR, dtype="float32"), SR) == 0.0


def test_snr_is_capped() -> None:
    """A digitally clean signal is capped, not returned as infinity."""
    settings = get_settings()
    tone = np.sin(2 * np.pi * 200 * np.arange(SR) / SR).astype("float32")

    assert audio_quality.estimate_snr(tone, SR) <= settings.max_snr_db


# --------------------------------------------------------------------------
# The cheap companions
# --------------------------------------------------------------------------

def test_clipping_is_measured(tmp_path: Path) -> None:
    """A saturated clip reports a non-zero clipping ratio."""
    saturated = np.ones(SR, dtype="float32")
    path = tmp_path / "saturated.wav"
    sf.write(path, saturated, SR, subtype="PCM_16")

    samples, _ = audio_quality.load_audio(path)

    assert audio_quality.clipping_ratio(samples) > 0.5


def test_a_clean_clip_is_not_reported_as_clipping(sine_wav: Path) -> None:
    """Normal audio reports no clipping at all."""
    samples, _ = audio_quality.load_audio(sine_wav)

    assert audio_quality.clipping_ratio(samples) == 0.0


def test_speech_ratio_separates_speech_from_silence(sine_wav: Path) -> None:
    """A gated tone has speech in some frames and not others."""
    samples, sr = audio_quality.load_audio(sine_wav)

    ratio = audio_quality.speech_ratio(samples, sr)

    assert 0.0 < ratio < 1.0


def test_silence_has_no_speech() -> None:
    """Digital silence contains no speech frames."""
    assert audio_quality.speech_ratio(np.zeros(SR, dtype="float32"), SR) == 0.0


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------

def test_check_accepts_a_clean_clip(sine_wav: Path) -> None:
    """A clean clip returns "ok"."""
    samples, sr = audio_quality.load_audio(sine_wav)

    assert audio_quality.check(samples, sr) == "ok"


def test_noisy_clip_is_rejected(noisy_wav: Path) -> None:
    """A clip below the SNR threshold is marked unacceptable with a reason."""
    samples, sr = audio_quality.load_audio(noisy_wav)

    report = audio_quality.check_samples(samples, sr)

    assert report.is_acceptable is False
    assert report.reason == "low_snr"
    assert report.verdict == "repeat"


def test_check_returns_only_ok_or_repeat(sine_wav: Path, noisy_wav: Path) -> None:
    """The gate's vocabulary is exactly two words."""
    clean, sr = audio_quality.load_audio(sine_wav)
    noisy, _ = audio_quality.load_audio(noisy_wav)

    assert {audio_quality.check(clean, sr), audio_quality.check(noisy, sr)} <= {
        "ok",
        "repeat",
    }


def test_silence_is_detected(tmp_path: Path) -> None:
    """An all-zero clip is reported as silent rather than as clean."""
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(SR, dtype="float32"), SR, subtype="PCM_16")

    report = audio_quality.check_audio(path)

    assert report.is_silent is True
    assert report.is_acceptable is False
    assert report.reason == "silent"


def test_a_very_short_clip_is_rejected_before_anything_else() -> None:
    """Too short to transcribe is its own reason, checked first."""
    settings = get_settings()
    samples = np.zeros(int(settings.min_duration_s * SR / 2), dtype="float32")

    report = audio_quality.check_samples(samples, SR)

    assert report.reason == "too_short"


def test_a_clipped_clip_is_rejected_for_clipping() -> None:
    """Saturation is reported as clipping, not as low SNR."""
    saturated = np.ones(SR, dtype="float32")
    saturated[::2] = -1.0

    report = audio_quality.check_samples(saturated, SR)

    assert report.is_acceptable is False
    assert report.reason == "clipping"


def test_the_snr_threshold_is_read_from_config(noisy_wav: Path) -> None:
    """Changing min_snr_db in config changes which reason the gate reports.

    Proof that the threshold is configuration and not a constant in the module.
    The verdict stays "repeat" either way: this fixture is buried at 0 dB, so
    once the SNR floor stops rejecting it, `speech_ratio` does — there is no
    speech standing clear of that much noise. What the threshold controls is
    the diagnosis, which is what gets asserted.
    """
    samples, sr = audio_quality.load_audio(noisy_wav)

    assert audio_quality.check_samples(samples, sr).reason == "low_snr"

    relaxed = config.with_overrides(min_snr_db=0.0)
    config.set_settings(relaxed)
    try:
        assert audio_quality.check_samples(samples, sr).reason != "low_snr"
    finally:
        config.reset_settings()


def test_an_argument_overrides_the_configured_floor(noisy_wav: Path) -> None:
    """check() takes an explicit min_snr_db, for calibration sweeps."""
    samples, sr = audio_quality.load_audio(noisy_wav)

    assert audio_quality.check_samples(samples, sr).reason == "low_snr"
    assert audio_quality.check_samples(samples, sr, min_snr_db=0.0).reason != "low_snr"


# --------------------------------------------------------------------------
# The retry line
# --------------------------------------------------------------------------

def test_rejected_clip_gets_retry_prompt(noisy_wav: Path) -> None:
    """A rejected report produces a spoken retry line."""
    samples, sr = audio_quality.load_audio(noisy_wav)

    report = audio_quality.check_samples(samples, sr)
    line = audio_quality.retry_prompt(report)

    assert line
    assert "répéter" in line


def test_an_accepted_clip_gets_no_retry_line(sine_wav: Path) -> None:
    """Nothing is spoken when the clip was fine."""
    samples, sr = audio_quality.load_audio(sine_wav)

    report = audio_quality.check_samples(samples, sr)

    assert audio_quality.retry_prompt(report) == ""


def test_every_rejection_reason_has_a_line() -> None:
    """No reason can be produced that the agent has no words for."""
    reasons = {"too_short", "silent", "no_speech", "clipping", "low_snr"}

    assert reasons <= set(audio_quality.RETRY_LINES)
