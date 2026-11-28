"""Add noisy copies of some clips to the test audio set.

Takes five clips from `data/audio/manifest.csv`, mixes each with noise at
SNR 20, 10 and 5 dB, writes the results to `data/audio/noisy/` and adds a row
per noisy clip to the manifest.

A call line is noisy: a speakerphone in a waiting room, a bad mobile connection,
somebody else talking nearby. Three SNRs give three difficulties — 20 dB is a
quiet room, 10 dB is a busy one, 5 dB is the case where a human would also ask
the caller to repeat themselves.

The noise is whatever `data/audio/noise/*.wav` holds, or, if that is empty,
synthetic babble generated here from a fixed seed and saved to
`data/audio/noise/babble.wav`, so a later run mixes the identical noise.

The script is idempotent: it regenerates every noisy row rather than appending
to what is already there, so running it twice does not grow the manifest.

    python scripts/audio/add_noise.py
    python scripts/audio/add_noise.py --clips 5 --snr 20 10 5 --seed 7
    python scripts/audio/add_noise.py --noise data/audio/noise/waiting_room.wav
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST = _PROJECT_ROOT / "data" / "audio" / "manifest.csv"

SAMPLE_RATE = 16_000

DEFAULT_SEED = 7
DEFAULT_CLIP_COUNT = 5
DEFAULT_SNRS = [20, 10, 5]

#: Leave this much headroom when a mix would otherwise clip.
PEAK_CEILING = 0.99

#: How much babble to generate, in seconds — long enough that each clip can take
#: a different segment of it.
BABBLE_SECONDS = 30

#: How many overlapping "voices" the synthetic babble sums.
BABBLE_VOICES = 8

#: The band synthetic babble is limited to, roughly telephone speech.
BABBLE_BAND_HZ = (300.0, 3400.0)

MANIFEST_COLUMNS = [
    "clip_id",
    "text",
    "lang",
    "intent",
    "exam_id",
    "source",
    "noise_snr_db",
    "file",
]


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

@dataclass
class Mixed:
    """One noisy clip that was written."""

    clip_id: str
    source_clip: str
    snr_db: int
    path: Path
    seconds: float
    gain_applied: float
    measured_snr_db: float


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------

def _bandpass(audio: np.ndarray, rate: int, low: float, high: float) -> np.ndarray:
    """Band-limit a signal, falling back to the raw signal without SciPy."""
    try:
        from scipy.signal import butter, sosfilt
    except ImportError:  # pragma: no cover - SciPy is a declared dependency
        return audio
    nyquist = rate / 2.0
    sos = butter(
        4,
        [max(low / nyquist, 1e-4), min(high / nyquist, 0.999)],
        btype="band",
        output="sos",
    )
    return np.asarray(sosfilt(sos, audio), dtype="float32")


def make_babble(seconds: int, rate: int, seed: int) -> np.ndarray:
    """Generate synthetic babble: several band-limited, slowly modulated voices.

    Not a substitute for a real recording of a waiting room, but deterministic,
    free, and far more speech-like than white noise — its energy sits in the same
    band as the speech it masks, which is what makes an ASR model struggle.
    """
    rng = np.random.default_rng(seed)
    samples = int(seconds * rate)
    mixture = np.zeros(samples, dtype="float32")

    for _ in range(BABBLE_VOICES):
        voice = rng.standard_normal(samples).astype("float32")
        voice = _bandpass(voice, rate, *BABBLE_BAND_HZ)
        # A slow envelope, so the babble swells and fades like real speech
        # instead of sitting at one level.
        envelope_rate = 4.0  # Hz, roughly syllable speed
        points = int(seconds * envelope_rate) + 1
        coarse = rng.uniform(0.2, 1.0, size=points).astype("float32")
        envelope = np.interp(
            np.linspace(0, points - 1, samples),
            np.arange(points),
            coarse,
        ).astype("float32")
        mixture += voice * envelope

    peak = float(np.abs(mixture).max())
    if peak > 0:
        mixture /= peak
    return mixture


def load_or_make_noise(
    noise_arg: Path | None,
    noise_dir: Path,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Return the noise signal and a label saying where it came from."""
    if noise_arg is not None:
        audio, rate = sf.read(noise_arg, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
        return audio.astype("float32"), noise_arg.name

    existing = sorted(noise_dir.glob("*.wav")) if noise_dir.is_dir() else []
    if existing:
        return load_or_make_noise(existing[0], noise_dir, seed)

    noise_dir.mkdir(parents=True, exist_ok=True)
    babble = make_babble(BABBLE_SECONDS, SAMPLE_RATE, seed)
    target = noise_dir / "babble.wav"
    sf.write(target, babble, SAMPLE_RATE, subtype="PCM_16")
    print(f"generated synthetic babble: {target}")
    return babble, target.name


# --------------------------------------------------------------------------
# Mixing
# --------------------------------------------------------------------------

def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def noise_segment(
    noise: np.ndarray,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Take a ``length``-sample segment of the noise, tiling it if it is short."""
    if noise.size < length:
        repeats = int(np.ceil(length / max(noise.size, 1)))
        noise = np.tile(noise, repeats)
    offset = int(rng.integers(0, max(noise.size - length, 1)))
    return noise[offset : offset + length].copy()


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    snr_db: int,
) -> tuple[np.ndarray, float, float]:
    """Mix speech and noise at a target SNR.

    Returns the mix, the gain applied to avoid clipping (1.0 if none was
    needed), and the SNR actually measured on the result — which is the number
    worth trusting, since the anti-clipping gain scales speech and noise
    together and so leaves the ratio intact.

    The SNR is computed on whole-clip RMS. The clips are tightly trimmed, so
    this is within a fraction of a dB of an active-speech measurement; it would
    not be if they carried long leading silences.
    """
    speech_rms = _rms(speech)
    noise_rms = _rms(noise)
    if speech_rms <= 0 or noise_rms <= 0:
        raise ValueError("cannot set an SNR against a silent signal")

    wanted_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    scaled_noise = noise * (wanted_noise_rms / noise_rms)

    mixed = speech + scaled_noise

    gain = 1.0
    peak = float(np.abs(mixed).max())
    if peak > PEAK_CEILING:
        gain = PEAK_CEILING / peak
        mixed = mixed * gain
        scaled_noise = scaled_noise * gain

    measured = 20 * np.log10(_rms(speech * gain) / _rms(scaled_noise))
    return mixed.astype("float32"), gain, float(measured)


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------

def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read every row of the manifest."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    """Write the manifest back, columns in their canonical order."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS})


def choose_clips(
    rows: list[dict[str, str]],
    root: Path,
    count: int,
    rng: np.random.Generator,
) -> list[dict[str, str]]:
    """Pick which clips get noisy copies, spreading them across intents.

    Only clips whose audio exists can be mixed, so a set where the recorded half
    is still missing simply draws from what has been generated so far.
    """
    usable = [
        row
        for row in rows
        if row.get("source") in ("tts", "recorded") and (root / row["file"]).exists()
    ]
    if not usable:
        return []

    by_intent: dict[str, list[dict[str, str]]] = {}
    for row in usable:
        by_intent.setdefault(row.get("intent", ""), []).append(row)
    for pool in by_intent.values():
        pool.sort(key=lambda r: r["clip_id"])

    chosen: list[dict[str, str]] = []
    intents = sorted(by_intent)
    while len(chosen) < count and any(by_intent[i] for i in intents):
        for intent in intents:
            if len(chosen) >= count:
                break
            if by_intent[intent]:
                index = int(rng.integers(0, len(by_intent[intent])))
                chosen.append(by_intent[intent].pop(index))
    return sorted(chosen, key=lambda r: r["clip_id"])


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the manifest path, how many clips, which SNRs and the seed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="the manifest to read and extend")
    parser.add_argument("--clips", type=int, default=DEFAULT_CLIP_COUNT,
                        help="how many clips get noisy copies")
    parser.add_argument("--snr", type=int, nargs="+", default=DEFAULT_SNRS,
                        help="target signal-to-noise ratios in dB")
    parser.add_argument("--noise", type=Path, default=None,
                        help="a noise WAV; default: data/audio/noise, else babble")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="fixed seed for clip choice and babble generation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Mix the noisy clips, write them, and update the manifest."""
    args = parse_args(argv)

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        print("run: python scripts/audio/pick_audio_sentences.py", file=sys.stderr)
        return 1

    audio_dir = args.manifest.parent
    rows = read_manifest(args.manifest)
    clean = [row for row in rows if row.get("source") != "noisy"]

    rng = np.random.default_rng(args.seed)
    chosen = choose_clips(clean, _PROJECT_ROOT, args.clips, rng)
    if not chosen:
        print(
            "no clips with audio on disk yet — run make_tts_clips.py, or record "
            "the sentences, then run this again",
            file=sys.stderr,
        )
        return 1
    if len(chosen) < args.clips:
        print(
            f"only {len(chosen)} clips have audio on disk, {args.clips} requested "
            "— mixing what exists"
        )

    noise, noise_label = load_or_make_noise(args.noise, audio_dir / "noise", args.seed)
    print(f"noise: {noise_label} ({noise.size / SAMPLE_RATE:.1f} s)")
    print(f"mixing {len(chosen)} clips at SNR {', '.join(f'{s} dB' for s in args.snr)}\n")

    header = f"{'noisy clip':<18}{'from':<9}{'snr':>6}{'measured':>10}{'gain':>7}{'sec':>7}"
    print(header)
    print("-" * len(header))

    noisy_dir = audio_dir / "noisy"
    noisy_dir.mkdir(parents=True, exist_ok=True)

    written: list[Mixed] = []
    new_rows: list[dict[str, str]] = []
    for row in chosen:
        speech, rate = sf.read(_PROJECT_ROOT / row["file"], dtype="float32")
        if rate != SAMPLE_RATE:
            print(f"  {row['clip_id']}: unexpected rate {rate}, skipped", file=sys.stderr)
            continue
        for snr_db in args.snr:
            segment = noise_segment(noise, speech.size, rng)
            try:
                mixed, gain, measured = mix_at_snr(speech, segment, snr_db)
            except ValueError as exc:
                print(f"  {row['clip_id']} @ {snr_db} dB: {exc}", file=sys.stderr)
                continue

            clip_id = f"{row['clip_id']}-snr{snr_db}"
            relative = f"data/audio/noisy/{clip_id}.wav"
            sf.write(_PROJECT_ROOT / relative, mixed, SAMPLE_RATE, subtype="PCM_16")

            written.append(
                Mixed(
                    clip_id=clip_id,
                    source_clip=row["clip_id"],
                    snr_db=snr_db,
                    path=_PROJECT_ROOT / relative,
                    seconds=mixed.size / SAMPLE_RATE,
                    gain_applied=gain,
                    measured_snr_db=measured,
                )
            )
            new_rows.append(
                {
                    "clip_id": clip_id,
                    "text": row["text"],
                    "lang": row["lang"],
                    "intent": row["intent"],
                    "exam_id": row["exam_id"],
                    "source": "noisy",
                    "noise_snr_db": str(snr_db),
                    "file": relative,
                }
            )
            print(
                f"{clip_id:<18}{row['clip_id']:<9}{snr_db:>6}"
                f"{measured:>10.2f}{gain:>7.3f}{mixed.size / SAMPLE_RATE:>7.2f}"
            )

    print("-" * len(header))
    write_manifest(clean + new_rows, args.manifest)
    print(f"{len(written)} noisy clips written to {noisy_dir}")
    print(f"manifest now {len(clean) + len(new_rows)} rows: "
          f"{len(clean)} clean + {len(new_rows)} noisy")

    attenuated = [m for m in written if m.gain_applied < 1.0]
    if attenuated:
        print(
            f"\n{len(attenuated)} mixes were scaled down to stay below full scale "
            "(the SNR is unchanged — speech and noise scale together)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
