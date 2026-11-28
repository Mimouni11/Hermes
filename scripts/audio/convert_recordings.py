"""Convert the recorded clips to the WAV format the manifest expects.

A phone or a laptop records to whatever it likes — `.m4a` at 48 kHz stereo, in
practice. `data/audio/manifest.csv` names a 16 kHz mono WAV per clip, which is
what Whisper consumes and what the TTS clips already are. This script walks the
manifest's ``recorded`` rows, finds whatever audio file sits next to the name it
wants, and writes the WAV.

Two backends, in order:

1. **ffmpeg**, if it is on PATH — exact control over rate, channels and bit
   depth, and it strips the source metadata.
2. **librosa + soundfile** otherwise, which reaches ffmpeg through audioread
   anyway but goes through a deprecated path.

Recorder metadata is dropped on purpose: a phone writes the device model and a
creation timestamp into an `.m4a`, and neither belongs in the repository
(Rule 3). The original files are left alone — nothing here deletes a recording.

    python scripts/audio/convert_recordings.py
    python scripts/audio/convert_recordings.py --force
    python scripts/audio/convert_recordings.py --normalize --peak-dbfs -3
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST = _PROJECT_ROOT / "data" / "audio" / "manifest.csv"

SAMPLE_RATE = 16_000

#: Extensions looked for, in preference order, when hunting for a recording.
SOURCE_EXTENSIONS = [
    ".m4a",
    ".mp3",
    ".aac",
    ".opus",
    ".ogg",
    ".flac",
    ".webm",
    ".mp4",
    ".3gp",
    ".amr",
    ".wma",
    ".aiff",
    ".caf",
]

#: Below this RMS a clip is almost certainly a failed take, not quiet speech.
QUIET_RMS_DBFS = -45.0

#: Above this peak a recording was probably recorded too hot and may be clipped.
HOT_PEAK = 0.999

#: How many consecutive full-scale samples count as real clipping.
CLIP_RUN = 3


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

@dataclass
class Converted:
    """What happened to one recording."""

    clip_id: str
    source: Path | None
    target: Path
    backend: str = ""
    seconds: float = 0.0
    source_rate: int = 0
    source_channels: int = 0
    peak: float = 0.0
    rms_dbfs: float = 0.0
    clipped_runs: int = 0
    gain: float = 1.0
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether the WAV was written."""
        return not self.error


# --------------------------------------------------------------------------
# Finding the sources
# --------------------------------------------------------------------------

def find_source(target: Path) -> Path | None:
    """Find the recording that should become ``target``.

    Looks for the same stem with any known audio extension, in the directory
    the manifest points at. A `.wav` already in place is only treated as a
    source when it is not the target itself.
    """
    for extension in SOURCE_EXTENSIONS:
        candidate = target.with_suffix(extension)
        if candidate.exists() and candidate.resolve() != target.resolve():
            return candidate
    return None


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

def probe_source(path: Path) -> tuple[int, int]:
    """Return the source's sample rate and channel count, or zeros if unknown."""
    try:
        info = sf.info(path)
        return info.samplerate, info.channels
    except Exception:  # noqa: BLE001 — not a libsndfile format, try audioread
        pass
    try:
        import audioread

        with audioread.audio_open(str(path)) as handle:
            return handle.samplerate, handle.channels
    except Exception:  # noqa: BLE001
        return 0, 0


def convert_with_ffmpeg(source: Path, target: Path, ffmpeg: str) -> None:
    """Convert to 16 kHz mono PCM_16 with ffmpeg, dropping all metadata."""
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-loglevel", "error",
            "-y",
            "-i", str(source),
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            "-map_metadata", "-1",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        raise RuntimeError(
            f"ffmpeg exited {result.returncode}: {detail[-1] if detail else 'no output'}"
        )


def convert_with_librosa(source: Path, target: Path) -> None:
    """Convert to 16 kHz mono PCM_16 with librosa, as the fallback."""
    import librosa

    audio, _ = librosa.load(str(source), sr=SAMPLE_RATE, mono=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, audio, SAMPLE_RATE, subtype="PCM_16")


def inspect(path: Path) -> tuple[float, float, float, int]:
    """Return a WAV's duration, peak, RMS in dBFS and its clipped-run count."""
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    seconds = audio.size / rate if rate else 0.0
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    rms_dbfs = 20 * float(np.log10(rms)) if rms > 0 else -999.0

    # Count runs of CLIP_RUN or more samples pinned at full scale: a single
    # sample touching 1.0 is normal, a run of them is distortion.
    full = np.abs(audio) >= 0.9999
    runs = 0
    length = 0
    for flag in full:
        if flag:
            length += 1
            if length == CLIP_RUN:
                runs += 1
        else:
            length = 0
    return seconds, peak, rms_dbfs, runs


def normalise(path: Path, peak_dbfs: float) -> float:
    """Scale a WAV to a target peak, returning the gain applied."""
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak <= 0:
        return 1.0
    gain = (10 ** (peak_dbfs / 20.0)) / peak
    sf.write(path, (audio * gain).astype("float32"), rate, subtype="PCM_16")
    return gain


def convert_row(
    row: dict[str, str],
    root: Path,
    ffmpeg: str | None,
    force: bool,
    do_normalise: bool,
    peak_dbfs: float,
) -> Converted:
    """Convert one manifest row's recording, and measure the result."""
    target = root / row["file"]
    result = Converted(clip_id=row["clip_id"], source=None, target=target)

    source = find_source(target)
    if source is None:
        if target.exists():
            result.backend = "already wav"
            (
                result.seconds,
                result.peak,
                result.rms_dbfs,
                result.clipped_runs,
            ) = inspect(target)
            return result
        result.error = "no recording found"
        return result

    result.source = source
    result.source_rate, result.source_channels = probe_source(source)

    if target.exists() and not force:
        result.error = "wav exists (use --force)"
        return result

    try:
        if ffmpeg is not None:
            convert_with_ffmpeg(source, target, ffmpeg)
            result.backend = "ffmpeg"
        else:
            convert_with_librosa(source, target)
            result.backend = "librosa"
    except Exception as exc:  # noqa: BLE001 — report per clip, keep going
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    if do_normalise:
        result.gain = normalise(target, peak_dbfs)

    (
        result.seconds,
        result.peak,
        result.rms_dbfs,
        result.clipped_runs,
    ) = inspect(target)
    return result


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------

def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read every row of the manifest."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the manifest path, overwrite flag and normalisation options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="the manifest naming the clips to convert")
    parser.add_argument("--source", type=str, default="recorded",
                        help="which manifest source to convert")
    parser.add_argument("--force", action="store_true",
                        help="overwrite WAVs that already exist")
    parser.add_argument("--normalize", "--normalise", dest="normalize",
                        action="store_true",
                        help="scale each clip to --peak-dbfs (off by default: "
                             "your recordings are not modified unless you ask)")
    parser.add_argument("--peak-dbfs", type=float, default=-3.0,
                        help="target peak when --normalize is given")
    parser.add_argument("--ffmpeg", type=str, default=None,
                        help="path to ffmpeg; default: found on PATH")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Convert every recorded row of the manifest and report clip by clip."""
    args = parse_args(argv)

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        print("run: python scripts/audio/pick_audio_sentences.py", file=sys.stderr)
        return 1

    rows = [r for r in read_manifest(args.manifest) if r.get("source") == args.source]
    if not rows:
        print(f"no rows with source {args.source!r} in the manifest", file=sys.stderr)
        return 1

    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    print(f"backend: {'ffmpeg at ' + ffmpeg if ffmpeg else 'librosa (ffmpeg not on PATH)'}")
    print(f"target format: {SAMPLE_RATE} Hz mono PCM_16")
    if args.normalize:
        print(f"normalising each clip to {args.peak_dbfs:+.1f} dBFS peak")
    print(f"converting {len(rows)} {args.source} clips\n")

    header = (
        f"{'clip':<9}{'source':<8}{'in':>12}{'sec':>7}{'peak':>7}"
        f"{'rms dB':>9}{'clip?':>7}"
    )
    print(header)
    print("-" * len(header))

    results = [
        convert_row(row, _PROJECT_ROOT, ffmpeg, args.force, args.normalize,
                    args.peak_dbfs)
        for row in rows
    ]

    for item in results:
        if not item.ok:
            print(f"{item.clip_id:<9}{'—':<8}{'':>12}{'':>7}{'':>7}{'':>9}{'':>7}"
                  f"  {item.error}")
            continue
        source_label = item.source.suffix.lstrip(".") if item.source else "wav"
        incoming = (
            f"{item.source_rate // 1000} kHz {item.source_channels}ch"
            if item.source_rate
            else "unknown"
        )
        warn = "YES" if item.clipped_runs else ""
        print(
            f"{item.clip_id:<9}{source_label:<8}{incoming:>12}{item.seconds:>7.2f}"
            f"{item.peak:>7.3f}{item.rms_dbfs:>9.1f}{warn:>7}"
        )

    print("-" * len(header))
    written = [r for r in results if r.ok and r.backend not in ("", "already wav")]
    skipped = [r for r in results if not r.ok]
    print(f"{len(written)} converted, {len(results) - len(written) - len(skipped)} "
          f"already WAV, {len(skipped)} skipped")

    quiet = [r for r in results if r.ok and r.rms_dbfs < QUIET_RMS_DBFS]
    hot = [r for r in results if r.ok and r.clipped_runs]
    if quiet:
        print(
            f"\nvery quiet, check these are not failed takes: "
            f"{', '.join(r.clip_id for r in quiet)}"
        )
    if hot:
        print(
            f"\nclipped (recorded too loud, distortion is already in the "
            f"recording): {', '.join(r.clip_id for r in hot)}"
        )
    if skipped:
        print(
            f"\n{len(skipped)} clips were skipped — see the reasons above.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
