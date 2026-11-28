"""Synthesise the TTS half of the test audio set.

Reads `data/audio/manifest.csv`, takes the rows whose ``source`` is ``tts`` and
writes one 16 kHz mono WAV per row, at the path the manifest already names.

Two engines, in order:

1. **Piper** — runs locally, needs no network, and sounds closest to a French
   speaker. It needs a voice model on disk: put ``*.onnx`` and its
   ``*.onnx.json`` in `models/piper/` and they are found automatically.
   Free voices: https://huggingface.co/rhasspy/piper-voices (fr_FR/*).
2. **gTTS** — the fallback when no Piper voice is installed. It sends each
   sentence to Google's public endpoint, so it needs a network connection.

Both are resampled to 16 kHz mono, because that is what Whisper consumes and
what the recorded clips are specified at: one sample rate across the set.

    python scripts/audio/make_tts_clips.py
    python scripts/audio/make_tts_clips.py --engine gtts --force
    python scripts/audio/make_tts_clips.py --voice models/piper/fr_FR-siwis-medium.onnx
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST = _PROJECT_ROOT / "data" / "audio" / "manifest.csv"
DEFAULT_VOICE_DIR = _PROJECT_ROOT / "models" / "piper"

#: Whisper's native rate, and the rate the recorded clips are specified at.
SAMPLE_RATE = 16_000

#: The language code both engines take.
LANGUAGE = "fr"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

@dataclass
class Result:
    """What happened to one clip."""

    clip_id: str
    engine: str
    path: Path | None
    seconds: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether the clip was written."""
        return self.path is not None and not self.error


# --------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------

def write_16k_mono(samples, source_rate: int, path: Path) -> float:
    """Resample to 16 kHz mono, write a WAV, and return its duration."""
    import librosa
    import numpy as np
    import soundfile as sf

    audio = np.asarray(samples, dtype="float32")
    if audio.ndim > 1:
        audio = librosa.to_mono(audio.T if audio.shape[0] > audio.shape[1] else audio)
    if source_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=SAMPLE_RATE)

    peak = float(abs(audio).max()) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
    return len(audio) / SAMPLE_RATE


# --------------------------------------------------------------------------
# Piper
# --------------------------------------------------------------------------

def find_piper_voice(explicit: Path | None, voice_dir: Path) -> Path | None:
    """Return the Piper voice model to use, or None if none is installed."""
    if explicit is not None:
        return explicit if explicit.exists() else None
    if not voice_dir.is_dir():
        return None
    voices = sorted(voice_dir.glob("*.onnx"))
    return voices[0] if voices else None


def synthesise_piper(text: str, path: Path, voice_path: Path) -> float:
    """Synthesise one sentence with Piper and write it at 16 kHz mono."""
    import numpy as np
    from piper.voice import PiperVoice

    voice = PiperVoice.load(str(voice_path))

    with tempfile.TemporaryDirectory() as workspace:
        raw = Path(workspace) / "piper.wav"
        with wave.open(str(raw), "wb") as handle:
            # The API was renamed between Piper releases; both write a WAV.
            synth = getattr(voice, "synthesize_wav", None) or voice.synthesize
            synth(text, handle)

        with wave.open(str(raw), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            frames = handle.readframes(handle.getnframes())

    audio = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return write_16k_mono(audio, rate, path)


# --------------------------------------------------------------------------
# gTTS
# --------------------------------------------------------------------------

def synthesise_gtts(text: str, path: Path) -> float:
    """Synthesise one sentence with gTTS and write it at 16 kHz mono.

    gTTS returns MP3, so it is decoded and resampled rather than written
    straight out. This call leaves the machine: the sentence is sent to
    Google's public translate endpoint.
    """
    import librosa
    from gtts import gTTS

    with tempfile.TemporaryDirectory() as workspace:
        mp3 = Path(workspace) / "gtts.mp3"
        gTTS(text=text, lang=LANGUAGE).save(str(mp3))
        audio, rate = librosa.load(str(mp3), sr=SAMPLE_RATE, mono=True)

    return write_16k_mono(audio, rate, path)


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------

def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read every row of the manifest."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------

def synthesise_row(
    row: dict[str, str],
    root: Path,
    engine: str,
    voice_path: Path | None,
    force: bool,
) -> Result:
    """Write one clip, choosing the engine and reporting what happened."""
    clip_id = row["clip_id"]
    target = root / row["file"]

    if target.exists() and not force:
        return Result(clip_id, "skipped", target, error="exists (use --force)")

    attempts: list[str] = []
    if engine in ("auto", "piper") and voice_path is not None:
        attempts.append("piper")
    if engine in ("auto", "gtts"):
        attempts.append("gtts")
    if engine == "piper" and voice_path is None:
        return Result(clip_id, "piper", None, error="no Piper voice model found")

    last_error = "no engine available"
    for attempt in attempts:
        try:
            if attempt == "piper":
                assert voice_path is not None
                seconds = synthesise_piper(row["text"], target, voice_path)
            else:
                seconds = synthesise_gtts(row["text"], target)
            return Result(clip_id, attempt, target, seconds=seconds)
        except Exception as exc:  # noqa: BLE001 — report and try the next engine
            last_error = f"{type(exc).__name__}: {exc}"
    return Result(clip_id, attempts[-1] if attempts else engine, None, error=last_error)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the manifest path, engine choice, voice model and overwrite flag."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="the manifest written by pick_audio_sentences.py")
    parser.add_argument("--engine", choices=["auto", "piper", "gtts"], default="auto",
                        help="which engine to use (auto tries Piper, then gTTS)")
    parser.add_argument("--voice", type=Path, default=None,
                        help="a Piper .onnx voice model; default: first in models/piper")
    parser.add_argument("--voice-dir", type=Path, default=DEFAULT_VOICE_DIR,
                        help="where to look for Piper voices")
    parser.add_argument("--force", action="store_true",
                        help="overwrite clips that already exist")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Synthesise every TTS row of the manifest and report clip by clip."""
    args = parse_args(argv)

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        print("run: python scripts/audio/pick_audio_sentences.py", file=sys.stderr)
        return 1

    rows = [r for r in read_manifest(args.manifest) if r.get("source") == "tts"]
    if not rows:
        print("no rows with source 'tts' in the manifest", file=sys.stderr)
        return 1

    voice_path = find_piper_voice(args.voice, args.voice_dir)
    if voice_path is not None:
        print(f"Piper voice: {voice_path}")
    elif args.engine == "piper":
        print(f"no Piper voice in {args.voice_dir}", file=sys.stderr)
        return 1
    else:
        print(f"no Piper voice in {args.voice_dir} — falling back to gTTS (network)")

    print(f"synthesising {len(rows)} clips at {SAMPLE_RATE} Hz mono\n")
    header = f"{'clip':<10}{'engine':<10}{'seconds':>9}  file"
    print(header)
    print("-" * len(header))

    results = [
        synthesise_row(row, _PROJECT_ROOT, args.engine, voice_path, args.force)
        for row in rows
    ]
    for result in results:
        if result.ok:
            print(
                f"{result.clip_id:<10}{result.engine:<10}{result.seconds:>9.2f}  "
                f"{result.path.relative_to(_PROJECT_ROOT) if result.path else ''}"
            )
        else:
            print(f"{result.clip_id:<10}{result.engine:<10}{'—':>9}  {result.error}")

    written = [r for r in results if r.ok]
    print("-" * len(header))
    print(f"{len(written)} of {len(rows)} clips written")

    failed = [r for r in results if not r.ok and "exists" not in r.error]
    if failed:
        print(
            "\nSome clips failed. If this is a network error, install a Piper "
            "voice for offline synthesis:",
            file=sys.stderr,
        )
        print(
            "  download a fr_FR voice (.onnx and .onnx.json) from "
            "https://huggingface.co/rhasspy/piper-voices into models/piper/",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
