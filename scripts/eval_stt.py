"""Evaluate the speech-to-text front end over the whole test audio set.

Runs each Whisper model size over every clip in `data/audio/manifest.csv`, twice
— once on the audio as it is, once after :func:`hermes.listen.audio_quality.normalize`
— and writes one row per clip x model x audio variant to
`reports/stt_results.csv`, then prints the summary.

What the comparison is for:

- **model size**: "small" against "base". If the gap is small, the project ships
  "base" and the demo answers faster on a laptop.
- **raw against normalized**: whether the level fix in the audio front end earns
  its place, or whether Whisper's own normalisation already handles it.
- **by source**: synthetic speech is easy, a real voice is harder, and a real
  voice at 5 dB SNR is the case that decides whether the gate is needed.

The first run downloads the model weights, which is the only step that needs
the network. Nothing else here leaves the machine.

    python scripts/eval_stt.py
    python scripts/eval_stt.py --models base
    python scripts/eval_stt.py --models small base --variants raw normalized
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hermes.listen import audio_quality  # noqa: E402
from hermes.config import get_settings  # noqa: E402
from hermes.listen.stt import STT, normalise_for_wer  # noqa: E402

DEFAULT_MODELS = ["small", "base"]
DEFAULT_VARIANTS = ["raw", "normalized"]

RESULT_COLUMNS = [
    "clip_id",
    "source",
    "bucket",
    "noise_snr_db",
    "intent",
    "model",
    "variant",
    "reference",
    "hypothesis",
    "wer",
    "confidence",
    "no_speech_prob",
    "duration_s",
    "latency_s",
    "snr_estimate_db",
    "gate_verdict",
    "gate_reason",
]


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

@dataclass
class Row:
    """One evaluated clip, for one model and one audio variant."""

    clip_id: str
    source: str
    bucket: str
    noise_snr_db: str
    intent: str
    model: str
    variant: str
    reference: str
    hypothesis: str
    wer: float
    confidence: float
    no_speech_prob: float
    duration_s: float
    latency_s: float
    snr_estimate_db: float
    gate_verdict: str
    gate_reason: str

    def as_dict(self) -> dict[str, object]:
        """Render as a CSV row, rounding the floats to something readable."""
        return {
            "clip_id": self.clip_id,
            "source": self.source,
            "bucket": self.bucket,
            "noise_snr_db": self.noise_snr_db,
            "intent": self.intent,
            "model": self.model,
            "variant": self.variant,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
            "wer": round(self.wer, 4),
            "confidence": round(self.confidence, 4),
            "no_speech_prob": round(self.no_speech_prob, 4),
            "duration_s": round(self.duration_s, 3),
            "latency_s": round(self.latency_s, 3),
            "snr_estimate_db": round(self.snr_estimate_db, 2),
            "gate_verdict": self.gate_verdict,
            "gate_reason": self.gate_reason,
        }


@dataclass
class Bucket:
    """Accumulated WER and latency for one model x source bucket."""

    wers: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)

    def add(self, wer: float, latency: float) -> None:
        self.wers.append(wer)
        self.latencies.append(latency)

    @property
    def mean_wer(self) -> float:
        return float(np.mean(self.wers)) if self.wers else float("nan")

    @property
    def mean_latency(self) -> float:
        return float(np.mean(self.latencies)) if self.latencies else float("nan")


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------

def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read every row of the manifest."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bucket_of(row: dict[str, str]) -> str:
    """Group a clip for the summary: tts, recorded, or noisy at its SNR."""
    source = row.get("source", "")
    if source == "noisy":
        snr = (row.get("noise_snr_db") or "?").strip()
        return f"noisy {snr} dB"
    return source


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate between two already-normalised strings.

    Uses jiwer when it is installed, and falls back to a plain Levenshtein
    distance over words otherwise, so the script still runs on a machine where
    the dependency has not been synced yet. Both give the same number.
    """
    try:
        import jiwer

        return float(jiwer.wer(reference, hypothesis))
    except ImportError:
        return _fallback_wer(reference, hypothesis)


def _fallback_wer(reference: str, hypothesis: str) -> float:
    """Word-level edit distance over the reference length."""
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0

    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1] / len(ref)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate(
    manifest_rows: list[dict[str, str]],
    root: Path,
    models: list[str],
    variants: list[str],
) -> list[Row]:
    """Transcribe every clip with every model and variant, and score it."""
    results: list[Row] = []

    clips: list[tuple[dict[str, str], np.ndarray, int]] = []
    for row in manifest_rows:
        path = root / row["file"]
        if not path.exists():
            print(f"  {row['clip_id']}: no audio at {row['file']}, skipped")
            continue
        samples, sr = audio_quality.load_audio(path)
        clips.append((row, samples, sr))

    if not clips:
        return results

    total = len(clips) * len(variants) * len(models)
    print(f"{len(clips)} clips with audio on disk, {total} transcriptions to run\n")

    done = 0
    started = time.perf_counter()

    for model_size in models:
        print(f"loading {model_size}... (first run downloads the weights)", flush=True)
        engine = STT(model_size=model_size)
        # Force the load now, so the download does not land inside the first
        # clip's latency measurement.
        _ = engine.model
        print(f"  {model_size} on {engine.device}/{engine.compute_type}\n", flush=True)

        header = (
            f"{'':>9}  {'model':<7}{'clip':<16}{'variant':<12}"
            f"{'latency':>9}{'wer':>7}  remaining"
        )
        print(header, flush=True)
        print("-" * len(header), flush=True)

        for row, samples, sr in clips:
            reference = normalise_for_wer(row["text"])
            for variant in variants:
                audio = (
                    samples if variant == "raw" else audio_quality.normalize(samples)
                )
                report = audio_quality.check_samples(audio, sr)
                result = engine.transcribe_full(audio, sr)
                hypothesis = normalise_for_wer(result.text)
                wer = word_error_rate(reference, hypothesis)

                results.append(
                    Row(
                        clip_id=row["clip_id"],
                        source=row.get("source", ""),
                        bucket=bucket_of(row),
                        noise_snr_db=row.get("noise_snr_db", ""),
                        intent=row.get("intent", ""),
                        model=model_size,
                        variant=variant,
                        reference=reference,
                        hypothesis=hypothesis,
                        wer=wer,
                        confidence=result.confidence,
                        no_speech_prob=result.no_speech_prob,
                        duration_s=result.duration_s,
                        latency_s=result.latency_s,
                        snr_estimate_db=report.snr_db,
                        gate_verdict=report.verdict,
                        gate_reason=report.reason or "",
                    )
                )

                # Progress, printed per transcription and flushed, because a
                # silent CPU run of this length is indistinguishable from a hang.
                # The estimate is a running mean, so it is wrong early and
                # settles after a few clips; it also cannot know that a model
                # still to be loaded will download first.
                done += 1
                elapsed = time.perf_counter() - started
                remaining = (elapsed / done) * (total - done)
                print(
                    f"[{done:>3}/{total:<3}]  {model_size:<7}{row['clip_id']:<16}"
                    f"{variant:<12}{result.latency_s:>8.2f}s{wer:>7.2f}"
                    f"  {_duration(remaining)} left",
                    flush=True,
                )

        print(flush=True)

    print(f"all {done} transcriptions done in {_duration(time.perf_counter() - started)}\n")
    return results


def _duration(seconds: float) -> str:
    """Render a number of seconds as m:ss, for the progress line."""
    if seconds < 0 or seconds != seconds:  # negative or NaN
        return "  ?  "
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}:{rest:02d}"


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def write_results(rows: list[Row], path: Path) -> None:
    """Write the per-clip results table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def print_wer_table(rows: list[Row], models: list[str], variants: list[str]) -> None:
    """Print mean WER by bucket, one column per model and variant."""
    buckets: dict[tuple[str, str, str], Bucket] = {}
    for row in rows:
        buckets.setdefault((row.bucket, row.model, row.variant), Bucket()).add(
            row.wer, row.latency_s
        )

    order = ["tts", "recorded", "noisy 20 dB", "noisy 10 dB", "noisy 5 dB"]
    present = [b for b in order if any(k[0] == b for k in buckets)]
    present += sorted({k[0] for k in buckets} - set(present))

    columns = [(m, v) for m in models for v in variants]
    print("\nmean WER by source and model (lower is better)")
    header = f"{'source':<16}" + "".join(f"{m + '/' + v:>18}" for m, v in columns)
    print(header)
    print("-" * len(header))
    for bucket in present:
        line = f"{bucket:<16}"
        for model, variant in columns:
            entry = buckets.get((bucket, model, variant))
            line += f"{entry.mean_wer:>18.3f}" if entry else f"{'—':>18}"
        print(line)
    print("-" * len(header))

    line = f"{'ALL':<16}"
    for model, variant in columns:
        matching = [r.wer for r in rows if r.model == model and r.variant == variant]
        line += f"{np.mean(matching):>18.3f}" if matching else f"{'—':>18}"
    print(line)


def print_latency_table(rows: list[Row], models: list[str]) -> None:
    """Print mean latency and the real-time factor per model."""
    print("\nlatency per model")
    header = f"{'model':<12}{'mean s':>10}{'median s':>11}{'max s':>9}{'xRT':>8}"
    print(header)
    print("-" * len(header))
    for model in models:
        matching = [r for r in rows if r.model == model]
        if not matching:
            continue
        latencies = [r.latency_s for r in matching]
        audio_seconds = sum(r.duration_s for r in matching)
        factor = sum(latencies) / audio_seconds if audio_seconds else float("nan")
        print(
            f"{model:<12}{np.mean(latencies):>10.2f}{np.median(latencies):>11.2f}"
            f"{np.max(latencies):>9.2f}{factor:>8.2f}"
        )
    print("-" * len(header))
    print("xRT = seconds of compute per second of audio; below 1.0 is realtime")


def print_gate_report(rows: list[Row]) -> None:
    """Print the SNR estimate per clip and what the quality gate decided.

    The gate runs on the audio, not the transcript, so its verdict is identical
    for every model — one row per clip and variant is enough.
    """
    seen: dict[tuple[str, str], Row] = {}
    for row in rows:
        seen.setdefault((row.clip_id, row.variant), row)

    print("\nquality gate: estimated SNR and verdict")
    header = f"{'clip':<16}{'bucket':<14}{'raw SNR':>9}{'norm SNR':>10}{'verdict':>10}{'reason':>12}"
    print(header)
    print("-" * len(header))

    clip_ids = sorted({cid for cid, _ in seen})
    rejected: list[str] = []
    for clip_id in clip_ids:
        raw = seen.get((clip_id, "raw"))
        norm = seen.get((clip_id, "normalized"))
        reference = raw or norm
        if reference is None:
            continue
        verdict = reference.gate_verdict
        if verdict == "repeat":
            rejected.append(clip_id)
        print(
            f"{clip_id:<16}{reference.bucket:<14}"
            f"{(raw.snr_estimate_db if raw else float('nan')):>9.1f}"
            f"{(norm.snr_estimate_db if norm else float('nan')):>10.1f}"
            f"{verdict:>10}{reference.gate_reason or '—':>12}"
        )
    print("-" * len(header))
    if rejected:
        print(f"{len(rejected)} clips rejected by check(): {', '.join(rejected)}")
    else:
        print("no clips rejected by check() — the threshold may be too permissive")

    settings = get_settings()
    print(
        f"thresholds in config.py: min_snr_db={settings.min_snr_db}, "
        f"min_speech_ratio={settings.min_speech_ratio}, "
        f"max_clipping_ratio={settings.max_clipping_ratio}"
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the manifest path, which models, which variants and the output path."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path,
                        default=settings.paths.audio / "manifest.csv",
                        help="the clips to evaluate")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Whisper model sizes to compare")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        choices=["raw", "normalized"],
                        help="run on the audio as-is, normalised, or both")
    parser.add_argument("--out", type=Path,
                        default=settings.paths.reports / "stt_results.csv",
                        help="where the per-clip results go")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Evaluate every clip, write the results table and print the summary."""
    args = parse_args(argv)

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        print("run: python scripts/pick_audio_sentences.py", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parents[1]
    manifest_rows = read_manifest(args.manifest)
    print(f"{len(manifest_rows)} rows in {args.manifest.name}")
    print(f"models: {', '.join(args.models)}   variants: {', '.join(args.variants)}")

    rows = evaluate(manifest_rows, root, args.models, args.variants)
    if not rows:
        print("nothing evaluated — no clip in the manifest has audio", file=sys.stderr)
        return 1

    write_results(rows, args.out)
    print_wer_table(rows, args.models, args.variants)
    print_latency_table(rows, args.models)
    print_gate_report(rows)
    print(f"\n{len(rows)} rows written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
