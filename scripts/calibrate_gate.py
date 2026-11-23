"""Calibrate the quality gate's thresholds against measured WER.

Reads `reports/stt_results.csv` and answers one question: below what signal
quality is a transcript too wrong to act on? It bins the clips by estimated
SNR, by confidence and by no_speech_prob, prints mean WER per bin, and finds
where WER crosses the usable limit.

Two thresholds come out, and they sit at different points in the pipeline:

- ``min_snr_db`` gates **before** transcription. It is the cheap one: the audio
  is measured and a bad turn costs nothing but a request to repeat.
- ``min_confidence`` gates **after** transcription, since confidence only
  exists once Whisper has decoded. It catches what the SNR estimate misses —
  a clean recording of a mumble scores well on SNR and badly on confidence.

Neither is a hard rule in this module: it prints what the data supports, and
the numbers are written into `config.py` by hand so the choice stays visible.

    python scripts/calibrate_gate.py
    python scripts/calibrate_gate.py --limit 0.5 --variant raw
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hermes.config import get_settings  # noqa: E402

#: A transcript with a word error rate above this is not worth acting on: more
#: than half its words are wrong, so any slot filled from it is a guess.
DEFAULT_WER_LIMIT = 0.5

SNR_BINS = [-np.inf, 5, 8, 10, 12, 14, 16, 20, 25, 30, np.inf]
CONFIDENCE_BINS = [0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01]
NO_SPEECH_BINS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01]


@dataclass
class Clip:
    """One evaluated clip, reduced to what calibration needs."""

    clip_id: str
    bucket: str
    wer: float
    snr: float
    confidence: float
    no_speech: float


def read_results(path: Path, variant: str, model: str | None) -> list[Clip]:
    """Read the evaluation table, keeping one variant and optionally one model."""
    clips: list[Clip] = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != variant:
                continue
            if model is not None and row["model"] != model:
                continue
            clips.append(
                Clip(
                    clip_id=row["clip_id"],
                    bucket=row["bucket"],
                    wer=float(row["wer"]),
                    snr=float(row["snr_estimate_db"]),
                    confidence=float(row["confidence"]),
                    no_speech=float(row["no_speech_prob"]),
                )
            )
    return clips


def _label(low: float, high: float, unit: str = "") -> str:
    if low == -np.inf:
        return f"< {high:g}{unit}"
    if high == np.inf:
        return f">= {low:g}{unit}"
    return f"{low:g}–{high:g}{unit}"


def print_bins(
    clips: list[Clip],
    value: str,
    bins: list[float],
    limit: float,
    unit: str = "",
) -> None:
    """Print mean WER per bin of one measurement."""
    print(f"\nWER by {value}")
    header = f"{'bin':<14}{'clips':>7}{'mean WER':>10}{'max WER':>9}{'usable?':>9}"
    print(header)
    print("-" * len(header))
    for low, high in zip(bins, bins[1:]):
        group = [
            c for c in clips if low <= getattr(c, value) < high
        ]
        if not group:
            continue
        mean = float(np.mean([c.wer for c in group]))
        worst = max(c.wer for c in group)
        verdict = "ok" if mean <= limit else "TOO HIGH"
        print(
            f"{_label(low, high, unit):<14}{len(group):>7}{mean:>10.3f}"
            f"{worst:>9.3f}{verdict:>9}"
        )
    print("-" * len(header))


def find_threshold(
    clips: list[Clip],
    value: str,
    limit: float,
    higher_is_better: bool = True,
) -> tuple[float, float, float] | None:
    """Find the cut that separates usable transcripts from unusable ones.

    Walks every observed value as a candidate cut and keeps the one that is
    least aggressive while still holding the rejected group's mean WER at or
    above ``limit`` — so the gate only throws away audio that really is beyond
    use, rather than everything it could.

    Returns (threshold, mean WER rejected, mean WER kept), or None when no cut
    separates the two: either nothing is bad enough to reject, or everything is.
    """
    observed = sorted({getattr(c, value) for c in clips})
    best: tuple[float, float, float] | None = None

    for cut in observed:
        if higher_is_better:
            rejected = [c for c in clips if getattr(c, value) < cut]
            kept = [c for c in clips if getattr(c, value) >= cut]
        else:
            rejected = [c for c in clips if getattr(c, value) > cut]
            kept = [c for c in clips if getattr(c, value) <= cut]
        if not rejected or not kept:
            continue

        rejected_wer = float(np.mean([c.wer for c in rejected]))
        kept_wer = float(np.mean([c.wer for c in kept]))
        if rejected_wer >= limit:
            # Keep the cut that rejects the most while staying justified.
            if best is None or (
                cut > best[0] if higher_is_better else cut < best[0]
            ):
                best = (cut, rejected_wer, kept_wer)
    return best


def report_threshold(
    name: str,
    found: tuple[float, float, float] | None,
    clips: list[Clip],
    value: str,
    limit: float,
    higher_is_better: bool = True,
) -> float | None:
    """Print what a threshold would do to the set."""
    print(f"\n{name}")
    if found is None:
        print(
            f"  no cut on {value} isolates a group with mean WER >= {limit:g} — "
            "either nothing is bad enough to reject, or the measure does not "
            "separate good from bad"
        )
        return None

    cut, rejected_wer, kept_wer = found
    if higher_is_better:
        rejected = [c for c in clips if getattr(c, value) < cut]
    else:
        rejected = [c for c in clips if getattr(c, value) > cut]

    print(f"  threshold: {cut:.3g}")
    print(f"  rejects {len(rejected)} of {len(clips)} clips, mean WER {rejected_wer:.3f}")
    print(f"  keeps   {len(clips) - len(rejected)}, mean WER {kept_wer:.3f}")
    print(f"  rejected: {', '.join(sorted(c.clip_id for c in rejected))}")
    return cut


def print_bucket_summary(clips: list[Clip], limit: float) -> None:
    """Print mean WER and SNR per source bucket, to see the cliff."""
    print("\nWER and estimated SNR by source")
    header = f"{'bucket':<16}{'clips':>7}{'mean WER':>10}{'mean SNR':>10}{'min SNR':>9}"
    print(header)
    print("-" * len(header))
    for bucket in sorted({c.bucket for c in clips}, key=lambda b: -_bucket_rank(b)):
        group = [c for c in clips if c.bucket == bucket]
        flag = "  <-- over the limit" if np.mean([c.wer for c in group]) > limit else ""
        print(
            f"{bucket:<16}{len(group):>7}"
            f"{np.mean([c.wer for c in group]):>10.3f}"
            f"{np.mean([c.snr for c in group]):>10.1f}"
            f"{min(c.snr for c in group):>9.1f}{flag}"
        )
    print("-" * len(header))


def _bucket_rank(bucket: str) -> float:
    """Order buckets from cleanest to noisiest for display."""
    if bucket == "tts":
        return 100.0
    if bucket == "recorded":
        return 50.0
    try:
        return float(bucket.split()[1])
    except (IndexError, ValueError):
        return 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the results path, the WER limit and which rows to use."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path,
                        default=settings.paths.reports / "stt_results.csv",
                        help="the table written by eval_stt.py")
    parser.add_argument("--limit", type=float, default=DEFAULT_WER_LIMIT,
                        help="the WER above which a transcript is unusable")
    parser.add_argument("--variant", default="raw", choices=["raw", "normalized"],
                        help="which audio variant to calibrate on")
    parser.add_argument("--model", default=None,
                        help="only use rows from this model (default: all)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print the calibration tables and the thresholds they support."""
    args = parse_args(argv)

    if not args.results.exists():
        print(f"results not found: {args.results}", file=sys.stderr)
        print("run: python scripts/eval_stt.py", file=sys.stderr)
        return 1

    clips = read_results(args.results, args.variant, args.model)
    if not clips:
        print("no matching rows in the results table", file=sys.stderr)
        return 1

    print(f"{len(clips)} clips  variant={args.variant}  "
          f"model={args.model or 'all'}  WER limit={args.limit:g}")

    print_bucket_summary(clips, args.limit)
    print_bins(clips, "snr", SNR_BINS, args.limit, unit=" dB")
    print_bins(clips, "confidence", CONFIDENCE_BINS, args.limit)
    print_bins(clips, "no_speech", NO_SPEECH_BINS, args.limit)

    snr_cut = report_threshold(
        "min_snr_db  (gates before transcription)",
        find_threshold(clips, "snr", args.limit),
        clips,
        "snr",
        args.limit,
    )
    confidence_cut = report_threshold(
        "min_confidence  (gates after transcription)",
        find_threshold(clips, "confidence", args.limit),
        clips,
        "confidence",
        args.limit,
    )
    report_threshold(
        "max_no_speech_prob  (gates after transcription)",
        find_threshold(clips, "no_speech", args.limit, higher_is_better=False),
        clips,
        "no_speech",
        args.limit,
        higher_is_better=False,
    )

    settings = get_settings()
    print("\n" + "=" * 60)
    print("put these in config.py / .env")
    print("=" * 60)
    print(f"  MIN_SNR_DB={snr_cut:.1f}" if snr_cut else "  MIN_SNR_DB unchanged")
    print(
        f"  MIN_CONFIDENCE={confidence_cut:.2f}" if confidence_cut
        else "  MIN_CONFIDENCE unchanged"
    )
    print(f"\ncurrently: min_snr_db={settings.min_snr_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
