"""Build the intent centroids used by the intent gate.

Reads the labelled intent examples from `data/raw`, embeds them and writes one
averaged vector per intent to `data/processed`. Re-run whenever the examples
change.

    python scripts/intent/build_intent_centroids.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse the input CSV path, output path and overwrite flag."""
    ...


def main() -> int:
    """Build and save the centroids, reporting counts per intent."""
    ...


if __name__ == "__main__":
    raise SystemExit(main())
