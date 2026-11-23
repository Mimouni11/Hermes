"""Build the embedded FAQ index used for retrieval.

Reads the FAQ question/answer CSV from `data/raw`, embeds every question and
writes the entries plus their embedding matrix to `data/processed`. Re-run
whenever the FAQ changes or the embedding model is swapped.

    python scripts/build_faq_index.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse the input CSV path, output path and overwrite flag."""
    ...


def main() -> int:
    """Build and save the FAQ index, reporting how many entries were embedded."""
    ...


if __name__ == "__main__":
    raise SystemExit(main())
