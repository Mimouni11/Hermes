"""Build the exam gazetteer used for fuzzy exam matching.

Reads the exam catalogue from `data/raw`, expands every official name, alias,
abbreviation and body part into normalised surface forms, and writes the
surface-form to exam-id map to `data/processed`.

    python scripts/build_exam_gazetteer.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse the catalogue path, output path and overwrite flag."""
    ...


def main() -> int:
    """Build and save the gazetteer, reporting surface forms per exam."""
    ...


if __name__ == "__main__":
    raise SystemExit(main())
