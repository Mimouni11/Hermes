"""Tests for settings loading, path layout and threshold defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet")


def test_project_root_is_found_from_the_package() -> None:
    """The root is located without depending on the working directory."""
    ...


def test_env_file_is_read(tmp_path: Path) -> None:
    """Values in a `.env` file reach the settings object."""
    ...


def test_real_environment_wins_over_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """An already-set variable is not overwritten by `.env`."""
    ...


def test_missing_env_file_uses_defaults(tmp_path: Path) -> None:
    """Settings load with no `.env` present, using the documented defaults."""
    ...


def test_writable_directories_are_created(tmp_path: Path) -> None:
    """Building the paths creates `data/processed` and `models`."""
    ...


def test_settings_are_cached() -> None:
    """Repeated calls return the same settings object."""
    ...


def test_requiring_a_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asking for an absent Gemini key raises a clear error."""
    ...
