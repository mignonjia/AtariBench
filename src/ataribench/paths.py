"""Path helpers for command-line entrypoints."""

from __future__ import annotations

from pathlib import Path


def default_project_dir() -> Path:
    """Return the project directory used for relative config and run outputs."""

    return Path.cwd()
