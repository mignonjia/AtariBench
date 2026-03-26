"""Thin wrapper for the visualization CLI."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_local_paths() -> None:
    project_dir = Path(__file__).resolve().parent
    candidate = str(project_dir)
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


if __package__ in {None, ""}:
    _bootstrap_local_paths()
    from viz.render import main
else:
    from .viz.render import main


if __name__ == "__main__":
    raise SystemExit(main())
