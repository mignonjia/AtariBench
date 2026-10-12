"""Test bootstrap for the Atari runner."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
candidate = str(SRC_DIR)
if candidate not in sys.path:
    sys.path.insert(0, candidate)
