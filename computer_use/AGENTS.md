# Computer Use Agent Notes

Purpose:
- Own experimental manual-control and ALE setup scripts.

Scope:
- This directory is peripheral to the main AtariBench runner.
- Changes here should not affect `main.py`, `batch_run.py`, or the main `tests/` suite unless explicitly intended.

Key files:
- `common.py`: shared manual-control helpers and a sample interactive loop
- `breakout.py`, `assault.py`, `freeway.py`, `pacman.py`, `tennis.py`: game-specific experiments
- `setup.py`: ALE build/setup helper

Edit guidance:
- Treat this directory as experimental and loosely coupled.
- Prefer isolating changes here from the main runner path unless the task is specifically about manual control or ALE packaging.
