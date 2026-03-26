# Games Agent Notes

Purpose:
- Own supported game definitions, environment helpers, and prompt assembly.

Key files:
- `registry.py`: `GameSpec` definitions and supported games
- `env.py`: ALE environment creation and info/frame helpers
- `prompt_builder.py`: assembles prompt text and ordered image paths
- `prompts/*.py`: per-game action maps and prompt text

Edit guidance:
- Add new game support by updating `registry.py` and creating a matching module under `prompts/`.
- Keep action names normalized and aligned with the model output parser expectations in `core/clip.py`.
- Prompt changes should preserve the ordering contract between `IMG_HOLDER` references and `image_paths`.

Important invariants:
- Registry currently exposes `assault` and `breakout` as supported main-runner games.
- `prompt_builder.py` appends reward clips, recent clips, and the current frame in that order.
- Lost-life behavior is controlled per game via `MAX_LOST_LIVES`.

Usual follow-up tests:
- `python -m unittest tests.test_games tests.test_pipeline`
