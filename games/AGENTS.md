# Games Agent Notes

Purpose:
- Own supported game definitions, environment helpers, and prompt assembly.

Key files:
- `registry.py`: `GameSpec` definitions and supported games
- `env.py`: ALE environment creation and info/frame helpers
- `prompt_builder.py`: assembles prompt text, ordered image paths, and optional structured messages
- `prompts/*.py`: per-game action maps and prompt text

Edit guidance:
- Add new game support by updating `registry.py` and creating a matching module under `prompts/`.
- Keep action names normalized and aligned with the model output parser expectations in `core/clip.py`.
- Keep prompt placeholder ordering aligned with the `image_paths` list emitted by `prompt_builder.py`.
- Keep named selection presets in `registry.py` aligned with the documented config aliases and `tests/test_games.py`.

Important invariants:
- Registry discovers all prompt-backed games, and also exposes named selections:
  `selected` = `breakout`, `assault`
  `full` = all currently registered games.
- Config-driven batch runs can reference those same selections through
  `config/common.yaml` game aliases plus `config/runs.yaml` entries.
- `prompt_builder.py` supports two prompt modes:
  `structured_history` appends reward clips, recent clips, and the current frame in that order;
  `append_only` builds a chronological user/assistant transcript, serializes it for saved artifacts, and is the only prompt mode that can carry explicit context-cache hints downstream.
- Gameplay prompts should treat life loss as a gameplay setback that consumes
  time, not as a separate runner termination rule.

Usual follow-up tests:
- `python -m unittest tests.test_games tests.test_pipeline`
