# Tests Agent Notes

Purpose:
- Own regression coverage for the main runner, batch orchestration, model-layer helpers, and storage logic.

Key files:
- `test_pipeline.py`: runtime loop behavior
- `test_batch_run.py`: batch orchestration and retry behavior
- `test_games.py`: registry and prompt/game layer behavior
- `test_llm.py`: provider and thinking-mode behavior
- `test_model_thinking_options.py`: declared thinking-mode matrix and optional live requests
- `test_run_storage.py`: canonical storage layout
- `test_clip.py`, `test_trajectory.py`: parser and persistence behavior
- `test_cleanup_invalid_runs.py`: cleanup-script behavior

Edit guidance:
- Extend tests in the same area as the code you change.
- Prefer targeted unit tests over large integration fixtures.
- Most tests assume execution from the repo root and import via `PROJECT_DIR`.
- Keep fixture expectations aligned with the maintained source tree, not with archived run data.

Useful commands:
- `python -m unittest`
- `python -m unittest tests.test_pipeline`
- `python -m unittest tests.test_batch_run`
- `python -m unittest tests.test_model_thinking_options`
