# Core Agent Notes

Purpose:
- Own the main runner loop and persisted run artifacts.

Key files:
- `pipeline.py`: end-to-end runtime loop
- `clip.py`: parse raw model text into executable actions
- `trajectory.py`: persist frames, turns, and summary artifacts

Edit guidance:
- Keep `PipelineRunner.run()` behavior consistent with the tests in `tests/test_pipeline.py`.
- If you change parse fallback behavior, update both `clip.py` and the tests that inspect stored turn records.
- If you change prompt/turn artifact shape, inspect `games/prompt_builder.py`, `viz/render.py`, and related tests.
- If you change saved artifact names or summary fields, inspect `main.py`, `run_storage.py`, `cleanup_invalid_runs.py`, and related tests.

Important invariants:
- Frame budget is `duration_seconds * fps`.
- Planned actions execute for `frames_per_action` frames.
- Parse failures fall back to a single `noop` while preserving the raw response and parse errors.
- `append_only` prompt mode forces effective clip counts to `-1`.
- `context_cache` is only effective for `append_only`; other prompt modes disable it before the provider call and in persisted summaries.
- Episode termination resets the env and marks the stored turn with `new_game_started=True` instead of ending the run early.
- `minimal_logging` is recorded in `summary.json`, but pruning happens after rendering in the CLI layer, not inside `Trajectory.finalize()`.

Usual follow-up tests:
- `python -m unittest tests.test_pipeline tests.test_clip tests.test_trajectory tests.test_main`
