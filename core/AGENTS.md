# Core Agent Notes

Purpose:
- Own the main runner loop and persisted run artifacts.

Key files:
- `pipeline.py`: end-to-end runtime loop
- `clip.py`: parse raw model text into executable actions
- `trajectory.py`: persist frames, turns, and summary artifacts

Edit guidance:
- Keep `PipelineRunner.run()` behavior consistent with the tests in `tests/test_pipeline.py`.
- If you change parse fallback behavior, update both `clip.py` and the tests that inspect `turns.jsonl`.
- If you change saved artifact names or summary fields, inspect `viz/render.py`, `run_storage.py`, and related tests.

Important invariants:
- Frame budget is `duration_seconds * fps`.
- Planned actions execute for `frames_per_action` frames.
- Parse failures fall back to a single `noop` while preserving the raw response and parse errors.

Usual follow-up tests:
- `python -m unittest tests.test_pipeline tests.test_clip tests.test_trajectory`
