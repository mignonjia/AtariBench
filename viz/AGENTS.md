# Viz Agent Notes

Purpose:
- Own rendering stored run artifacts into videos.

Key files:
- `render.py`: loads `turns.jsonl`, composes whiteboard frames, and encodes mp4 output

Edit guidance:
- Treat `turns.jsonl` and stored frame filenames as external contracts from `core/trajectory.py`.
- If you change panel layout or metadata fields, verify the renderer still handles missing or partial turn data gracefully.
- Avoid changing output filenames unless the caller path in `main.py`, `batch_run.py`, and `visualize.py` changes with it.
- Remember that rendering creates a temporary `visualization_frames/` directory, which `apply_minimal_logging_policy()` may later prune.

Important invariants:
- Rendering expects `frames/frame_*.png`.
- Rendering expects turn metadata in `turns.jsonl`.
- Default output is `visualization.mp4` inside the run directory.
- Rendering shells out to `ffmpeg`, so environment/setup docs must stay aligned with that dependency.
