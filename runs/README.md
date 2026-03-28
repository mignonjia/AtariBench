# Runs Directory

This folder stores persisted AtariBench outputs.

## Structure

```text
runs/
  README.md
  _batches/
    <batch_label_or_timestamp>/
      batch_summary.json
      logs/
        <game>_<job_label>_run_<n>.log
  <game>/
    .model_summary.lock
    _batches/
      <batch_timestamp>/
        logs/
          <game>_<job_label>_run_<n>.log
    <model>/
      <run_id>/
        frames/
        prompts/
        responses/
        turns.jsonl
        summary.json
        visualization.mp4
    model_summary.json
    model_summary_30s.json
  model_summary.json
  model_summary_30s.json
```

## Notes

- `<game>/` is the canonical root for one Atari game such as `breakout` or `assault`.
- `<model>/` groups all stored runs for one model under that game.
- `<run_id>/` is one completed run directory.
- Single runs use a timestamp like `MMDD_HHMMSS`.
- Config-driven batch runs use names like `<batch_timestamp>_cfg_001_run_001`.
- `_batches/` stores batch-level metadata and per-run text logs.
- `<game>/_batches/` is used when a batch targets one canonical game.
- `runs/_batches/` is used for shared multi-game batch metadata.
- `summary.json` stores the final metrics and resolved config for one run.
- `turns.jsonl` stores the per-turn textual interaction log.
- `visualization.mp4` is the rendered video when `ffmpeg` is available.
- `model_summary.json` aggregates eligible stored runs for one game.
- `model_summary_30s.json` is a legacy per-game aggregate for 30-second runs.
- `runs/model_summary.json` aggregates across games.
- `runs/model_summary_30s.json` is a legacy top-level aggregate for 30-second runs. **These are the main results.**
- `.model_summary.lock` is an internal lock file used while refreshing summaries.

## Minimal Logging

With `--minimal-logging` or `minimal_logging: true`, the per-run directory is pruned after rendering and keeps only, which is sufficient for generate model_summary.json and model_summary_30s.json:

- `summary.json`
- `turns.jsonl`
- `visualization.mp4`