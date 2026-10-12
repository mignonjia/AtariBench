# Output and Batch Logging

AtariBench writes generated run data under `runs/`. The directory is ignored by git because it contains local experiment outputs, videos, frames, prompts, and provider responses.

## Run Directories

Successful runs are stored under canonical per-game roots:

```text
runs/<game>/<model>/
```

Each final canonical run directory is created under that model root.

- Single runs use `MMDD_HHMMSS`.
- Config-driven batch runs use `{batch_timestamp}_cfg_xxx_run_xxx`.
- If a name already exists, a numeric suffix is appended such as `_2`.

Each completed run writes:

- `frames/`
- `prompts/turn_XXXX.txt`
- `prompts/turn_XXXX.html`
- `responses/turn_XXXX.txt`
- `turns.jsonl`
- `summary.json`
- `visualization.mp4`

`turns.jsonl` records per-turn token usage when the backing SDK reports it.
`summary.json` records run-level `input_tokens`, `output_tokens`, `total_tokens`, `thinking_tokens`, `cached_input_tokens`, plus token-usage coverage counts across turns.
It also records `context_cache`, `text_feedback_mode`, and `max_actions_per_turn`, so ablations stay distinct in aggregated summaries. Action diagnostics include parse/fallback turn counts and rates, average actions per turn, and the largest executed action chunk.

If `--minimal-logging` is enabled, or `minimal_logging: true` is set in batch config, the run is pruned after video rendering and only these files remain:

- `turns.jsonl`
- `summary.json`
- `visualization.mp4`

## Model Summaries

Per-game summaries live at:

- `runs/<game>/model_summary.json`
- `runs/<game>/model_summary_30s.json`

Cross-game flat summaries live at:

- `runs/model_summary.json`
- `runs/model_summary_30s.json`

`model_summary.json` includes all successful runs for that setting, including shorter debug runs that still finished with `stop_reason=frame_budget`.
Per-game `model_summary_30s.json` keeps the previous benchmark view and only includes full 30-second canonical runs.
Top-level `runs/model_summary_30s.json` is the baseline no-thinking benchmark table, limited to random, DeepSeek, GLM, Gemini 2.5 Flash, and GPT-5.4 Mini entries whose thinking mode is `off` or `none` and whose thinking level is unset/none.

Both summaries are aggregated per setting, not by mixing different prompt/thinking/clip, textual-feedback, game-rule, output-format, or action-chunk configurations into one average.
They include average reward/lost-life standard error plus average and latest token totals so you can estimate variance and per-setting API cost from stored runs.
They are rebuilt from stored run `summary.json` files on disk, not limited to the runs from the most recent batch.

## Batch Metadata

Shared cross-game batch metadata is stored under:

```text
runs/_batches/<runs_config_stem>_<batch_timestamp>/
```

Example:

```text
config/debug.yaml -> runs/_batches/debug_0328_111856/
```

Single-game batch metadata is stored under:

```text
runs/<game>/_batches/<batch_timestamp>/
```

Each batch writes its own `batch_summary.json`, and that file only describes the runs from that batch.

After a batch finishes, the runner refreshes both `model_summary.json` and `model_summary_30s.json` for each canonical game that had at least one successful run in that batch.
This refresh still happens if the batch had some failures; it is not limited to all-success batches.
The refreshed per-game summary files aggregate across stored eligible runs on disk for that game and then rebuild the matching top-level files in `runs/`.

Each started run prints one flat log line with the key config fields:

- `model_name`
- `thinking_mode`
- `prompt_mode`
- `text_feedback_mode`
- `max_actions_per_turn`
- `history_clips`
- `non_zero_reward_clips`
- `games`
- `selected_game`
- `seed`
- `current_num_run`
- `total_num_runs`
- `output_dir`
