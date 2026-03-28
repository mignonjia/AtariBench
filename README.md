# AtariBench

Compact Atari runner for multimodal LLM experiments.

## Setup

Create or update the `ale` conda environment from [`environment.yaml`](environment.yaml), then export the provider keys you need:

```bash
conda env create -f environment.yaml
```

This installs the Python dependencies plus the `ffmpeg` binary used for video rendering.

```bash
source ~/.zshrc
conda activate ale
export GEMINI_API_KEY="YOUR_GEMINI_KEY"
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY"
```

## Main Command

The main workflow is config-driven batch mode.

- [`config/common.yaml`](config/common.yaml): shared defaults such as game selections, retries, concurrency caps, duration, and rendering settings
- [`config/runs.yaml`](config/runs.yaml): the full set of batch run definitions
- [`config/sample_runs.yaml`](config/sample_runs.yaml): a smaller debug batch

For a quick debug run, use `sample_runs.yaml`. If you want even faster debug cycles, lower `duration_seconds` in [`config/common.yaml`](config/common.yaml).

Run the debug batch with:

```bash
python batch_run.py \
  --common-config config/common.yaml \
  --runs-config config/sample_runs.yaml
```

Use [`config/runs.yaml`](config/runs.yaml) instead of `sample_runs.yaml` when you want the full batch.

Optional minimal logging:

```bash
python batch_run.py \
  --common-config config/common.yaml \
  --runs-config config/sample_runs.yaml \
  --minimal-logging
```

This keeps only `summary.json`, `turns.jsonl`, and `visualization.mp4` in each run directory after rendering. In config-driven mode, you can also set `minimal_logging: true` in `common.yaml` or per-run entries.

- `games` can be a single game key, `selected`, `full`, or a list mixing those values.
- `seed_start: 0` with `num_runs: 3` expands to seeds `0`, `1`, and `2`.
- Allowed `thinking_mode` values are model-specific; see [`llm/model_thinking.json`](llm/model_thinking.json).

## Other Commands

Single run:

```bash
python main.py \
  --game breakout \
  --model gemini-2.5-flash \
  --thinking off \
  --prompt-mode structured_history \
  --duration-seconds 30
```

## Prompt Modes

Supported values:

- `structured_history`
- `append_only`

`structured_history` uses curated recent clips plus non-zero-reward clips.

`append_only` uses a chronological user/assistant transcript. At serving time this is sent as structured chat messages, not hand-written role tags.

## Thinking Modes

Thinking support is model-specific.

- The allowed modes per model live in [`llm/model_thinking.json`](llm/model_thinking.json).
- The effective resolved request settings come from [`llm/common.py`](llm/common.py) via `describe_effective_thinking_mode()`.
- `thinking_mode` is the requested user-facing knob.
- `thinking_level` and `thinking_budget` are the resolved provider-specific settings recorded in summaries.

## Important Options

- `--duration-seconds`: total game budget
- `--prompt-mode`: `structured_history` or `append_only`
- `--minimal-logging`: after rendering, keep only `summary.json`, `turns.jsonl`, and `visualization.mp4`
- `max_concurrency_by_company`: config-driven per-company concurrency caps in `common.yaml`
- `max_retries`: config-driven transient retry count in `common.yaml`
- `retry_backoff_seconds`: config-driven transient retry backoff base in `common.yaml`
- `render_video_fps`: config-driven visualization FPS in `common.yaml`

## Output

Successful runs are stored under canonical per-game roots:

```text
runs/<game>/<model>/
```

Each final canonical run directory is created under that model root.

- Single runs use `MMDD_HHMMSS`
- Config-driven batch runs use `{batch_timestamp}_cfg_xxx_run_xxx`
- If a name already exists, a numeric suffix is appended such as `_2`

Each completed run writes:

- `frames/`
- `prompts/turn_XXXX.txt`
- `prompts/turn_XXXX.html`
- `responses/turn_XXXX.txt`
- `turns.jsonl`
- `summary.json`
- `visualization.mp4`

If `--minimal-logging` is enabled, or `minimal_logging: true` is set in batch config, the run is pruned after video rendering and only these remain:

- `turns.jsonl`
- `summary.json`
- `visualization.mp4`

Per-game summaries live at:

- `runs/<game>/model_summary.json`
- `runs/<game>/model_summary_30s.json`

Cross-game flat summaries live at:

- `runs/model_summary.json`
- `runs/model_summary_30s.json`

`model_summary.json` includes all successful runs for that setting, including shorter debug runs that still finished with `stop_reason=frame_budget`.
`model_summary_30s.json` keeps the previous benchmark view and only includes full 30-second canonical runs.
Both summaries are aggregated per setting, not by mixing different prompt/thinking/clip configurations into one average.
They are rebuilt from stored run `summary.json` files on disk, not limited to the runs from the most recent batch.

## Batch Logging

Shared cross-game batch metadata is stored under:

- `runs/_batches/<runs_config_stem>_<batch_timestamp>/`

Example:

- `config/sample_runs.yaml` -> `runs/_batches/sample_runs_0328_111856/`

Single-game batch metadata is stored under:

- `runs/<game>/_batches/<batch_timestamp>/`

Each batch writes its own `batch_summary.json`, and that file only describes the runs from that batch.

After a batch finishes, the runner refreshes both `model_summary.json` and `model_summary_30s.json` for each canonical game that had at least one successful run in that batch.
This refresh still happens if the batch had some failures; it is not limited to all-success batches.
The refreshed per-game summary files aggregate across stored eligible runs on disk for that game and then rebuild the matching top-level files in `runs/`.

Each started run prints one flat log line with the key config fields, including:

- `model_name`
- `thinking_mode`
- `prompt_mode`
- `history_clips`
- `non_zero_reward_clips`
- `games`
- `selected_game`
- `seed`
- `current_num_run`
- `total_num_runs`
- `output_dir`

## Notes

- The main runtime entrypoint is [`main.py`](main.py).
- Batch orchestration lives in [`batch_run.py`](batch_run.py).
- Video rendering lives in [`visualize.py`](visualize.py) and [`viz/render.py`](viz/render.py).
- Gameplay is always time-budgeted now; life loss consumes time but is not a separate runner termination rule.
