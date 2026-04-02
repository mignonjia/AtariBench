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

- [`config/common.yaml`](config/common.yaml): shared defaults such as game selections, incomplete-run retries, concurrency caps, duration, and rendering settings
- [`config/runs.yaml`](config/runs.yaml): the full set of batch run definitions
- [`config/debug.yaml`](config/debug.yaml): a smaller debug batch

Some params:

- `games` can be a single game key, `selected`, `full`, or a list mixing those values.
- Allowed `thinking_mode` values are model-specific; see [`llm/model_thinking.json`](llm/model_thinking.json).

For a quick debug run, use `debug.yaml`. If you want even faster debug cycles, lower `duration_seconds` in [`config/common.yaml`](config/common.yaml).

Run the debug batch with:

```bash
python batch_run.py --common-config config/common.yaml --runs-config config/debug.yaml
```

Use [`config/runs.yaml`](config/runs.yaml) instead of `debug.yaml` when you want the full batch.

Optional minimal logging:

```bash
python batch_run.py --common-config config/common.yaml --runs-config config/debug.yaml  --minimal-logging
```

This keeps only `summary.json`, `turns.jsonl`, and `visualization.mp4` in each run directory after rendering. See [`runs/breakout/gemini-2.5-flash/0401_020222_cfg_001_run_001`](runs/breakout/gemini-2.5-flash/0401_020222_cfg_001_run_001) for an example run of complete logging, and [`runs/breakout/gemini-2.5-flash/0331_235632_cfg_001_run_001`](runs/breakout/gemini-2.5-flash/0331_235632_cfg_001_run_001) for an example run of minimal logging


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
It is also the only prompt mode where AtariBench can add explicit context-cache hints.

## Thinking Modes

Thinking support is model-specific.

- The allowed modes per model live in [`llm/model_thinking.json`](llm/model_thinking.json).
- The effective resolved request settings come from [`llm/common.py`](llm/common.py) via `describe_effective_thinking_mode()`.
- `thinking_mode` is the requested user-facing knob.
- `thinking_level` and `thinking_budget` are the resolved provider-specific settings recorded in summaries.
- `input_tokens`, `output_tokens`, `total_tokens`, `thinking_tokens`, and `cached_input_tokens` are recorded for cost estimation when the provider reports them.

## Important Options

- `--duration-seconds`: total game budget
- `--prompt-mode`: `structured_history` or `append_only`
- `--context-cache`: enable explicit cache hints for `append_only`; `structured_history` remains unchanged
- `--minimal-logging`: after rendering, keep only `summary.json`, `turns.jsonl`, and `visualization.mp4`
- `max_concurrency_by_company`: config-driven per-company concurrency caps in `common.yaml`
- `max_retries`: config-driven incomplete-run retry count in `common.yaml`
- `retry_backoff_seconds`: config-driven incomplete-run retry backoff base in `common.yaml`
- `render_video_fps`: config-driven visualization FPS in `common.yaml`

Transient provider errors such as `429 RESOURCE_EXHAUSTED`, network timeouts, and empty provider responses are retried inside the active turn by the LLM adapter. Batch-level reruns are reserved for incomplete runs that exited cleanly without reaching the full frame budget.

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

`turns.jsonl` records per-turn token usage when the backing SDK reports it.
`summary.json` records run-level `input_tokens`, `output_tokens`, `total_tokens`, `thinking_tokens`, `cached_input_tokens`, plus token-usage coverage counts across turns.
It also records `context_cache` so cached and uncached append-only runs stay distinct in aggregated summaries.

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
They also include average reward/lost-life standard error plus average and latest token totals so you can estimate variance and per-setting API cost from stored runs.
They are rebuilt from stored run `summary.json` files on disk, not limited to the runs from the most recent batch.

See [`runs/breakout/gemini-2.5-flash/0401_020222_cfg_001_run_001`](runs/breakout/gemini-2.5-flash/0401_020222_cfg_001_run_001) for an example run of complete logging, and [`runs/breakout/gemini-2.5-flash/0331_235632_cfg_001_run_001`](runs/breakout/gemini-2.5-flash/0331_235632_cfg_001_run_001) for an example run of minimal logging

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
