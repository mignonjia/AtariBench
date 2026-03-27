# AtariBench

Compact Atari runner for multimodal LLM experiments.

## Setup

Use the `ale` conda environment and export the provider keys you need:

```bash
source ~/.zshrc
conda activate ale
export GEMINI_API_KEY="YOUR_GEMINI_KEY"
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY"
```

## Single Run

```bash
python main.py \
  --game breakout \
  --model gemini-2.5-flash \
  --thinking off \
  --prompt-mode structured_history \
  --duration-seconds 30
```

Append-only prompt mode:

```bash
python main.py \
  --game breakout \
  --model gemini-2.5-flash \
  --thinking off \
  --prompt-mode append_only \
  --duration-seconds 5
```

## Batch Run

Job-driven batch mode:

```bash
python batch_run.py \
  --game selected \
  --job gemini-2.5-flash:3:off \
  --max-concurrency 1
```

`--game` accepts:

- one game key such as `breakout`
- `selected` = `breakout`, `assault`
- `full` = all registered prompt-backed games

Each `--job` is:

```text
MODEL:COUNT[:THINKING]
```

Examples:

```text
gemini-2.5-flash:3:off
gpt-5.4-mini:1:none
claude-sonnet-4-6:2:medium
```

## Config Batch Run

Config-driven batch mode uses:

- [`config/common.yaml`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/config/common.yaml): shared defaults
- [`config/model_game_specific.yaml`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/config/model_game_specific.yaml): per-setting entries

Run it with:

```bash
python batch_run.py \
  --common-config config/common.yaml \
  --model-game-specific-config config/model_game_specific.yaml
```

Per-setting entries can specify:

- `model_name`
- `thinking_mode`
- `prompt_mode`
- `history_clips`
- `non_zero_reward_clips`
- `games`
- `seed_start`
- `num_runs`

`games` can be:

- one game key
- `selected`
- `full`
- a list mixing those values

`seed_start: 0` with `num_runs: 3` expands to seeds `0`, `1`, and `2`.

## Prompt Modes

Supported values:

- `structured_history`
- `append_only`

`structured_history` uses curated recent clips plus non-zero-reward clips.

`append_only` uses a chronological user/assistant transcript. At serving time this is sent as structured chat messages, not hand-written role tags.

## Thinking Modes

Thinking support is model-specific.

- The allowed modes per model live in [`llm/model_thinking.json`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/model_thinking.json).
- The effective resolved request settings come from [`llm/common.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/common.py) via `describe_effective_thinking_mode()`.
- `thinking_mode` is the requested user-facing knob.
- `thinking_level` and `thinking_budget` are the resolved provider-specific settings recorded in summaries.

## Important Flags

- `--duration-seconds`: total game budget
- `--history-clips`: recent clips for `structured_history`
- `--non-zero-reward-clips`: reward-bearing clips for `structured_history`
- `--prompt-mode`: `structured_history` or `append_only`
- `--max-concurrency`: batch parallelism
- `--max-retries`: transient retry count
- `--retry-backoff-seconds`: transient retry backoff base
- `--render-video-fps`: visualization FPS

## Output

Successful runs are stored under canonical per-game roots:

```text
runs/<game>/<model-or-config-label>/<timestamp>/
```

Each completed run writes:

- `frames/`
- `prompts/turn_XXXX.txt`
- `prompts/turn_XXXX.html`
- `responses/turn_XXXX.txt`
- `turns.jsonl`
- `summary.json`
- `visualization.mp4`

Per-game summaries live at:

- `runs/<game>/model_summary.json`

Cross-game flat summaries live at:

- `runs/model_summary.json`

These summaries are aggregated per setting, not by mixing different prompt/thinking/clip configurations into one average.

## Notes

- The main runtime entrypoint is [`main.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/main.py).
- Batch orchestration lives in [`batch_run.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/batch_run.py).
- Video rendering lives in [`visualize.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/visualize.py) and [`viz/render.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/viz/render.py).
- Gameplay is always time-budgeted now; life loss consumes time but is not a separate runner termination rule.
