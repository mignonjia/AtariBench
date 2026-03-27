# AtariBench

This runner executes Atari games with multimodal LLMs in a repeatable workflow.

## Setup

Use the `ale` conda environment and set the API key for the provider you want:

```bash
source ~/.zshrc
conda activate ale
export GEMINI_API_KEY="YOUR_GEMINI_KEY"
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
```

## Single Run

Run Breakout once:

```bash
python AtariBench/main.py \
  --game breakout \
  --model gemini-2.5-flash \
  --thinking off \
  --prompt-mode structured_history \
  --duration-seconds 30
```

Run Breakout once on OpenAI:

```bash
python AtariBench/main.py \
  --game breakout \
  --model gpt-5.4 \
  --provider openai \
  --thinking low \
  --prompt-mode structured_history \
  --duration-seconds 30
```

## Batch Run

Use `AtariBench/batch_run.py`.

Basic shape:

```bash
python AtariBench/batch_run.py \
  --game breakout \
  --job MODEL:COUNT:THINKING \
  --prompt-mode structured_history \
  --max-concurrency 1
```

Example: run Gemini 3 Flash Preview 5 times with minimal thinking:

```bash
python AtariBench/batch_run.py \
  --game breakout \
  --job gemini-3-flash-preview:5:minimal \
  --max-concurrency 1 \
  --max-retries 2 \
  --retry-backoff-seconds 10
```

Example: mix multiple models in one batch:

```bash
python AtariBench/batch_run.py \
  --game breakout \
  --job gemini-2.5-flash:5:off \
  --job gemini-3-flash-preview:5:minimal \
  --max-concurrency 1
```

## Job Format

Each `--job` is:

```text
MODEL:COUNT[:THINKING]
```

Examples:

```text
gemini-2.5-flash:5:off
gemini-3-flash-preview:5:minimal
gemini-3.1-pro-preview:1:low
```

## Thinking Modes

Supported values:

- `default`: no explicit thinking config
- `off`: sends `thinking_budget=0`
- `minimal`: sends `thinking_level=MINIMAL`
- `low`: sends `thinking_level=LOW`
- `on`: sends `thinking_level=MEDIUM`

Logged metadata:

- `thinking_mode` records the requested mode
- `thinking_level` records the named level used for the request
- `thinking_budget` is only set for `off`

## Important Flags

- `--duration-seconds 30`: total game budget
- `--history-clips 3`: number of recent clips included in structured-history mode
- `--non-zero-reward-clips 3`: number of reward/life-loss clips included in structured-history mode
- `--prompt-mode structured_history|append_only`: choose curated history vs chronological append-only transcript prompting
- `--max-concurrency`: how many runs to execute at once
- `--max-retries`: retries for transient `429` and `503` failures
- `--retry-backoff-seconds`: base backoff between retries
- `--render-video-fps 30`: output FPS for the visualization video

## Prompt Modes

- `structured_history`: the original prompt style, with separate recent-history and non-zero-reward-history sections
- `append_only`: a chronological transcript where prior assistant actions are followed by user-provided observed states, rewards, and updated instructions

## Output Layout

A batch creates:

```text
AtariBench/runs/batches/<game>_<timestamp>/
  batch_summary.json
  logs/
  runs/
    <model>/
      run_001/
        <game>/
          <timestamp>/
            frames/
            prompts/
              turn_0001.txt
              turn_0001.html
            responses/
            turns.jsonl
            summary.json
            visualization.mp4
```

## What Gets Logged

Each completed run stores:

- raw prompt text in `prompts/turn_XXXX.txt`
- HTML prompt render in `prompts/turn_XXXX.html`
- raw model response in `responses/turn_XXXX.txt`
- frame trajectory in `frames/`
- structured turn data in `turns.jsonl`
- run summary in `summary.json`
- whiteboard video in `visualization.mp4`

The HTML prompt render expands each `IMG_HOLDER` to the actual referenced image file so you can inspect which screenshot was sent.

Important summary metadata now includes:

- `thinking_mode`, `thinking_level`, `thinking_budget`
- `history_clips`, `non_zero_reward_clips`
- `prompt_mode`

## Reading Results

Batch status summary:

```bash
cat AtariBench/runs/batches/<batch>/batch_summary.json
```

Open one run summary:

```bash
cat AtariBench/runs/batches/<batch>/runs/<model>/run_001/<game>/<timestamp>/summary.json
```

Open one rendered video:

```bash
open AtariBench/runs/batches/<batch>/runs/<model>/run_001/<game>/<timestamp>/visualization.mp4
```

## Notes

- The primary entrypoint is `AtariBench/main.py`.
- Breakout currently runs at `30 FPS`.
- Each planned action executes for `3` frames, which is `0.1` seconds.
- Some models reject specific thinking or reasoning levels. In that case, use a supported mode for that model.
