# Repository Index

This index covers the maintained source tree and intentionally excludes stored run data and legacy prompt variants.

## Top Level

[`main.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/main.py)
- Single-run CLI.
- Validates the model/thinking pair, resolves the output layout, runs one pipeline, and writes back video metadata into `summary.json`.

[`batch_run.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/batch_run.py)
- Batch orchestration CLI.
- Supports ad hoc `--job` specs and config-driven batches from `config/common.yaml` plus `config/runs.yaml`.
- Expands game selections, enforces company-level concurrency caps, retries transient failures, and renders videos for successful runs.

[`run_storage.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/run_storage.py)
- Canonical storage helpers for maintained games.
- Resolves per-game/per-model output roots and rebuilds per-game plus cross-game summary files for both all-success runs and 30-second-only runs.
- Groups summary averages by full setting key, including prompt and thinking metadata.

[`cleanup_invalid_runs.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/cleanup_invalid_runs.py)
- Maintenance script for pruning invalid or partial stored runs and refreshing summaries.

[`visualize.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/visualize.py)
- Thin CLI wrapper around [`viz/render.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/viz/render.py).

[`README.md`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/README.md)
- User-facing setup and usage guide.

## Config

[`config/common.yaml`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/config/common.yaml)
- Shared batch defaults.
- Declares `max_concurrency_by_company`, fallback retry/thinking settings, duration, and named game selections.

[`config/runs.yaml`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/config/runs.yaml)
- Main config-driven batch settings.
- Each entry declares model, thinking mode, prompt mode, target games, seed behavior, and run count.

[`config/sample_runs.yaml`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/config/sample_runs.yaml)
- Smaller example/debug batch config.

## Core Runtime

[`core/pipeline.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/core/pipeline.py)
- Main environment loop.
- Builds prompts, queries the active model client, executes actions, records turn/frame data, and emits the run summary.

[`core/clip.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/core/clip.py)
- Parses raw model text into normalized thoughts and executable action sequences.

[`core/trajectory.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/core/trajectory.py)
- Persists frames, prompts, responses, turn records, prompt HTML, and final summary artifacts.

## Games

[`games/registry.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/games/registry.py)
- Discovers prompt-backed games and builds `GameSpec` entries.
- Exposes named presets such as `selected` and `full`.

[`games/env.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/games/env.py)
- ALE environment creation plus normalized frame/info/life-loss helpers.

[`games/prompt_builder.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/games/prompt_builder.py)
- Builds `structured_history` and `append_only` prompt packages.
- Produces both flattened prompt text and, when needed, structured message lists for provider adapters.

[`games/prompts/`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/games/prompts)
- Maintained per-game prompt modules and shared prompt helpers.

## LLM Layer

[`llm/common.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/common.py)
- Shared model/provider inference and thinking-mode resolution.
- Loads supported model/mode pairs from `model_thinking.json`.

[`llm/__init__.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/__init__.py)
- Public client factory and LLM-layer exports.

[`llm/gemini_client.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/gemini_client.py)
- Gemini adapter.

[`llm/openai_client.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/openai_client.py)
- OpenAI adapter.

[`llm/anthropic_client.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/anthropic_client.py)
- Anthropic adapter.

[`llm/retry.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/retry.py)
- Shared retry classification and backoff helpers for provider calls.

[`llm/model_thinking.json`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/model_thinking.json)
- Declared supported thinking modes per model.

## Visualization

[`viz/render.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/viz/render.py)
- Loads stored frame and turn artifacts and encodes the whiteboard-style video output.

## Tests

[`tests/test_pipeline.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_pipeline.py)
- Pipeline behavior, prompt modes, and persisted metadata.

[`tests/test_batch_run.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_batch_run.py)
- Batch parsing, config merging, scheduling, retry behavior, and subprocess command shape.

[`tests/test_games.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_games.py)
- Registry and game-selection coverage.

[`tests/test_llm.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_llm.py)
- Provider adapter behavior and retry helpers.

[`tests/test_model_thinking_options.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_model_thinking_options.py)
- Thinking-mode compatibility checks, including optional live-provider coverage.

[`tests/test_run_storage.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_run_storage.py)
- Canonical storage layout and model-summary aggregation.

[`tests/test_clip.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_clip.py)
- Response parsing behavior.

[`tests/test_trajectory.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_trajectory.py)
- Artifact persistence and turn/frame recording.

[`tests/test_cleanup_invalid_runs.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_cleanup_invalid_runs.py)
- Cleanup-script coverage.

## Peripheral Area

[`computer_use/`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/computer_use)
- Separate manual-control and ALE setup experiments.
- Not part of the maintained `main.py` / `batch_run.py` execution path.
