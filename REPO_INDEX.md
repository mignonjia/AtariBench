# Repository Index

This index covers the maintained source tree and intentionally excludes stored run data and legacy prompt variants.

## Top Level

[`main.py`](main.py)
- Single-run CLI.
- Validates the model/thinking pair, resolves the output layout, runs one pipeline, and writes back video metadata into `summary.json`.

[`batch_run.py`](batch_run.py)
- Batch orchestration CLI.
- Supports ad hoc `--job` specs and config-driven batches from `config/common.yaml` plus `config/runs.yaml`.
- Expands game selections, enforces company-level concurrency caps, retries incomplete runs, and renders videos for successful runs.

[`run_storage.py`](run_storage.py)
- Canonical storage helpers for maintained games.
- Resolves per-game/per-model output roots and rebuilds per-game plus cross-game summary files for both all-success runs and 30-second-only runs.
- Groups summary averages by full setting key, including prompt and thinking metadata.

[`cleanup_invalid_runs.py`](cleanup_invalid_runs.py)
- Maintenance script for pruning invalid or partial stored runs and refreshing summaries.

[`visualize.py`](visualize.py)
- Thin CLI wrapper around [`viz/render.py`](viz/render.py).

[`README.md`](README.md)
- User-facing setup and usage guide.

## Config

[`config/common.yaml`](config/common.yaml)
- Shared batch defaults.
- Declares `max_concurrency_by_company`, fallback retry/thinking settings, duration, and named game selections.

[`config/runs.yaml`](config/runs.yaml)
- Main config-driven batch settings.
- Each entry declares model, thinking mode, prompt mode, target games, seed behavior, and run count.

[`config/sample_runs.yaml`](config/sample_runs.yaml)
- Smaller example/debug batch config.

## Core Runtime

[`core/pipeline.py`](core/pipeline.py)
- Main environment loop.
- Builds prompts, queries the active model client, executes actions, records turn/frame data, and emits the run summary.

[`core/clip.py`](core/clip.py)
- Parses raw model text into normalized thoughts and executable action sequences.

[`core/trajectory.py`](core/trajectory.py)
- Persists frames, prompts, responses, turn records, prompt HTML, and final summary artifacts.

## Games

[`games/registry.py`](games/registry.py)
- Discovers prompt-backed games and builds `GameSpec` entries.
- Exposes named presets such as `selected` and `full`.

[`games/env.py`](games/env.py)
- ALE environment creation plus normalized frame/info/life-loss helpers.

[`games/prompt_builder.py`](games/prompt_builder.py)
- Builds `structured_history` and `append_only` prompt packages.
- Produces both flattened prompt text and, when needed, structured message lists for provider adapters.

[`games/prompts/`](games/prompts/)
- Maintained per-game prompt modules and shared prompt helpers.

## LLM Layer

[`llm/common.py`](llm/common.py)
- Shared model/provider inference and thinking-mode resolution.
- Loads supported model/mode pairs from `model_thinking.json`.

[`llm/__init__.py`](llm/__init__.py)
- Public client factory and LLM-layer exports.

[`llm/gemini_client.py`](llm/gemini_client.py)
- Gemini adapter.

[`llm/openai_client.py`](llm/openai_client.py)
- OpenAI adapter.

[`llm/anthropic_client.py`](llm/anthropic_client.py)
- Anthropic adapter.

[`llm/retry.py`](llm/retry.py)
- Shared retry classification and backoff helpers for provider calls.
- Keeps transient provider failures inside the active turn instead of escalating them to batch-level reruns.

[`llm/model_thinking.json`](llm/model_thinking.json)
- Declared supported thinking modes per model.

## Visualization

[`viz/render.py`](viz/render.py)
- Loads stored frame and turn artifacts and encodes the whiteboard-style video output.

## Tests

[`tests/test_pipeline.py`](tests/test_pipeline.py)
- Pipeline behavior, prompt modes, and persisted metadata.

[`tests/test_batch_run.py`](tests/test_batch_run.py)
- Batch parsing, config merging, scheduling, retry behavior, and subprocess command shape.

[`tests/test_games.py`](tests/test_games.py)
- Registry and game-selection coverage.

[`tests/test_llm.py`](tests/test_llm.py)
- Provider adapter behavior and retry helpers.

[`tests/test_model_thinking_options.py`](tests/test_model_thinking_options.py)
- Thinking-mode compatibility checks, including optional live-provider coverage.

[`tests/test_run_storage.py`](tests/test_run_storage.py)
- Canonical storage layout and model-summary aggregation.

[`tests/test_clip.py`](tests/test_clip.py)
- Response parsing behavior.

[`tests/test_trajectory.py`](tests/test_trajectory.py)
- Artifact persistence and turn/frame recording.

[`tests/test_cleanup_invalid_runs.py`](tests/test_cleanup_invalid_runs.py)
- Cleanup-script coverage.

## Peripheral Area

[`computer_use/`](computer_use/)
- Separate manual-control and ALE setup experiments.
- Not part of the maintained `main.py` / `batch_run.py` execution path.
