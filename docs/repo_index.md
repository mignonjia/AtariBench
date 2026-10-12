# Repository Index

This index covers the maintained source tree and intentionally excludes stored run data and legacy prompt variants.

## Top Level

[`../pyproject.toml`](../pyproject.toml)
- Package metadata and console script entrypoints.

[`../main.py`](../main.py), [`../batch_run.py`](../batch_run.py)
- Compatibility wrappers for the installable package entrypoints.

[`../README.md`](../README.md)
- User-facing setup and usage guide.

[`logging/README.md`](logging/README.md)
- Detailed run-output layout, model-summary files, and batch metadata logging guide.

[`index.html`](index.html), [`app.js`](app.js), [`styles.css`](styles.css)
- GitHub Pages project website served from the repository `docs/` directory.

[`assets/`](assets/), [`data/leaderboard.json`](data/leaderboard.json)
- Website media assets and leaderboard data used by the project page.

## Package Entrypoints

[`../src/ataribench/main.py`](../src/ataribench/main.py)
- Single-run CLI exposed as `ataribench-run`.
- Validates the model/thinking pair, resolves the output layout, runs one pipeline, and writes back video metadata into `summary.json`.

[`../src/ataribench/batch_run.py`](../src/ataribench/batch_run.py)
- Batch orchestration CLI exposed as `ataribench-batch`.
- Supports config-driven batches from `config/common.yaml` plus `config/runs.yaml`.
- Expands game selections, enforces company-level concurrency caps, retries incomplete runs, and renders videos for successful runs.

[`../src/ataribench/run_storage.py`](../src/ataribench/run_storage.py)
- Canonical storage helpers for maintained games.
- Resolves per-game/per-model output roots and rebuilds per-game plus cross-game summary files for both all-success runs and 30-second-only runs.
- Groups summary averages by full setting key, including prompt and thinking metadata.

[`../src/ataribench/cleanup_invalid_runs.py`](../src/ataribench/cleanup_invalid_runs.py)
- Maintenance script for pruning invalid or partial stored runs and refreshing summaries.

[`../src/ataribench/visualize.py`](../src/ataribench/visualize.py)
- Thin CLI wrapper around [`../src/ataribench/viz/render.py`](../src/ataribench/viz/render.py).

## Config

[`../config/common.yaml`](../config/common.yaml)
- Shared batch defaults.
- Declares `max_concurrency_by_company`, fallback retry/thinking settings, duration, and named game selections.

[`../config/runs.yaml`](../config/runs.yaml)
- Main config-driven batch settings.
- Each entry declares model, thinking mode, prompt mode, target games, seed behavior, and run count.

[`../config/debug.yaml`](../config/debug.yaml)
- Smaller debug batch config.

## Core Runtime

[`../src/ataribench/core/pipeline.py`](../src/ataribench/core/pipeline.py)
- Main environment loop.
- Builds prompts, queries the active model client, executes actions, records turn/frame data, and emits the run summary.

[`../src/ataribench/core/clip.py`](../src/ataribench/core/clip.py)
- Parses raw model text into normalized thoughts and executable action sequences.

[`../src/ataribench/core/trajectory.py`](../src/ataribench/core/trajectory.py)
- Persists frames, prompts, responses, turn records, prompt HTML, and final summary artifacts.

## Games

[`../src/ataribench/games/registry.py`](../src/ataribench/games/registry.py)
- Discovers prompt-backed games and builds `GameSpec` entries.
- Exposes the named preset `full` (all registered games).

[`../src/ataribench/games/env.py`](../src/ataribench/games/env.py)
- ALE environment creation plus normalized frame/info/life-loss helpers.

[`../src/ataribench/games/prompt_builder.py`](../src/ataribench/games/prompt_builder.py)
- Builds `structured_history` and `append_only` prompt packages.
- Produces both flattened prompt text and, when needed, structured message lists for provider adapters.

[`../src/ataribench/games/prompts/`](../src/ataribench/games/prompts/)
- Maintained per-game prompt modules and shared prompt helpers.

## LLM Layer

[`../src/ataribench/llm/common.py`](../src/ataribench/llm/common.py)
- Shared model/provider inference and thinking-mode resolution.
- Loads supported model/mode pairs from `model_thinking.json`.

[`../src/ataribench/llm/__init__.py`](../src/ataribench/llm/__init__.py)
- Public client factory and LLM-layer exports.

[`../src/ataribench/llm/gemini_client.py`](../src/ataribench/llm/gemini_client.py)
- Gemini adapter.

[`../src/ataribench/llm/openai_client.py`](../src/ataribench/llm/openai_client.py)
- OpenAI adapter.

[`../src/ataribench/llm/anthropic_client.py`](../src/ataribench/llm/anthropic_client.py)
- Anthropic adapter.

[`../src/ataribench/llm/together_client.py`](../src/ataribench/llm/together_client.py)
- Together adapter.

[`../src/ataribench/llm/retry.py`](../src/ataribench/llm/retry.py)
- Shared retry classification and backoff helpers for provider calls.
- Keeps transient provider failures inside the active turn instead of escalating them to batch-level reruns.

[`../src/ataribench/llm/model_thinking.json`](../src/ataribench/llm/model_thinking.json)
- Declared supported thinking modes per model.

## Visualization

[`../src/ataribench/viz/render.py`](../src/ataribench/viz/render.py)
- Loads stored frame and turn artifacts and encodes the whiteboard-style video output.

## Tests

[`../tests/test_pipeline.py`](../tests/test_pipeline.py)
- Pipeline behavior, prompt modes, and persisted metadata.

[`../tests/test_batch_run.py`](../tests/test_batch_run.py)
- Batch parsing, config merging, scheduling, retry behavior, and subprocess command shape.

[`../tests/test_games.py`](../tests/test_games.py)
- Registry and game-selection coverage.

[`../tests/test_llm.py`](../tests/test_llm.py)
- Provider adapter behavior and retry helpers.

[`../tests/test_run_storage.py`](../tests/test_run_storage.py)
- Canonical storage layout and model-summary aggregation.

[`../tests/test_clip.py`](../tests/test_clip.py)
- Response parsing behavior.

[`../tests/test_trajectory.py`](../tests/test_trajectory.py)
- Artifact persistence and turn/frame recording.

[`../tests/test_cleanup_invalid_runs.py`](../tests/test_cleanup_invalid_runs.py)
- Cleanup-script coverage.

## Local-Only Areas

Local experiments, stored run data, and paper scratch outputs are ignored by default and intentionally omitted from this maintained source index.
