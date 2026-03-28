# Repository Index

This index covers the source tree and intentionally excludes `runs/`.

## Top Level

[`main.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/main.py)
- Single-run CLI.
- Builds `PipelineConfig`, runs one game/model setting, stores artifacts, and renders video.

[`batch_run.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/batch_run.py)
- Batch CLI.
- Supports job-driven runs and config-driven runs via `config/common.yaml` plus `config/runs.yaml`.
- Expands game selections, retries transient failures, and renders video for successful runs.

[`run_storage.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/run_storage.py)
- Canonical run-layout helpers.
- Rebuilds per-game and cross-game summaries.
- Summary averages are grouped per setting, not across mixed thinking/prompt/clip settings.

[`cleanup_invalid_runs.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/cleanup_invalid_runs.py)
- Optional maintenance script for deleting incomplete, invalid, or non-30-second runs and refreshing summaries.

[`visualize.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/visualize.py)
- CLI entrypoint for rendering a stored run via `viz.render`.

[`config/common.yaml`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/config/common.yaml)
- Shared batch defaults.

[`config/runs.yaml`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/config/runs.yaml)
- Per-setting batch entries such as model, thinking mode, prompt mode, games, and run count.

[`config/sample_runs.yaml`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/config/sample_runs.yaml)
- Smaller debug/example batch config.

## Core Runtime

[`core/pipeline.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/core/pipeline.py)
- Main environment loop.
- Builds prompts, calls the LLM client, executes actions, records trajectory data, and emits the final summary.

[`core/clip.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/core/clip.py)
- Parses raw model text into normalized thoughts and actions.

[`core/trajectory.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/core/trajectory.py)
- Persists frames, prompts, responses, turn data, and run summaries.
- Renders prompt HTML, including append-only chat-style transcripts.

## Game Layer

[`games/registry.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/games/registry.py)
- Defines `GameSpec`, discovers all prompt-backed games, and exposes named selections such as `selected` and `full`.

[`games/env.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/games/env.py)
- ALE environment creation and frame/info helpers.

[`games/prompt_builder.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/games/prompt_builder.py)
- Builds prompt payloads for `structured_history` and `append_only`.
- Append-only can emit structured user/assistant messages for the provider adapters.

[`games/prompts/`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/games/prompts)
- Per-game prompt modules plus shared prompt templates.

## Model Layer

[`llm/common.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/common.py)
- Provider inference, supported-thinking validation, and effective thinking resolution.

[`llm/gemini_client.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/gemini_client.py)
- Gemini adapter.

[`llm/openai_client.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/openai_client.py)
- OpenAI adapter.

[`llm/anthropic_client.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/anthropic_client.py)
- Anthropic adapter.

[`llm/model_thinking.json`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/model_thinking.json)
- Declares the supported thinking modes for each model.

## Visualization

[`viz/render.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/viz/render.py)
- Generates whiteboard-style videos from stored frames and `turns.jsonl`.

## Tests

[`tests/test_pipeline.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_pipeline.py)
- Pipeline behavior, prompt modes, and persisted metadata.

[`tests/test_batch_run.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_batch_run.py)
- Batch job/config parsing, subprocess command shape, retry behavior, and game selection expansion.

[`tests/test_games.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_games.py)
- Registry and game-selection coverage.

[`tests/test_llm.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_llm.py)
- Provider adapters, request-shape logic, and retry helpers.

[`tests/test_model_thinking_options.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_model_thinking_options.py)
- Thinking-mode compatibility checks, with optional live provider requests.

[`tests/test_run_storage.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_run_storage.py)
- Canonical storage layout and summary aggregation.

[`tests/test_cleanup_invalid_runs.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/test_cleanup_invalid_runs.py)
- Cleanup-script coverage.

## Peripheral Area

[`computer_use/`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/computer_use)
- Separate manual-control experiments.
- Not part of the main `main.py` / `batch_run.py` execution path.
