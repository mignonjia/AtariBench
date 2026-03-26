# Repository Index

This index covers the source tree and intentionally excludes `runs/`.

## Top Level

`main.py`
- Single-run CLI.
- Builds `PipelineConfig`, resolves output layout, constructs the model client, and executes `PipelineRunner`.

`batch_run.py`
- Batch CLI.
- Expands `MODEL:COUNT[:THINKING]` jobs, runs subprocess-backed jobs with bounded concurrency, retries transient failures, and optionally renders videos.

`run_storage.py`
- Canonical on-disk layout helpers for per-game and per-model run storage.
- Recomputes `model_summary.json` for canonical games.

`visualize.py`
- CLI entrypoint for rendering a stored run via `viz.render`.

`README.md`
- User-facing setup and execution examples.

## Core Runtime

`core/pipeline.py`
- Main environment loop.
- Builds prompts, calls the LLM client, executes actions frame-by-frame, records trajectory data, and emits the final summary.

`core/clip.py`
- Parses raw model text into a normalized thought and action list.

`core/trajectory.py`
- Persists frames, prompts, responses, turn metadata, and run summaries.

## Game Layer

`games/registry.py`
- Defines `GameSpec` and the supported game registry.
- Current canonical games in the main registry: `assault`, `breakout`.

`games/env.py`
- Environment creation and frame/info normalization helpers.

`games/prompt_builder.py`
- Builds the multimodal prompt payload from recent and reward-bearing trajectory clips.

`games/prompts/`
- Per-game prompt modules plus shared prompt templates.
- The registry imports these modules to obtain action maps and prompt text.

## Model Layer

`llm/common.py`
- Thinking-mode metadata and provider inference/resolution.

`llm/gemini.py`
- Gemini adapter.

`llm/openai_client.py`
- OpenAI Responses API adapter with image inlining.

`llm/__init__.py`
- Exports model clients and `build_model_client`.

## Visualization

`viz/render.py`
- Generates whiteboard-style videos from stored frames and `turns.jsonl`.

## Tests

`tests/test_pipeline.py`
- Main runtime behavior: frame budget, life loss, parse fallback, thinking metadata.

`tests/test_batch_run.py`
- Batch job parsing, subprocess command shape, retry logic, fallback thinking behavior.

`tests/test_games.py`
- Registry and prompt/game layer coverage.

`tests/test_llm.py`
- Provider selection and model-layer helpers.

`tests/test_run_storage.py`
- Canonical storage layout and summary generation.

`tests/test_clip.py`
- Response parsing behavior.

`tests/test_trajectory.py`
- Turn/frame persistence behavior.

## Peripheral Area

`computer_use/`
- Separate manual-control and ALE setup experiments.
- Not part of the main `main.py` / `batch_run.py` execution path.
