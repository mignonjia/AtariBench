# AtariBench Agent Guide

Compact Atari runner for multimodal LLM experiments.

Scope:
- Ignore generated artifacts such as `.git/`, `__pycache__/`, and `.DS_Store`.
- Treat `src/ataribench/llm/api.sh` as local environment glue, not a source-of-truth interface.
- Keep `website/` standalone during website layout work; do not move it into the Python package.

Start here:
- [docs/repo_index.md](docs/repo_index.md)

Primary CLIs:
- [`main.py`](main.py): single-run CLI
- [`batch_run.py`](batch_run.py): config-driven batch orchestration
- `ataribench-visualize`: video rendering CLI; implementation lives in [`src/ataribench/visualize.py`](src/ataribench/visualize.py)
- `ataribench-clean`: dry-run or apply cleanup for invalid/non-30-second stored runs; implementation lives in [`src/ataribench/cleanup_invalid_runs.py`](src/ataribench/cleanup_invalid_runs.py)

Config files:
- [`environment.yaml`](environment.yaml): single source of truth for the Conda environment and `ffmpeg`
- [`config/common.yaml`](config/common.yaml): shared batch defaults
- `common.yaml` also defines named game selections, fallback retries, optional `context_cache`, optional `minimal_logging`, and provider-level concurrency caps via `max_concurrency_by_company`.
- [`config/runs.yaml`](config/runs.yaml): per-setting batch entries
- [`config/debug.yaml`](config/debug.yaml) is an additional debug-oriented run config.

Main source areas:
- `src/ataribench/core/`: pipeline loop, parsing, trajectory persistence
- `src/ataribench/games/`: registry, env helpers, prompt assembly, per-game prompts
- `src/ataribench/llm/`: provider adapters, thinking-mode logic, and transient retry handling
- `runs/`: persisted run artifacts and aggregated summary files
- `src/ataribench/viz/`: whiteboard-style video rendering
- `tests/`: unit coverage for runner paths

Summary files:
- Per-run `summary.json` stores the final metrics and config metadata for one completed run.
- Per-game `model_summary.json` groups all successful runs by setting under one game and reports averages plus the latest run pointer.
- Per-game `model_summary_30s.json` keeps the 30-second-only benchmark view for that same game.
- Top-level `runs/model_summary.json` flattens the per-game all-success summaries into one cross-game list of entries.
- Top-level `runs/model_summary_30s.json` flattens the per-game 30-second-only summaries into one cross-game list of entries.
- Aggregation is per setting key, so prompt mode, thinking fields, and clip settings are not mixed into one average.

Working guidance:
- Keep `src/ataribench/main.py`, `src/ataribench/batch_run.py`, `src/ataribench/core/pipeline.py`, persisted `summary.json` fields, `src/ataribench/run_storage.py`, and docs aligned when changing runtime config.
- Keep the main public runtime workflows centered on `ataribench-run` and config-driven `ataribench-batch`; treat richer per-run knobs as config-driven or internal plumbing unless there is a strong reason to expose them on the CLI.
- Keep `src/ataribench/visualize.py`, `src/ataribench/cleanup_invalid_runs.py`, and their help text aligned with the artifact layout in `src/ataribench/core/trajectory.py` and `src/ataribench/run_storage.py`.
- Keep per-game and top-level `model_summary.json` schemas aligned with `src/ataribench/run_storage.py`, `README.md`, and `tests/test_run_storage.py`.
- Keep `config/common.yaml`, `config/runs.yaml`, `environment.yaml`, CLI flags, and `tests/test_batch_run.py` aligned when changing batch config behavior.
- Keep `src/ataribench/llm/common.py`, `src/ataribench/llm/model_thinking.json`, `src/ataribench/llm/retry.py`, provider adapters, and tests aligned when changing provider behavior.
- Keep `src/ataribench/games/registry.py`, `src/ataribench/games/prompts/`, and `tests/test_games.py` aligned when adding or removing supported games.
- Keep `tests/test_main.py` aligned with `src/ataribench/main.py` when changing single-run CLI behavior.

Useful commands:
- `ataribench-run --game breakout --model gemini-2.5-flash --thinking off --prompt-mode structured_history --duration-seconds 30`
- `ataribench-batch --common-config config/common.yaml --runs-config config/debug.yaml`
- `ataribench-clean --project-dir .`
- `python -m unittest`

Directory notes:
- [src/ataribench/core/AGENTS.md](src/ataribench/core/AGENTS.md)
- [src/ataribench/games/AGENTS.md](src/ataribench/games/AGENTS.md)
- [src/ataribench/llm/AGENTS.md](src/ataribench/llm/AGENTS.md)
- [src/ataribench/viz/AGENTS.md](src/ataribench/viz/AGENTS.md)
- [tests/AGENTS.md](tests/AGENTS.md)
