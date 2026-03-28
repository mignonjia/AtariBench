# AtariBench Agent Guide

Compact Atari runner for multimodal LLM experiments.

Scope:
- Ignore generated artifacts such as `.git/`, `__pycache__/`, and `.DS_Store`.
- Treat `llm/api.sh` as local environment glue, not a source-of-truth interface.

Start here:
- [REPO_INDEX.md](REPO_INDEX.md)

Primary entrypoints:
- [`main.py`](main.py): single-run CLI
- [`batch_run.py`](batch_run.py): batch orchestration
- [`visualize.py`](visualize.py): video rendering

Config files:
- [`environment.yaml`](environment.yaml): single source of truth for the Conda environment and `ffmpeg`
- [`config/common.yaml`](config/common.yaml): shared batch defaults
- `common.yaml` also defines named game selections and provider-level concurrency caps via `max_concurrency_by_company`.
- [`config/runs.yaml`](config/runs.yaml): per-setting batch entries
- [`config/sample_runs.yaml`](config/sample_runs.yaml) is a smaller example config.

Main source areas:
- `core/`: pipeline loop, parsing, trajectory persistence
- `games/`: registry, env helpers, prompt assembly, per-game prompts
- `llm/`: provider adapters and thinking-mode logic
- `runs/`: persisted run artifacts and aggregated summary files
- `viz/`: whiteboard-style video rendering
- `tests/`: unit coverage for runner paths
- `computer_use/`: separate experiments, not part of the main runner path

Summary files:
- Per-run `summary.json` stores the final metrics and config metadata for one completed run.
- Per-game `model_summary.json` groups all successful runs by setting under one game and reports averages plus the latest run pointer.
- Per-game `model_summary_30s.json` keeps the 30-second-only benchmark view for that same game.
- Top-level `runs/model_summary.json` flattens the per-game all-success summaries into one cross-game list of entries.
- Top-level `runs/model_summary_30s.json` flattens the per-game 30-second-only summaries into one cross-game list of entries.
- Aggregation is per setting key, so prompt mode, thinking fields, and clip settings are not mixed into one average.

Working guidance:
- Keep `main.py`, `batch_run.py`, `core/pipeline.py`, persisted `summary.json` fields, `run_storage.py`, and docs aligned when changing runtime config.
- Maintain only two supported public command shapes: one single-run command via `main.py` and one config-driven batch command via `batch_run.py`.
- Treat richer batch-run settings as config-driven or internal plumbing, not as public CLI surface area.
- Keep per-game and top-level `model_summary.json` schemas aligned with `run_storage.py`, `README.md`, and `tests/test_run_storage.py`.
- Keep `config/common.yaml`, `config/runs.yaml`, `environment.yaml`, CLI flags, and `tests/test_batch_run.py` aligned when changing batch config behavior.
- Keep `llm/common.py`, `llm/model_thinking.json`, provider adapters, and tests aligned when changing provider behavior.
- Keep `games/registry.py`, `games/prompts/`, and `tests/test_games.py` aligned when adding or removing supported games.
- Prefer main-runner edits before touching `computer_use/`.

Useful commands:
- `python main.py --game breakout --model gemini-2.5-flash --thinking off --prompt-mode structured_history --duration-seconds 30`
- `python batch_run.py --common-config config/common.yaml --runs-config config/sample_runs.yaml`
- `python -m unittest`

Directory notes:
- [core/AGENTS.md](core/AGENTS.md)
- [games/AGENTS.md](games/AGENTS.md)
- [llm/AGENTS.md](llm/AGENTS.md)
- [runs/AGENTS.md](runs/AGENTS.md)
- [computer_use/AGENTS.md](computer_use/AGENTS.md)
- [viz/AGENTS.md](viz/AGENTS.md)
- [tests/AGENTS.md](tests/AGENTS.md)
