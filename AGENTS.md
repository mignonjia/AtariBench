# AtariBench Agent Guide

Compact Atari runner for multimodal LLM experiments.

Scope:
- Ignore `runs/` unless a task explicitly targets stored results.
- Ignore generated artifacts such as `.git/`, `__pycache__/`, and `.DS_Store`.
- Treat `llm/api.sh` as local environment glue, not a source-of-truth interface.

Start here:
- [REPO_INDEX.md](/Users/mingjiahuo/Desktop/ataribench/AtariBench/REPO_INDEX.md)

Primary entrypoints:
- [`main.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/main.py): single-run CLI
- [`batch_run.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/batch_run.py): batch orchestration
- [`visualize.py`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/visualize.py): video rendering

Config files:
- [`config/common.yaml`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/config/common.yaml): shared batch defaults
- [`config/runs.yaml`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/config/runs.yaml): per-setting batch entries
  [`config/sample_runs.yaml`](/Users/mingjiahuo/Desktop/ataribench/AtariBench/config/sample_runs.yaml) is a smaller example config.
  `common.yaml` may include `max_concurrency_by_company` for provider-level caps.

Main source areas:
- `core/`: pipeline loop, parsing, trajectory persistence
- `games/`: registry, env helpers, prompt assembly, per-game prompts
- `llm/`: provider adapters and thinking-mode logic
- `viz/`: whiteboard-style video rendering
- `tests/`: unit coverage for runner paths
- `computer_use/`: separate experiments, not part of the main runner path

Working guidance:
- Keep `main.py`, `batch_run.py`, persisted `summary.json` fields, `run_storage.py`, and docs aligned when changing runtime config.
- Keep `llm/common.py`, provider adapters, and tests aligned when changing provider behavior.
- Prefer main-runner edits before touching `computer_use/`.

Useful commands:
- `python main.py --game breakout --model gemini-2.5-flash --thinking off --prompt-mode structured_history --duration-seconds 30`
- `python main.py --game breakout --model gemini-2.5-flash --thinking off --prompt-mode append_only --duration-seconds 5`
- `python batch_run.py --game selected --job gemini-2.5-flash:1:off --max-concurrency 1`
- `python batch_run.py --common-config config/common.yaml --model-game-specific-config config/runs.yaml`
- `python -m unittest`

Directory notes:
- [core/AGENTS.md](/Users/mingjiahuo/Desktop/ataribench/AtariBench/core/AGENTS.md)
- [games/AGENTS.md](/Users/mingjiahuo/Desktop/ataribench/AtariBench/games/AGENTS.md)
- [llm/AGENTS.md](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/AGENTS.md)
- [viz/AGENTS.md](/Users/mingjiahuo/Desktop/ataribench/AtariBench/viz/AGENTS.md)
- [tests/AGENTS.md](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/AGENTS.md)
