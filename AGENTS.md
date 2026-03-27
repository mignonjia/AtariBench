# AtariBench Agent Guide

This repo is a compact Atari runner for multimodal LLM experiments.

Scope rules:
- Ignore `runs/`. It is large, generated, and not part of the source index unless a task explicitly targets stored results.
- Ignore `.git/`, `__pycache__/`, `.DS_Store`, and other generated artifacts.
- Treat `llm/api.sh` as local environment glue, not a source-of-truth API contract.

Primary entrypoints:
- `main.py`: single-run CLI
- `batch_run.py`: batch orchestration and retry logic
- `visualize.py`: render stored runs into videos

Main source areas:
- `core/`: pipeline loop, response parsing, trajectory persistence
- `games/`: game registry, env helpers, prompt assembly, per-game prompt specs
- `llm/`: Gemini/OpenAI client adapters and provider selection
- `viz/`: whiteboard-style video rendering
- `tests/`: unit coverage for the core runner paths
- `computer_use/`: separate manual-control experiments, not part of the main runner path

Working guidance:
- Start with [REPO_INDEX.md](/Users/mingjiahuo/Desktop/ataribench/AtariBench/REPO_INDEX.md).
- Prefer edits in the main runner path before touching `computer_use/`.
- When adding a game, update `games/registry.py`, add a prompt module under `games/prompts/`, and extend tests.
- When changing provider behavior, keep `llm/common.py`, provider adapters, and tests aligned.
- When changing CLI/runtime config, keep `main.py`, `batch_run.py`, persisted `summary.json` fields, `run_storage.py`, and docs aligned.

Useful commands:
- `python main.py --game breakout --model gemini-2.5-flash --thinking off --prompt-mode structured_history --duration-seconds 30`
- `python main.py --game breakout --model gemini-2.5-flash --thinking off --prompt-mode append_only --duration-seconds 5`
- `python batch_run.py --game breakout --job gpt-5.4-mini:1:off --max-concurrency 1`
- `python -m unittest`
- `python -m unittest tests.test_pipeline tests.test_batch_run`

Directory-level agent notes live in:
- [core/AGENTS.md](/Users/mingjiahuo/Desktop/ataribench/AtariBench/core/AGENTS.md)
- [games/AGENTS.md](/Users/mingjiahuo/Desktop/ataribench/AtariBench/games/AGENTS.md)
- [llm/AGENTS.md](/Users/mingjiahuo/Desktop/ataribench/AtariBench/llm/AGENTS.md)
- [viz/AGENTS.md](/Users/mingjiahuo/Desktop/ataribench/AtariBench/viz/AGENTS.md)
- [tests/AGENTS.md](/Users/mingjiahuo/Desktop/ataribench/AtariBench/tests/AGENTS.md)
- [computer_use/AGENTS.md](/Users/mingjiahuo/Desktop/ataribench/AtariBench/computer_use/AGENTS.md)
