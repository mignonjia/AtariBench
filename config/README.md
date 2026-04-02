# Config Directory

This folder contains the config-driven batch inputs for AtariBench.

## Files

- [`common.yaml`](common.yaml): shared batch defaults such as game selections, retries, concurrency caps, duration, rendering settings, and optional `minimal_logging`
- [`runs.yaml`](runs.yaml): the full set of batch run definitions
- [`debug.yaml`](debug.yaml): a smaller debug batch for quick iteration

## Usage - debug

Use `debug.yaml` when you want a fast debug run:

```bash
python batch_run.py --common-config config/common.yaml --runs-config config/debug.yaml
```

If you want even faster debug cycles, lower `duration_seconds` in [`common.yaml`](common.yaml).

## Usage - complete run

Use `runs.yaml` when you want the full batch:

```bash
python batch_run.py --common-config config/common.yaml --runs-config config/runs.yaml
```

You can also add option `--minimal-logging` to only keep `summary.json`, `turns.jsonl`, and `visualization.mp4` in each run directory after rendering. This is sufficient to generate the result files `model_summary_30s.json` in `runs` folder.


## Logging

See [`../runs/breakout/gemini-2.5-flash/0401_020222_cfg_001_run_001`](../runs/breakout/gemini-2.5-flash/0401_020222_cfg_001_run_001) for an example run of complete logging, and [`../runs/breakout/gemini-2.5-flash/0331_235632_cfg_001_run_001`](../runs/breakout/gemini-2.5-flash/0331_235632_cfg_001_run_001) for an example run of minimal logging
