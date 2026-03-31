# Config Directory

This folder contains the config-driven batch inputs for AtariBench.

## Files

- [`common.yaml`](common.yaml): shared batch defaults such as game selections, retries, concurrency caps, duration, rendering settings, and optional `minimal_logging`
- [`runs.yaml`](runs.yaml): the full set of batch run definitions
- [`sample_runs.yaml`](sample_runs.yaml): a smaller debug batch for quick iteration

## Usage

Use `sample_runs.yaml` when you want a fast debug run:

```bash
python batch_run.py --common-config config/common.yaml --runs-config config/debug.yaml
```

Use `runs.yaml` when you want the full batch:

```bash
python batch_run.py --common-config config/common.yaml --runs-config config/runs.yaml
```

If you want even faster debug cycles, lower `duration_seconds` in [`common.yaml`](common.yaml).
