# Riverraid Fire Action Analysis

Included runs have `summary.json` with `duration_seconds == 30` and a `turns.jsonl` file.
Only non-empty `planned_action_strings` lists are counted. Missing or empty planned-action turns are skipped.
A fire action is any planned action whose name contains `fire`, case-insensitively.

| Source | Model | Thinking Mode | Thinking Level | Thinking Budget | 30s Runs | Total Actions | Fire Actions | Fire % | Left Actions | Left % | Right Actions | Right % | Skipped Turns |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ataribench_videos/structured_history/runs/riverraid | google:gemini-3-flash-preview | none | high | none | 10 | 3024 | 1450 | 47.95 | 535 | 17.69 | 851 | 28.14 | 0 |
| ataribench_videos/structured_history/runs/riverraid | google:gemini-3-flash-preview | none | low | none | 10 | 3013 | 1317 | 43.71 | 619 | 20.54 | 661 | 21.94 | 0 |
| ataribench_videos/structured_history/runs/riverraid | google:gemini-3.1-pro-preview | none | high | none | 10 | 3001 | 456 | 15.19 | 304 | 10.13 | 515 | 17.16 | 0 |
| ataribench_videos/structured_history/runs/riverraid | google:gemini-3.1-pro-preview | none | low | none | 10 | 3014 | 443 | 14.70 | 361 | 11.98 | 595 | 19.74 | 0 |
| runs/riverraid | deepseek-ai/deepseek-v3.1 | off | none | none | 9 | 2705 | 2249 | 83.14 | 29 | 1.07 | 89 | 3.29 | 0 |
| runs/riverraid | gemini-2.5-flash | off | none | 0 | 10 | 3015 | 2071 | 68.69 | 900 | 29.85 | 976 | 32.37 | 0 |
| runs/riverraid | gpt-5.4-mini | none | none | none | 10 | 3012 | 1320 | 43.82 | 7 | 0.23 | 264 | 8.76 | 0 |
| runs/riverraid | random | off | none | none | 10 | 3005 | 1455 | 48.42 | 1021 | 33.98 | 970 | 32.28 | 0 |
| runs/riverraid | zai-org/glm-5.1 | off | none | none | 10 | 3007 | 2877 | 95.68 | 249 | 8.28 | 590 | 19.62 | 0 |

## Skipped Entries

- `missing_or_invalid_summary`: 10
