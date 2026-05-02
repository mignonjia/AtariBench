# Robotank Left/Right Action Analysis

Included runs have `summary.json` with `duration_seconds == 30` and a `turns.jsonl` file.
Only non-empty `planned_action_strings` lists are counted. Missing or empty planned-action turns are skipped.
Left/right actions are planned actions whose names contain `left` or `right`, case-insensitively.

| Source | Model | Thinking Mode | Thinking Level | Thinking Budget | 30s Runs | Total Actions | Left Actions | Left % | Right Actions | Right % | Skipped Turns |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ataribench_videos/structured_history/runs/robotank | google:gemini-3-flash-preview | none | high | none | 10 | 3009 | 591 | 19.64 | 676 | 22.47 | 0 |
| ataribench_videos/structured_history/runs/robotank | google:gemini-3-flash-preview | none | low | none | 10 | 3019 | 665 | 22.03 | 519 | 17.19 | 0 |
| ataribench_videos/structured_history/runs/robotank | google:gemini-3.1-pro-preview | none | high | none | 10 | 3016 | 868 | 28.78 | 763 | 25.30 | 0 |
| ataribench_videos/structured_history/runs/robotank | google:gemini-3.1-pro-preview | none | low | none | 10 | 3009 | 927 | 30.81 | 696 | 23.13 | 0 |
| runs/robotank | deepseek-ai/deepseek-v3.1 | off | none | none | 10 | 3003 | 1550 | 51.62 | 1088 | 36.23 | 0 |
| runs/robotank | gemini-2.5-flash | off | none | 0 | 10 | 3011 | 1579 | 52.44 | 1148 | 38.13 | 0 |
| runs/robotank | gpt-5.4-mini | none | none | none | 10 | 3012 | 150 | 4.98 | 71 | 2.36 | 0 |
| runs/robotank | random | off | none | none | 10 | 3004 | 963 | 32.06 | 1012 | 33.69 | 0 |
| runs/robotank | zai-org/glm-5.1 | off | none | none | 10 | 3000 | 1366 | 45.53 | 340 | 11.33 | 0 |

## Skipped Entries

- `missing_or_invalid_summary`: 1
