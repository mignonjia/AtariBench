#!/usr/bin/env python3
"""Measure Riverraid action keyword frequencies for 30-second runs.

This analysis intentionally counts only parsed planned actions from
``planned_action_strings``. Turns without a usable non-empty planned action list
are skipped and do not contribute to either denominator or numerator.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent

RUN_ROOTS = [
    (
        "ataribench_videos/structured_history/runs/riverraid",
        REPO_ROOT / "ataribench_videos" / "structured_history" / "runs" / "riverraid",
    ),
    ("runs/riverraid", REPO_ROOT / "runs" / "riverraid"),
]


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def norm_field(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value)
    return text if text else "none"


def group_key(source: str, summary: dict[str, Any], model_dir_name: str) -> tuple[str, str, str, str, str]:
    model_name = norm_field(summary.get("model_name") or model_dir_name)
    thinking_mode = norm_field(summary.get("thinking_mode"))
    thinking_level = norm_field(summary.get("thinking_level"))
    thinking_budget = norm_field(summary.get("thinking_budget"))
    return (source, model_name, thinking_mode, thinking_level, thinking_budget)


def group_label(key: tuple[str, str, str, str, str]) -> str:
    source, model_name, thinking_mode, thinking_level, thinking_budget = key
    return (
        f"{model_name} | thinking_mode={thinking_mode} | "
        f"thinking_level={thinking_level} | thinking_budget={thinking_budget} | source={source}"
    )


def pct(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator * 100.0 / denominator


def analyze_turns(turns_path: Path) -> dict[str, Any]:
    total_turns = 0
    counted_turns = 0
    skipped_turns_missing_planned_actions = 0
    malformed_lines = 0
    total_actions = 0
    fire_actions = 0
    left_actions = 0
    right_actions = 0

    with turns_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_turns += 1
            try:
                turn = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                skipped_turns_missing_planned_actions += 1
                continue

            actions = turn.get("planned_action_strings")
            if not isinstance(actions, list) or not actions:
                skipped_turns_missing_planned_actions += 1
                continue

            normalized_actions = [str(action) for action in actions]
            counted_turns += 1
            total_actions += len(normalized_actions)
            fire_actions += sum(1 for action in normalized_actions if "fire" in action.lower())
            left_actions += sum(1 for action in normalized_actions if "left" in action.lower())
            right_actions += sum(1 for action in normalized_actions if "right" in action.lower())

    return {
        "total_turns": total_turns,
        "counted_turns": counted_turns,
        "skipped_turns_missing_planned_actions": skipped_turns_missing_planned_actions,
        "malformed_lines": malformed_lines,
        "total_actions": total_actions,
        "fire_actions": fire_actions,
        "fire_percent": pct(fire_actions, total_actions),
        "left_actions": left_actions,
        "left_percent": pct(left_actions, total_actions),
        "right_actions": right_actions,
        "right_percent": pct(right_actions, total_actions),
    }


def main() -> None:
    group_acc: dict[tuple[str, str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "num_30s_runs": 0,
            "total_turns": 0,
            "counted_turns": 0,
            "skipped_turns_missing_planned_actions": 0,
            "malformed_lines": 0,
            "total_actions": 0,
            "fire_actions": 0,
            "left_actions": 0,
            "right_actions": 0,
        }
    )
    per_run_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []

    for source, root in RUN_ROOTS:
        if not root.exists():
            skipped_rows.append({"source": source, "path": str(root), "reason": "missing_root"})
            continue

        for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
                summary_path = run_dir / "summary.json"
                turns_path = run_dir / "turns.jsonl"

                summary = load_json(summary_path)
                if summary is None:
                    skipped_rows.append(
                        {"source": source, "path": str(run_dir), "reason": "missing_or_invalid_summary"}
                    )
                    continue
                if summary.get("duration_seconds") != 30:
                    skipped_rows.append({"source": source, "path": str(run_dir), "reason": "not_30s"})
                    continue
                if not turns_path.exists():
                    skipped_rows.append({"source": source, "path": str(run_dir), "reason": "missing_turns_jsonl"})
                    continue

                key = group_key(source, summary, model_dir.name)
                run_stats = analyze_turns(turns_path)
                acc = group_acc[key]
                acc["num_30s_runs"] += 1
                for field in (
                    "total_turns",
                    "counted_turns",
                    "skipped_turns_missing_planned_actions",
                    "malformed_lines",
                    "total_actions",
                    "fire_actions",
                    "left_actions",
                    "right_actions",
                ):
                    acc[field] += run_stats[field]

                per_run_rows.append(
                    {
                        "source": source,
                        "model_name": key[1],
                        "thinking_mode": key[2],
                        "thinking_level": key[3],
                        "thinking_budget": key[4],
                        "run_dir": str(run_dir),
                        **run_stats,
                    }
                )

    summary_rows = []
    for key in sorted(group_acc):
        acc = group_acc[key]
        summary_rows.append(
            {
                "source": key[0],
                "model_name": key[1],
                "thinking_mode": key[2],
                "thinking_level": key[3],
                "thinking_budget": key[4],
                "group_label": group_label(key),
                "num_30s_runs": acc["num_30s_runs"],
                "total_turns": acc["total_turns"],
                "counted_turns": acc["counted_turns"],
                "skipped_turns_missing_planned_actions": acc[
                    "skipped_turns_missing_planned_actions"
                ],
                "malformed_lines": acc["malformed_lines"],
                "total_actions": acc["total_actions"],
                "fire_actions": acc["fire_actions"],
                "fire_percent": pct(acc["fire_actions"], acc["total_actions"]),
                "left_actions": acc["left_actions"],
                "left_percent": pct(acc["left_actions"], acc["total_actions"]),
                "right_actions": acc["right_actions"],
                "right_percent": pct(acc["right_actions"], acc["total_actions"]),
            }
        )

    write_csv(OUTPUT_DIR / "riverraid_fire_action_summary.csv", summary_rows)
    write_csv(OUTPUT_DIR / "riverraid_fire_action_per_run.csv", per_run_rows)
    write_csv(OUTPUT_DIR / "riverraid_fire_action_skipped.csv", skipped_rows)

    with (OUTPUT_DIR / "riverraid_fire_action_summary.json").open("w") as f:
        json.dump(summary_rows, f, indent=2)
        f.write("\n")
    with (OUTPUT_DIR / "riverraid_fire_action_per_run.json").open("w") as f:
        json.dump(per_run_rows, f, indent=2)
        f.write("\n")

    write_markdown(OUTPUT_DIR / "riverraid_fire_action_summary.md", summary_rows, skipped_rows)

    print(f"Wrote {len(summary_rows)} groups to {OUTPUT_DIR / 'riverraid_fire_action_summary.csv'}")
    print(f"Wrote {len(per_run_rows)} runs to {OUTPUT_DIR / 'riverraid_fire_action_per_run.csv'}")
    print(f"Wrote {len(skipped_rows)} skipped entries to {OUTPUT_DIR / 'riverraid_fire_action_skipped.csv'}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    summary_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, str]],
) -> None:
    lines = [
        "# Riverraid Fire Action Analysis",
        "",
        "Included runs have `summary.json` with `duration_seconds == 30` and a `turns.jsonl` file.",
        "Only non-empty `planned_action_strings` lists are counted. Missing or empty planned-action turns are skipped.",
        "A fire action is any planned action whose name contains `fire`, case-insensitively.",
        "",
        "| Source | Model | Thinking Mode | Thinking Level | Thinking Budget | 30s Runs | Total Actions | Fire Actions | Fire % | Left Actions | Left % | Right Actions | Right % | Skipped Turns |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        fire_percent = row["fire_percent"]
        fire_text = "NA" if fire_percent is None else f"{fire_percent:.2f}"
        left_percent = row["left_percent"]
        left_text = "NA" if left_percent is None else f"{left_percent:.2f}"
        right_percent = row["right_percent"]
        right_text = "NA" if right_percent is None else f"{right_percent:.2f}"
        lines.append(
            "| {source} | {model_name} | {thinking_mode} | {thinking_level} | "
            "{thinking_budget} | {num_30s_runs} | {total_actions} | {fire_actions} | "
            "{fire_text} | {left_actions} | {left_text} | {right_actions} | "
            "{right_text} | {skipped_turns_missing_planned_actions} |".format(
                fire_text=fire_text,
                left_text=left_text,
                right_text=right_text,
                **row,
            )
        )

    skipped_by_reason: dict[str, int] = defaultdict(int)
    for row in skipped_rows:
        skipped_by_reason[row["reason"]] += 1

    lines.extend(["", "## Skipped Entries", ""])
    if skipped_by_reason:
        for reason, count in sorted(skipped_by_reason.items()):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- None")

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
