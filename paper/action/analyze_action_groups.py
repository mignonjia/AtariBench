#!/usr/bin/env python3
"""Measure grouped action frequencies for every game and model.

This analysis counts parsed planned actions from ``planned_action_strings`` in
``turns.jsonl``. Compound actions can contribute to multiple groups: for
example, ``upleftfire`` counts as horizontal, vertical, and fire/button.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from games.registry import list_game_keys  # noqa: E402
from run_storage import sanitize_model_label  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "all_games"
DEFAULT_RUN_ROOTS = (
    REPO_ROOT.parent / "structured_history",
)
ACTION_GROUPS = ("horizontal", "vertical", "fire", "noop")
HORIZONTAL_KEYWORDS = ("left", "right")
VERTICAL_KEYWORDS = ("up", "down", "top", "bottom")
FIRE_KEYWORDS = ("fire", "shoot", "punch", "bonk", "serve", "start")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count per-game, per-model planned action rates for horizontal, "
            "vertical, fire/button, and noop groups."
        )
    )
    parser.add_argument(
        "--run-root",
        action="append",
        type=Path,
        dest="run_roots",
        help=(
            "Root containing run directories. Can be passed more than once. "
            "Defaults to ../structured_history."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=30,
        help="Only include runs with this duration when summary.json is present. Default: 30.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def norm_field(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value)
    return text if text else "none"


def pct(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator * 100.0 / denominator


def normalize_action(action: Any) -> str:
    return " ".join(str(action).strip().strip("\"'").lower().split())


def classify_action(action: Any) -> set[str]:
    normalized = normalize_action(action)
    compact = re.sub(r"[\s_-]+", "", normalized)
    groups: set[str] = set()

    if "noop" in compact:
        groups.add("noop")
    if any(keyword in compact for keyword in HORIZONTAL_KEYWORDS):
        groups.add("horizontal")
    if any(keyword in compact for keyword in VERTICAL_KEYWORDS):
        groups.add("vertical")
    if any(keyword in compact for keyword in FIRE_KEYWORDS):
        groups.add("fire")

    return groups


def add_stats(target: dict[str, Any], stats: dict[str, Any]) -> None:
    for field in (
        "total_turns",
        "counted_turns",
        "skipped_turns_missing_planned_actions",
        "malformed_lines",
        "total_actions",
        "horizontal_actions",
        "vertical_actions",
        "fire_actions",
        "noop_actions",
        "unclassified_actions",
    ):
        target[field] += stats[field]


def analyze_turns(turns_path: Path) -> dict[str, Any]:
    stats = {
        "total_turns": 0,
        "counted_turns": 0,
        "skipped_turns_missing_planned_actions": 0,
        "malformed_lines": 0,
        "total_actions": 0,
        "horizontal_actions": 0,
        "vertical_actions": 0,
        "fire_actions": 0,
        "noop_actions": 0,
        "unclassified_actions": 0,
    }

    with turns_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["total_turns"] += 1
            try:
                turn = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed_lines"] += 1
                stats["skipped_turns_missing_planned_actions"] += 1
                continue

            actions = turn.get("planned_action_strings")
            if not isinstance(actions, list) or not actions:
                stats["skipped_turns_missing_planned_actions"] += 1
                continue

            stats["counted_turns"] += 1
            stats["total_actions"] += len(actions)
            for action in actions:
                groups = classify_action(action)
                if not groups:
                    stats["unclassified_actions"] += 1
                    continue
                for group in ACTION_GROUPS:
                    if group in groups:
                        stats[f"{group}_actions"] += 1

    add_percent_fields(stats)
    return stats


def add_percent_fields(row: dict[str, Any]) -> None:
    denominator = int(row["total_actions"])
    for group in (*ACTION_GROUPS, "unclassified"):
        row[f"{group}_percent"] = pct(int(row[f"{group}_actions"]), denominator)


def load_model_summary_entries(game_dir: Path) -> dict[str, list[dict[str, Any]]]:
    payload = load_json(game_dir / "model_summary_30s.json")
    if not payload:
        return {}

    by_model_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
    models = payload.get("models")
    if not isinstance(models, dict):
        return {}

    for raw_model_name, entries in models.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model_name = str(entry.get("model_name") or raw_model_name)
            for model_key in {sanitize_model_label(model_name), model_name}:
                by_model_dir[model_key].append(entry)
    return by_model_dir


def infer_model_name_from_dir(model_dir_name: str) -> str:
    for provider in ("google", "openai", "anthropic"):
        prefix = f"{provider}-"
        if model_dir_name.startswith(prefix):
            return f"{provider}:{model_dir_name.removeprefix(prefix)}"
    return model_dir_name


def infer_thinking_level_from_run_name(run_dir_name: str) -> str | None:
    match = re.search(r"(?:^|_)(high|low)(?:_|$)", run_dir_name)
    return match.group(1) if match else None


def infer_summary(
    game: str,
    model_dir_name: str,
    run_dir_name: str,
    model_summary_entries: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    entries = model_summary_entries.get(model_dir_name, [])
    entry: dict[str, Any] | None = None
    if len(entries) == 1:
        entry = entries[0]
    else:
        cfg_match = re.search(r"cfg_\d+", run_dir_name)
        if cfg_match:
            cfg = cfg_match.group(0)
            cfg_entries = [
                candidate
                for candidate in entries
                if cfg in str(candidate.get("latest_timestamp", ""))
                or cfg in str(candidate.get("latest_run_dir", ""))
            ]
            if len(cfg_entries) == 1:
                entry = cfg_entries[0]

    if entry is None:
        return {
            "game": game,
            "model_name": infer_model_name_from_dir(model_dir_name),
            "prompt_mode": "structured_history",
            "context_cache": False,
            "thinking_mode": None,
            "thinking_level": infer_thinking_level_from_run_name(run_dir_name),
            "thinking_budget": None,
            "frames_per_action": 3,
            "duration_seconds": None,
        }

    return {
        "game": game,
        "model_name": entry.get("model_name") or model_dir_name,
        "prompt_mode": entry.get("prompt_mode"),
        "context_cache": entry.get("context_cache"),
        "thinking_mode": entry.get("thinking_mode"),
        "thinking_level": entry.get("thinking_level"),
        "thinking_budget": entry.get("thinking_budget"),
        "frames_per_action": entry.get("frames_per_action"),
        "duration_seconds": None,
    }


def group_key(
    source: str,
    game: str,
    summary: dict[str, Any],
    model_dir_name: str,
) -> tuple[str, str, str, str, str, str, str, str, str]:
    return (
        source,
        game,
        norm_field(summary.get("model_name") or model_dir_name),
        norm_field(summary.get("prompt_mode")),
        norm_field(summary.get("context_cache")),
        norm_field(summary.get("thinking_mode")),
        norm_field(summary.get("thinking_level")),
        norm_field(summary.get("thinking_budget")),
        norm_field(summary.get("frames_per_action")),
    )


def row_prefix_from_key(key: tuple[str, str, str, str, str, str, str, str, str]) -> dict[str, str]:
    return {
        "source": key[0],
        "game": key[1],
        "model_name": key[2],
        "prompt_mode": key[3],
        "context_cache": key[4],
        "thinking_mode": key[5],
        "thinking_level": key[6],
        "thinking_budget": key[7],
        "frames_per_action": key[8],
    }


def empty_accumulator() -> dict[str, Any]:
    return {
        "num_runs": 0,
        "total_turns": 0,
        "counted_turns": 0,
        "skipped_turns_missing_planned_actions": 0,
        "malformed_lines": 0,
        "total_actions": 0,
        "horizontal_actions": 0,
        "vertical_actions": 0,
        "fire_actions": 0,
        "noop_actions": 0,
        "unclassified_actions": 0,
    }


def make_source_label(root: Path, relative_parts: tuple[str, ...], game_index: int) -> str:
    base = str(root.relative_to(REPO_ROOT)) if root.is_relative_to(REPO_ROOT) else root.name
    prefix = [part for part in relative_parts[:game_index] if part]
    return "/".join([base, *prefix])


def infer_path_metadata(
    root: Path,
    run_dir: Path,
    valid_games: set[str],
) -> tuple[str, str, str] | None:
    try:
        relative_parts = run_dir.relative_to(root).parts
    except ValueError:
        return None

    for index, part in enumerate(relative_parts):
        if part not in valid_games:
            continue
        model_dir_name = relative_parts[index + 1] if index + 1 < len(relative_parts) else "unknown"
        return make_source_label(root, relative_parts, index), part, model_dir_name
    return None


def iter_run_roots(cli_roots: list[Path] | None) -> list[tuple[str, Path]]:
    roots = cli_roots if cli_roots else list(DEFAULT_RUN_ROOTS)
    return [
        (
            str(path.relative_to(REPO_ROOT))
            if path.is_relative_to(REPO_ROOT)
            else str(path),
            path,
        )
        for path in roots
    ]


def build_rows(
    run_roots: list[tuple[str, Path]],
    duration_seconds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    valid_games = set(list_game_keys())
    group_acc: dict[tuple[str, str, str, str, str, str, str, str, str], dict[str, Any]] = defaultdict(
        empty_accumulator
    )
    per_run_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []

    for root_label, root in run_roots:
        if not root.exists():
            skipped_rows.append({"source": root_label, "game": "", "path": str(root), "reason": "missing_root"})
            continue

        for turns_path in sorted(root.rglob("turns.jsonl")):
            run_dir = turns_path.parent
            path_metadata = infer_path_metadata(root, run_dir, valid_games)
            if path_metadata is None:
                skipped_rows.append(
                    {
                        "source": root_label,
                        "game": "",
                        "path": str(run_dir),
                        "reason": "unknown_game_path",
                    }
                )
                continue
            source, game, model_dir_name = path_metadata

            summary_path = run_dir / "summary.json"
            summary = load_json(summary_path)
            if summary is None:
                game_dir = next((parent for parent in run_dir.parents if parent.name == game), None)
                model_summary_entries = load_model_summary_entries(game_dir) if game_dir else {}
                summary = infer_summary(
                    game=game,
                    model_dir_name=model_dir_name,
                    run_dir_name=run_dir.name,
                    model_summary_entries=model_summary_entries,
                )
            elif summary.get("duration_seconds") != duration_seconds:
                skipped_rows.append(
                    {
                        "source": source,
                        "game": game,
                        "path": str(run_dir),
                        "reason": f"not_{duration_seconds}s",
                    }
                )
                continue

            key = group_key(source, game, summary, model_dir_name)
            run_stats = analyze_turns(turns_path)
            acc = group_acc[key]
            acc["num_runs"] += 1
            add_stats(acc, run_stats)

            per_run_rows.append(
                {
                    **row_prefix_from_key(key),
                    "run_dir": str(run_dir),
                    **run_stats,
                }
            )

        for child in sorted(path for path in root.iterdir() if path.is_dir()):
            if child.name.startswith("_"):
                continue
            if (child / "turns.jsonl").exists():
                continue
            if any(child.rglob("turns.jsonl")):
                continue
            path_metadata = infer_path_metadata(root, child, valid_games)
            if path_metadata is None:
                if child.name not in {"frames", "prompts", "responses", "visualization_frames"}:
                    skipped_rows.append(
                        {
                            "source": root_label,
                            "game": "",
                            "path": str(child),
                            "reason": "no_turns_jsonl_under_dir",
                        }
                    )
                    continue

    summary_rows = []
    for key in sorted(group_acc):
        acc = group_acc[key]
        row = {**row_prefix_from_key(key), **acc}
        add_percent_fields(row)
        summary_rows.append(row)

    return summary_rows, per_run_rows, skipped_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def fmt_percent(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}"


def write_markdown(
    path: Path,
    summary_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, str]],
) -> None:
    lines = [
        "# All-Game Action Group Analysis",
        "",
        "Included runs have `turns.jsonl`; when `summary.json` is present, `duration_seconds == 30` is required.",
        "Only non-empty `planned_action_strings` lists are counted. Missing or empty planned-action turns are skipped.",
        "Groups are overlapping: compound actions such as `upleftfire` count as horizontal, vertical, and fire/button.",
        "Fire/button also counts prompt-specific action-button names: `shoot`, `punch`, `bonk`, `serve`, and `start`.",
        "",
        "| Source | Game | Model | Prompt | Thinking | Runs | Total Actions | Horizontal % | Vertical % | Fire/Button % | Noop % | Unclassified % |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        thinking = (
            f"{row['thinking_mode']}/"
            f"{row['thinking_level']}/"
            f"{row['thinking_budget']}"
        )
        lines.append(
            "| {source} | {game} | {model_name} | {prompt_mode} | {thinking} | "
            "{num_runs} | {total_actions} | {horizontal_text} | {vertical_text} | "
            "{fire_text} | {noop_text} | {unclassified_text} |".format(
                thinking=thinking,
                horizontal_text=fmt_percent(row["horizontal_percent"]),
                vertical_text=fmt_percent(row["vertical_percent"]),
                fire_text=fmt_percent(row["fire_percent"]),
                noop_text=fmt_percent(row["noop_percent"]),
                unclassified_text=fmt_percent(row["unclassified_percent"]),
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

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, per_run_rows, skipped_rows = build_rows(
        run_roots=iter_run_roots(args.run_roots),
        duration_seconds=args.duration_seconds,
    )

    write_csv(output_dir / "action_group_summary.csv", summary_rows)
    write_csv(output_dir / "action_group_per_run.csv", per_run_rows)
    write_csv(output_dir / "action_group_skipped.csv", skipped_rows)
    write_json(output_dir / "action_group_summary.json", summary_rows)
    write_json(output_dir / "action_group_per_run.json", per_run_rows)
    write_markdown(output_dir / "action_group_summary.md", summary_rows, skipped_rows)

    print(f"Wrote {len(summary_rows)} groups to {output_dir / 'action_group_summary.csv'}")
    print(f"Wrote {len(per_run_rows)} runs to {output_dir / 'action_group_per_run.csv'}")
    print(f"Wrote {len(skipped_rows)} skipped entries to {output_dir / 'action_group_skipped.csv'}")


if __name__ == "__main__":
    main()
