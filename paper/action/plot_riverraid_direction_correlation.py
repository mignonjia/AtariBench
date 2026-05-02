#!/usr/bin/env python3
"""Plot Riverraid left/right action rates against normalized performance."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT.parent / "structured_history"
OUTPUT_DIR = ROOT / "paper" / "action" / "all_games" / "correlation"
PERFORMANCE_SUMMARIES = (
    ROOT / "runs" / "model_summary_30s_claude_structured_history.json",
    ROOT / "runs" / "model_summary_30s_gemini_structured_history.json",
    ROOT / "runs" / "model_summary_30s_gpt_structured_history.json",
)


def norm_level(value: object) -> str:
    if value is None:
        return "none"
    text = str(value)
    return text if text and text != "None" else "none"


def infer_model_name(model_dir_name: str) -> str:
    for provider in ("google", "openai", "anthropic"):
        prefix = f"{provider}-"
        if model_dir_name.startswith(prefix):
            return f"{provider}:{model_dir_name.removeprefix(prefix)}"
    return model_dir_name


def infer_level(run_name: str) -> str:
    match = re.search(r"(?:^|_)(high|low)(?:_|$)", run_name)
    return match.group(1) if match else "none"


def load_json(path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def iter_riverraid_runs() -> list[tuple[Path, str, str]]:
    runs: list[tuple[Path, str, str]] = []
    for turns_path in sorted(RUN_ROOT.glob("*/riverraid/*/*/turns.jsonl")):
        model_dir = turns_path.parents[1].name
        summary = load_json(turns_path.parent / "summary.json")
        if summary:
            model_name = str(summary.get("model_name") or infer_model_name(model_dir))
            thinking_level = norm_level(summary.get("thinking_level"))
        else:
            model_name = infer_model_name(model_dir)
            thinking_level = infer_level(turns_path.parent.name)
        runs.append((turns_path, model_name, thinking_level))
    return runs


def analyze_turns(turns_path: Path) -> dict[str, int]:
    stats = {"total_actions": 0, "left_actions": 0, "right_actions": 0}
    with turns_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            turn = json.loads(line)
            actions = turn.get("planned_action_strings")
            if not isinstance(actions, list):
                continue
            for action in actions:
                compact = re.sub(r"[\s_-]+", "", str(action).lower())
                stats["total_actions"] += 1
                if "left" in compact:
                    stats["left_actions"] += 1
                if "right" in compact:
                    stats["right_actions"] += 1
    return stats


def load_riverraid_scores() -> dict[tuple[str, str], float]:
    raw_scores: dict[tuple[str, str], float] = {}
    for path in PERFORMANCE_SUMMARIES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload.get("entries", []):
            if entry.get("game") != "riverraid":
                continue
            key = (str(entry.get("model_name")), norm_level(entry.get("thinking_level")))
            raw_scores[key] = float(entry["avg_total_reward"])

    values = np.array(list(raw_scores.values()), dtype=float)
    shift = max(0.0, -float(values.min()))
    denom = max(float((values + shift).max()), 1.0)
    return {key: float((value + shift) / denom) for key, value in raw_scores.items()}


def pearson(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan"), float("nan")
    result = pearsonr(xs, ys)
    return float(result.statistic), float(result.pvalue)


def spearman(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan"), float("nan")
    result = spearmanr(xs, ys)
    return float(result.statistic), float(result.pvalue)


def build_rows() -> list[dict[str, object]]:
    acc: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"num_runs": 0, "total_actions": 0, "left_actions": 0, "right_actions": 0}
    )
    for turns_path, model_name, thinking_level in iter_riverraid_runs():
        key = (model_name, thinking_level)
        stats = analyze_turns(turns_path)
        acc[key]["num_runs"] += 1
        for field, value in stats.items():
            acc[key][field] += value

    scores = load_riverraid_scores()
    rows: list[dict[str, object]] = []
    for (model_name, thinking_level), stats in sorted(acc.items()):
        total = stats["total_actions"]
        if total == 0 or (model_name, thinking_level) not in scores:
            continue
        left_percent = stats["left_actions"] * 100.0 / total
        right_percent = stats["right_actions"] * 100.0 / total
        rows.append(
            {
                "model_name": model_name,
                "thinking_level": thinking_level,
                "num_runs": stats["num_runs"],
                "total_actions": total,
                "left_percent": left_percent,
                "right_percent": right_percent,
                "right_minus_left_percent": right_percent - left_percent,
                "normalized_score": scores[(model_name, thinking_level)],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_panel(ax: plt.Axes, rows: list[dict[str, object]], field: str, title: str) -> None:
    xs = np.array([float(row[field]) for row in rows], dtype=float)
    ys = np.array([float(row["normalized_score"]) for row in rows], dtype=float)
    r, p = pearson(xs, ys)
    rho, rho_p = spearman(xs, ys)
    ax.scatter(xs, ys, c=ys, cmap="RdYlGn", vmin=0, vmax=1, s=70, edgecolors="white", linewidths=0.5)
    if np.std(xs) > 0:
        slope, intercept = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="#111111", lw=1.4)
    for row in rows:
        label = f"{str(row['model_name']).split(':')[-1].replace('-', ' ')} {row['thinking_level']}"
        ax.annotate(label, (float(row[field]), float(row["normalized_score"])), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
    ax.set_title(f"{title}\nr={r:+.2f}, p={p:.3f}; rho={rho:+.2f}, p={rho_p:.3f}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Action rate (%)", fontsize=8)
    ax.set_ylabel("Riverraid normalized score", fontsize=8)
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def plot(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    add_panel(axes[0], rows, "left_percent", "Left Action Rate")
    add_panel(axes[1], rows, "right_percent", "Right Action Rate")
    add_panel(axes[2], rows, "right_minus_left_percent", "Right - Left Bias")
    fig.suptitle("Riverraid Directional Action Rate vs Performance", fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    if not rows:
        raise RuntimeError("No matched Riverraid direction rows found.")
    write_csv(OUTPUT_DIR / "riverraid_direction_correlation.csv", rows)
    plot(rows, OUTPUT_DIR / "riverraid_direction_correlation.png")
    print(f"Wrote {len(rows)} rows to {OUTPUT_DIR / 'riverraid_direction_correlation.csv'}")
    print(f"Saved {OUTPUT_DIR / 'riverraid_direction_correlation.png'}")


if __name__ == "__main__":
    main()
