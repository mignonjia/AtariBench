#!/usr/bin/env python3
"""Correlate action-group rates with normalized benchmark performance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTION_SUMMARY = ROOT / "paper" / "action" / "all_games" / "action_group_summary.csv"
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "action" / "all_games" / "correlation"
DEFAULT_PERFORMANCE_SUMMARIES = (
    ROOT / "runs" / "model_summary_30s_claude_structured_history.json",
    ROOT / "runs" / "model_summary_30s_gemini_structured_history.json",
    ROOT / "runs" / "model_summary_30s_gpt_structured_history.json",
)

ACTION_GROUPS = {
    "horizontal": ("horizontal_percent", "Horizontal"),
    "vertical": ("vertical_percent", "Vertical"),
    "fire": ("fire_percent", "Fire / Button"),
    "noop": ("noop_percent", "Noop"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot correlations between action-group rates and normalized scores."
    )
    parser.add_argument(
        "--action-summary",
        type=Path,
        default=DEFAULT_ACTION_SUMMARY,
        help=f"CSV from analyze_action_groups.py. Default: {DEFAULT_ACTION_SUMMARY}",
    )
    parser.add_argument(
        "--performance-summary",
        type=Path,
        action="append",
        dest="performance_summaries",
        help=(
            "Structured-history model_summary_30s JSON. Can be repeated. "
            "Defaults to the Claude/Gemini/GPT structured-history summaries."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for correlation CSV and PNGs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def norm_level(value: object) -> str:
    if value is None:
        return "none"
    text = str(value)
    return text if text and text != "None" else "none"


def load_action_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_performance_entries(paths: Iterable[Path]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        entries.extend(payload.get("entries", []))
    return entries


def performance_key(entry: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(entry.get("game")),
        str(entry.get("model_name")),
        norm_level(entry.get("thinking_level")),
    )


def action_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row["game"],
        row["model_name"],
        norm_level(row.get("thinking_level")),
    )


def build_normalized_scores(entries: list[dict[str, object]]) -> dict[tuple[str, str, str], float]:
    by_game: dict[str, list[tuple[tuple[str, str, str], float]]] = defaultdict(list)
    for entry in entries:
        value = entry.get("avg_total_reward")
        if value is None:
            continue
        by_game[str(entry.get("game"))].append((performance_key(entry), float(value)))

    normalized: dict[tuple[str, str, str], float] = {}
    for game_entries in by_game.values():
        values = np.array([value for _, value in game_entries], dtype=float)
        col_min = float(np.nanmin(values))
        shift = max(0.0, -col_min)
        shifted = values + shift
        col_max = float(np.nanmax(shifted))
        if col_max == 0:
            col_max = 1.0
        for (key, _), shifted_value in zip(game_entries, shifted):
            normalized[key] = float(np.clip(shifted_value / col_max, 0.0, 1.0))
    return normalized


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        avg_rank = (start + end) / 2.0 + 1.0
        ranks[order[start : end + 1]] = avg_rank
        start = end + 1
    return ranks


def pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    valid = ~(np.isnan(xs) | np.isnan(ys))
    xs = xs[valid]
    ys = ys[valid]
    if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return math.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def pearson_pvalue(xs: np.ndarray, ys: np.ndarray) -> float:
    valid = ~(np.isnan(xs) | np.isnan(ys))
    xs = xs[valid]
    ys = ys[valid]
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return math.nan
    return float(pearsonr(xs, ys).pvalue)


def spearman(xs: np.ndarray, ys: np.ndarray) -> float:
    valid = ~(np.isnan(xs) | np.isnan(ys))
    xs = xs[valid]
    ys = ys[valid]
    if len(xs) < 2:
        return math.nan
    return pearson(rankdata(xs), rankdata(ys))


def spearman_pvalue(xs: np.ndarray, ys: np.ndarray) -> float:
    valid = ~(np.isnan(xs) | np.isnan(ys))
    xs = xs[valid]
    ys = ys[valid]
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return math.nan
    return float(spearmanr(xs, ys).pvalue)


def significance_marker(pvalue: float) -> str:
    if math.isnan(pvalue):
        return ""
    if pvalue < 0.05:
        return "*"
    return ""


def demean(values: np.ndarray, labels: list[str]) -> np.ndarray:
    result = values.astype(float).copy()
    for label in sorted(set(labels)):
        indices = [i for i, item in enumerate(labels) if item == label]
        group_values = result[indices]
        result[indices] = group_values - np.nanmean(group_values)
    return result


def collect_records(
    action_rows: list[dict[str, str]],
    normalized_scores: dict[tuple[str, str, str], float],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in action_rows:
        key = action_key(row)
        if key not in normalized_scores:
            continue
        record: dict[str, object] = {
            "game": row["game"],
            "model_name": row["model_name"],
            "thinking_level": norm_level(row.get("thinking_level")),
            "row_label": f"{row['model_name']} ({norm_level(row.get('thinking_level'))})",
            "normalized_score": normalized_scores[key],
        }
        for group, (field, _) in ACTION_GROUPS.items():
            record[f"{group}_percent"] = float(row[field])
        records.append(record)
    return records


def compute_correlations(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    games = [str(record["game"]) for record in records]
    models = [str(record["row_label"]) for record in records]
    ys = np.array([float(record["normalized_score"]) for record in records], dtype=float)

    for group, (_, label) in ACTION_GROUPS.items():
        xs = np.array([float(record[f"{group}_percent"]) for record in records], dtype=float)
        row = {
            "action_group": group,
            "label": label,
            "n": len(records),
            "pearson_raw": pearson(xs, ys),
            "spearman_raw": spearman(xs, ys),
            "pearson_game_demeaned": pearson(demean(xs, games), demean(ys, games)),
            "pearson_model_demeaned": pearson(demean(xs, models), demean(ys, models)),
        }
        rows.append(row)
    return rows


def compute_per_game_correlations(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    games = sorted({str(record["game"]) for record in records}, key=lambda game: game.replace("_", " ").title())

    for game in games:
        game_records = [record for record in records if record["game"] == game]
        ys = np.array([float(record["normalized_score"]) for record in game_records], dtype=float)
        for group, (_, label) in ACTION_GROUPS.items():
            xs = np.array([float(record[f"{group}_percent"]) for record in game_records], dtype=float)
            rows.append(
                {
                    "game": game,
                    "action_group": group,
                    "label": label,
                    "n": len(game_records),
                    "pearson": pearson(xs, ys),
                    "pearson_p": pearson_pvalue(xs, ys),
                    "spearman": spearman(xs, ys),
                    "spearman_p": spearman_pvalue(xs, ys),
                }
            )
    return rows


def write_correlation_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_regression_line(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray) -> None:
    if len(xs) < 2 or np.std(xs) == 0:
        return
    slope, intercept = np.polyfit(xs, ys, 1)
    x_line = np.linspace(float(np.min(xs)), float(np.max(xs)), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#111111", lw=1.4, alpha=0.8)


def plot_combined(records: list[dict[str, object]], corr_rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    axes_flat = axes.ravel()
    color_values = np.array([float(record["normalized_score"]) for record in records], dtype=float)

    for index, (ax, corr_row) in enumerate(zip(axes_flat, corr_rows)):
        group = str(corr_row["action_group"])
        label = str(corr_row["label"])
        xs = np.array([float(record[f"{group}_percent"]) for record in records], dtype=float)
        ys = color_values
        scatter = ax.scatter(
            xs,
            ys,
            c=ys,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            s=34,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
        )
        add_regression_line(ax, xs, ys)
        ax.set_title(
            (
                f"{label}\n"
                f"r={corr_row['pearson_raw']:+.2f}, "
                f"rho={corr_row['spearman_raw']:+.2f}, "
                f"game-demeaned r={corr_row['pearson_game_demeaned']:+.2f}"
            ),
            fontsize=10,
            fontweight="bold",
        )
        if index >= 2:
            ax.set_xlabel("Action group rate (%)", fontsize=9)
        ax.set_xlim(-3, 103)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes_flat[0].set_ylabel("Normalized score", fontsize=9)
    axes_flat[2].set_ylabel("Normalized score", fontsize=9)
    cbar = fig.colorbar(scatter, ax=axes_flat.tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Normalized score", fontsize=9)
    fig.suptitle(
        "Action Group Rate vs Normalized Benchmark Score",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.88, hspace=0.46, wspace=0.18, right=0.88)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_individual(records: list[dict[str, object]], corr_rows: list[dict[str, object]], output_dir: Path) -> None:
    for corr_row in corr_rows:
        group = str(corr_row["action_group"])
        label = str(corr_row["label"])
        xs = np.array([float(record[f"{group}_percent"]) for record in records], dtype=float)
        ys = np.array([float(record["normalized_score"]) for record in records], dtype=float)

        fig, ax = plt.subplots(figsize=(6, 4.3))
        sc = ax.scatter(
            xs,
            ys,
            c=ys,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            s=38,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
        )
        add_regression_line(ax, xs, ys)
        ax.set_title(
            (
                f"{label} Action Rate vs Normalized Score\n"
                f"Pearson r={corr_row['pearson_raw']:+.2f}, "
                f"Spearman rho={corr_row['spearman_raw']:+.2f}, "
                f"game-demeaned r={corr_row['pearson_game_demeaned']:+.2f}"
            ),
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Action group rate (%)", fontsize=9)
        ax.set_ylabel("Normalized score", fontsize=9)
        ax.set_xlim(-3, 103)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label("Normalized score", fontsize=8)
        fig.savefig(output_dir / f"action_group_corr_{group}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def plot_per_game_heatmap(
    per_game_rows: list[dict[str, object]],
    metric: str,
    pvalue_metric: str,
    output_path: Path,
) -> None:
    groups = list(ACTION_GROUPS)
    games = sorted(
        {str(row["game"]) for row in per_game_rows},
        key=lambda game: game.replace("_", " ").title(),
    )
    matrix = np.full((len(groups), len(games)), np.nan)
    lookup = {
        (str(row["action_group"]), str(row["game"])): float(row[metric])
        for row in per_game_rows
    }
    pvalue_lookup = {
        (str(row["action_group"]), str(row["game"])): float(row[pvalue_metric])
        for row in per_game_rows
    }
    for i, group in enumerate(groups):
        for j, game in enumerate(games):
            matrix[i, j] = lookup.get((group, game), math.nan)

    avg_col = np.nanmean(matrix, axis=1)
    fig_width = max(11, len(games) * 0.5 + 1.3)
    fig, ax = plt.subplots(figsize=(fig_width, 3.2))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    grey_color = "#cccccc"

    for i in range(len(groups)):
        for j in range(len(games)):
            value = matrix[i, j]
            if np.isnan(value):
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=grey_color, zorder=2)
                )
                ax.text(j, i, "NA", ha="center", va="center", fontsize=7, color="#777777", zorder=3)
                continue
            text_color = "white" if abs(value) > 0.55 else "black"
            pvalue = pvalue_lookup.get((groups[i], games[j]), math.nan)
            marker = significance_marker(pvalue)
            ax.text(
                j,
                i,
                f"{value:+.2f}{marker}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
                fontweight="bold",
                zorder=3,
            )
            if marker:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#111111",
                        linewidth=1.5,
                        zorder=4,
                    )
                )

    summary_col_x = len(games) + 0.55
    ax.text(
        summary_col_x,
        -0.68,
        "Avg r",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )
    for i, value in enumerate(avg_col):
        ax.text(
            summary_col_x,
            i,
            f"{value:+.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
            fontweight="bold",
            clip_on=False,
        )
    ax.set_xlim(-0.5, summary_col_x + 0.55)

    metric_label = "Pearson r" if metric == "pearson" else "Spearman rho"
    ax.set_xticks(range(len(games)))
    ax.set_xticklabels([game.replace("_", " ").title() for game in games], rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([ACTION_GROUPS[group][1] for group in groups], fontsize=8)
    ax.set_title(
        (
            f"Per-Game Correlation: Action Rate vs Normalized Score ({metric_label})\n"
            "* p<0.05"
        ),
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label(metric_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    performance_paths = args.performance_summaries or list(DEFAULT_PERFORMANCE_SUMMARIES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    action_rows = load_action_rows(args.action_summary)
    performance_entries = load_performance_entries(performance_paths)
    normalized_scores = build_normalized_scores(performance_entries)
    records = collect_records(action_rows, normalized_scores)
    if not records:
        raise RuntimeError("No matched action/performance records found.")

    per_game_rows = compute_per_game_correlations(records)
    write_correlation_csv(args.output_dir / "action_group_per_game_correlation.csv", per_game_rows)
    plot_per_game_heatmap(
        per_game_rows,
        "pearson",
        "pearson_p",
        args.output_dir / "action_group_per_game_pearson.png",
    )
    plot_per_game_heatmap(
        per_game_rows,
        "spearman",
        "spearman_p",
        args.output_dir / "action_group_per_game_spearman.png",
    )

    print(f"Matched {len(records)} action/performance rows.")
    print(f"Saved {args.output_dir / 'action_group_per_game_pearson.png'}")
    print(f"Saved {args.output_dir / 'action_group_per_game_spearman.png'}")
    significant = [
        row
        for row in per_game_rows
        if not math.isnan(float(row["pearson_p"])) and float(row["pearson_p"]) < 0.05
    ]
    print("Significant per-game Pearson cells (p < 0.05):")
    if significant:
        for row in sorted(significant, key=lambda item: float(item["pearson_p"])):
            print(
                "  {game},{action_group},r={pearson:+.3f},p={pearson_p:.4f}".format(**row)
            )
    else:
        print("  none")


if __name__ == "__main__":
    main()
