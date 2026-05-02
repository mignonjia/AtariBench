#!/usr/bin/env python3
"""Plot action-group percentage heatmaps from action_group_summary.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "paper" / "action" / "all_games" / "action_group_summary.csv"
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "action" / "all_games" / "heatmaps"

ACTION_GROUPS = {
    "horizontal": ("horizontal_percent", "Horizontal Action %"),
    "vertical": ("vertical_percent", "Vertical Action %"),
    "fire": ("fire_percent", "Fire / Button Action %"),
    "noop": ("noop_percent", "Noop Action %"),
}

MODEL_ALIASES = {
    "anthropic:claude-opus-4-6": "claude-opus-4-6",
    "anthropic:claude-sonnet-4-6": "claude-sonnet-4-6",
    "google:gemini-3-flash-preview": "gemini-3-flash-preview",
    "google:gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "openai:gpt-5.4": "gpt-5.4",
    "openai:gpt-5.4-mini": "gpt-5.4-mini",
}
MODEL_DISPLAY_NAMES = {
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4 Mini",
}
MODEL_ORDER = [
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gpt-5.4",
    "gpt-5.4-mini",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
]
THINKING_LEVEL_ORDER = {"high": 0, "low": 1, "none": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-game action-group percentage heatmaps."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"CSV produced by analyze_action_groups.py. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated heatmaps. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--group",
        choices=sorted(ACTION_GROUPS),
        action="append",
        help="Action group to plot. Repeat for multiple groups. Default: all groups.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def canonical_model_name(raw_name: str) -> str:
    return MODEL_ALIASES.get(raw_name, raw_name)


def make_row_key(row: dict[str, str]) -> str:
    canonical = canonical_model_name(row["model_name"])
    thinking_level = row.get("thinking_level") or "none"
    thinking_level = thinking_level if thinking_level != "none" else "none"
    return f"{canonical}|{thinking_level}"


def make_label(row_key: str) -> str:
    canonical, thinking_level = row_key.split("|")
    base = MODEL_DISPLAY_NAMES.get(canonical, canonical)
    if thinking_level not in ("", "none"):
        return f"{base} ({thinking_level.capitalize()})"
    return base


def row_sort_key(row_key: str) -> tuple[int, int, str]:
    canonical, thinking_level = row_key.split("|")
    model_index = MODEL_ORDER.index(canonical) if canonical in MODEL_ORDER else 999
    thinking_index = THINKING_LEVEL_ORDER.get(thinking_level, 999)
    return model_index, thinking_index, row_key


def game_sort_key(game: str) -> str:
    return game.replace("_", " ").title()


def matrix_for_group(
    rows: list[dict[str, str]],
    group_field: str,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    row_keys = sorted({make_row_key(row) for row in rows}, key=row_sort_key)
    games = sorted({row["game"] for row in rows}, key=game_sort_key)
    matrix = np.full((len(row_keys), len(games)), np.nan)
    run_counts = np.full((len(row_keys), len(games)), np.nan)

    row_index = {key: i for i, key in enumerate(row_keys)}
    game_index = {game: j for j, game in enumerate(games)}
    for row in rows:
        i = row_index[make_row_key(row)]
        j = game_index[row["game"]]
        value = row.get(group_field)
        if value not in ("", "None", None):
            matrix[i, j] = float(value)
        run_counts[i, j] = float(row["num_runs"])

    return row_keys, games, matrix, run_counts


def add_row_average_column(
    ax: plt.Axes,
    values: np.ndarray,
    x_pos: float,
    header_y: float,
) -> None:
    ax.text(
        x_pos,
        header_y,
        "Avg %",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )
    for i, value in enumerate(values):
        text = "NA" if np.isnan(value) else f"{value:.1f}"
        ax.text(
            x_pos,
            i,
            text,
            ha="center",
            va="center",
            fontsize=7,
            color="#333333",
            fontweight="bold",
            clip_on=False,
        )
    ax.set_xlim(-0.5, x_pos + 0.55)


def plot_heatmap(
    rows: list[dict[str, str]],
    group_name: str,
    group_field: str,
    title: str,
    output_dir: Path,
) -> Path:
    row_keys, games, matrix, run_counts = matrix_for_group(rows, group_field)
    model_labels = [make_label(key) for key in row_keys]
    game_labels = [game.replace("_", " ").title() for game in games]
    n_models, n_games = matrix.shape

    fig_width = max(10.0, n_games * 0.48 + 1.2)
    fig_height = max(3.2, n_models * 0.42 + 1.2)
    summary_col_x = n_games + 0.55
    summary_header_y = -0.9
    avg_by_model = np.nanmean(matrix, axis=1)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        f"{group_name}_rate",
        ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    )
    grey_color = "#cccccc"

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor(grey_color)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=100, interpolation="nearest")

    for i in range(n_models):
        for j in range(n_games):
            value = matrix[i, j]
            if np.isnan(value):
                ax.add_patch(
                    mpatches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        color=grey_color,
                        zorder=2,
                    )
                )
                ax.text(
                    j,
                    i,
                    "NA",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#888888",
                    fontstyle="italic",
                    zorder=3,
                )
                continue

            text_color = "white" if value >= 62 else "black"
            ax.text(
                j,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=text_color,
                fontweight="bold",
                zorder=3,
            )
            if run_counts[i, j] < 10:
                ax.add_patch(
                    mpatches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#111111",
                        linewidth=1.2,
                        zorder=4,
                    )
                )

    ax.set_xticks(range(n_games))
    ax.set_xticklabels(game_labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_labels, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    add_row_average_column(ax, avg_by_model, summary_col_x, summary_header_y)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Planned actions (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0, 25, 50, 75, 100])

    plt.tight_layout()
    output_path = output_dir / f"action_group_heatmap_{group_name}.png"
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise RuntimeError(f"No rows found in {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = args.group or sorted(ACTION_GROUPS)

    for group_name in groups:
        group_field, title = ACTION_GROUPS[group_name]
        output_path = plot_heatmap(rows, group_name, group_field, title, args.output_dir)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
