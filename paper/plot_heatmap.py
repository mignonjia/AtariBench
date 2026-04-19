import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_PATHS = [
    ROOT / "runs" / "model_summary_30s.json",
    ROOT / "runs" / "results_gemini_jiaxi_30s.json",
]

MODEL_ALIASES = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "deepseek-ai/deepseek-v3.1": "deepseek-ai/deepseek-v3.1",
    "zai-org/glm-5.1": "zai-org/glm-5.1",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "google:gemini-3-flash-preview": "gemini-3-flash-preview",
    "google:gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
}

THINKING_SPLIT_MODELS = {
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
}

MODEL_SPECS = [
    ("gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("gpt-5.4-mini", "GPT-5.4 Mini"),
    ("deepseek-ai/deepseek-v3.1", "DeepSeek V3.1"),
    ("zai-org/glm-5.1", "GLM 5.1"),
    ("gemini-3-flash-preview:low", "Gemini 3 Flash (Low)"),
    ("gemini-3-flash-preview:high", "Gemini 3 Flash (High)"),
    ("gemini-3.1-pro-preview:low", "Gemini 3.1 Pro (Low)"),
    ("gemini-3.1-pro-preview:high", "Gemini 3.1 Pro (High)"),
]
REFERENCE_MODELS = [model for model, _ in MODEL_SPECS[:4]]


def load_entries(paths):
    entries = []
    for path in paths:
        with path.open() as handle:
            payload = json.load(handle)
        entries.extend(payload.get("entries", []))
    return entries


def canonical_model_name(model_name):
    return MODEL_ALIASES.get(model_name)


def canonical_row_key(entry):
    model_name = canonical_model_name(entry.get("model_name", ""))
    if model_name is None:
        return None
    if model_name not in THINKING_SPLIT_MODELS:
        return model_name

    thinking_level = entry.get("thinking_level")
    if thinking_level not in {"low", "high"}:
        return None
    return f"{model_name}:{thinking_level}"


def collect_best_rewards(entries):
    rewards = {}
    for entry in entries:
        row_key = canonical_row_key(entry)
        game = entry.get("game")
        value = entry.get("avg_total_reward")
        if row_key is None or game is None or value is None:
            continue
        key = (row_key, game)
        if key not in rewards or value > rewards[key]:
            rewards[key] = value
    return rewards


def add_row_average_column(ax, values, label, fmt, x_pos, header_y):
    ax.text(
        x_pos,
        header_y,
        label,
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )
    for i, value in enumerate(values):
        text = "NA" if np.isnan(value) else format(value, fmt)
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


entries = load_entries(DATA_PATHS)
reward = collect_best_rewards(entries)

all_games = sorted(
    {game for _, game in reward.keys()},
    key=lambda game: game.replace("_", " ").title(),
)
games = [game for game in all_games if all((model, game) in reward for model in REFERENCE_MODELS)]

all_models = [model for model, _ in MODEL_SPECS]
model_labels = [label for _, label in MODEL_SPECS]

n_models = len(all_models)
n_games = len(games)
game_labels = [game.replace("_", " ").title() for game in games]
summary_col_x = n_games + 0.55
summary_header_y = -0.9

matrix = np.full((n_models, n_games), np.nan)
for i, model in enumerate(all_models):
    for j, game in enumerate(games):
        if (model, game) in reward:
            matrix[i, j] = reward[(model, game)]

# Rank all available model scores per game using standard competition ranking.
rank_matrix = np.full((n_models, n_games), np.nan)
for j in range(n_games):
    rows_with_data = [i for i in range(n_models) if not np.isnan(matrix[i, j])]
    col = matrix[rows_with_data, j]
    sorted_order = np.argsort(-col)
    rank = 1
    for k, ki in enumerate(sorted_order):
        idx = rows_with_data[ki]
        if k > 0 and col[ki] == col[sorted_order[k - 1]]:
            prev_idx = rows_with_data[sorted_order[k - 1]]
            rank_matrix[idx, j] = rank_matrix[prev_idx, j]
        else:
            rank_matrix[idx, j] = rank
        rank = k + 2

avg_rank_by_model = np.nanmean(rank_matrix, axis=1)

cmap_r = mcolors.LinearSegmentedColormap.from_list(
    "gr", ["#1a9850", "#fee08b", "#d73027"]
)
grey_color = "#cccccc"

fig_width = max(8, n_games * 0.42)
fig_height = max(2.6, n_models * 0.55 + 1.4)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.set_facecolor(grey_color)

im = ax.imshow(
    rank_matrix,
    aspect="auto",
    cmap=cmap_r,
    vmin=1,
    vmax=n_models,
    interpolation="nearest",
)

for i in range(n_models):
    for j in range(n_games):
        if np.isnan(rank_matrix[i, j]):
            ax.add_patch(
                mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, color=grey_color, zorder=2
                )
            )

for i in range(n_models):
    for j in range(n_games):
        rv = rank_matrix[i, j]
        if np.isnan(rv):
            ax.text(
                j,
                i,
                "NA",
                ha="center",
                va="center",
                fontsize=8,
                color="#888888",
                fontstyle="italic",
                zorder=3,
            )
        else:
            norm_val = (rv - 1) / max(n_models - 1, 1)
            text_color = "black" if 0.25 < norm_val < 0.75 else "white"
            ax.text(
                j,
                i,
                str(int(rv)),
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
                fontweight="bold",
                zorder=3,
            )

ax.set_xticks(range(n_games))
ax.set_xticklabels(game_labels, rotation=40, ha="right", fontsize=7)
ax.set_yticks(range(n_models))
ax.set_yticklabels(model_labels, fontsize=8)
ax.set_title("Individual Benchmark Scores by Model", fontsize=10, fontweight="bold", pad=8)
add_row_average_column(
    ax,
    avg_rank_by_model,
    "Avg Rank",
    ".2f",
    summary_col_x,
    summary_header_y,
)

cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
cbar.set_label("Rank (1 = best)", fontsize=7)
cbar.ax.tick_params(labelsize=7)
cbar.set_ticks(list(range(1, n_models + 1)))
cbar.set_ticklabels([str(r) for r in range(1, n_models + 1)])

plt.tight_layout()
out_path = ROOT / "paper" / "plot_heatmap_rank.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close()

cmap_raw = mcolors.LinearSegmentedColormap.from_list(
    "rg", ["#d73027", "#fee08b", "#1a9850"]
)

col_min = np.nanmin(matrix, axis=0)
shift = np.where(col_min < 0, -col_min, 0)
shifted_matrix = matrix + shift
col_max = np.nanmax(shifted_matrix, axis=0)
col_max[col_max == 0] = 1

norm_matrix = np.full((n_models, n_games), np.nan)
for i in range(n_models):
    valid = ~np.isnan(matrix[i, :])
    norm_matrix[i, valid] = (matrix[i, valid] + shift[valid]) / col_max[valid]
norm_matrix = np.clip(norm_matrix, 0, 1)
avg_norm_by_model = np.nanmean(norm_matrix, axis=1)

fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
ax2.set_facecolor(grey_color)

im2 = ax2.imshow(
    norm_matrix,
    aspect="auto",
    cmap=cmap_raw,
    vmin=0,
    vmax=1,
    interpolation="nearest",
)

for i in range(n_models):
    for j in range(n_games):
        if np.isnan(norm_matrix[i, j]):
            ax2.add_patch(
                mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, color=grey_color, zorder=2
                )
            )
            ax2.text(
                j,
                i,
                "NA",
                ha="center",
                va="center",
                fontsize=8,
                color="#888888",
                fontstyle="italic",
                zorder=3,
            )
        else:
            nv = norm_matrix[i, j]
            text_color = "black" if 0.35 < nv < 0.85 else "white"
            ax2.text(
                j,
                i,
                f"{matrix[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=text_color,
                fontweight="bold",
                zorder=3,
            )

ax2.set_xticks(range(n_games))
ax2.set_xticklabels(game_labels, rotation=40, ha="right", fontsize=7)
ax2.set_yticks(range(n_models))
ax2.set_yticklabels(model_labels, fontsize=8)
ax2.set_title("Individual Benchmark Scores by Model (Raw)", fontsize=10, fontweight="bold", pad=8)

cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.02, pad=0.01)
cbar2.set_label("Normalized Score (per game)", fontsize=7)
cbar2.ax.tick_params(labelsize=7)
cbar2.set_ticks([0, 0.5, 1])
cbar2.set_ticklabels(["Low", "Mid", "High"])

plt.tight_layout()
out_path2 = ROOT / "paper" / "plot_heatmap_raw.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path2}")
plt.close()

fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height))
ax3.set_facecolor(grey_color)

im3 = ax3.imshow(
    norm_matrix,
    aspect="auto",
    cmap=cmap_raw,
    vmin=0,
    vmax=1,
    interpolation="nearest",
)

for i in range(n_models):
    for j in range(n_games):
        if np.isnan(norm_matrix[i, j]):
            ax3.add_patch(
                mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, color=grey_color, zorder=2
                )
            )
            ax3.text(
                j,
                i,
                "NA",
                ha="center",
                va="center",
                fontsize=8,
                color="#888888",
                fontstyle="italic",
                zorder=3,
            )
        else:
            nv = norm_matrix[i, j]
            text_color = "black" if 0.35 < nv < 0.85 else "white"
            ax3.text(
                j,
                i,
                f"{nv:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=text_color,
                fontweight="bold",
                zorder=3,
            )

ax3.set_xticks(range(n_games))
ax3.set_xticklabels(game_labels, rotation=40, ha="right", fontsize=7)
ax3.set_yticks(range(n_models))
ax3.set_yticklabels(model_labels, fontsize=8)
ax3.set_title("Individual Benchmark Scores by Model (Normalized)", fontsize=10, fontweight="bold", pad=8)
add_row_average_column(
    ax3,
    avg_norm_by_model,
    "Avg Norm",
    ".2f",
    summary_col_x,
    summary_header_y,
)

cbar3 = fig3.colorbar(im3, ax=ax3, fraction=0.02, pad=0.01)
cbar3.set_label("Normalized Score (per game)", fontsize=7)
cbar3.ax.tick_params(labelsize=7)
cbar3.set_ticks([0, 0.5, 1])
cbar3.set_ticklabels(["Low", "Mid", "High"])

plt.tight_layout()
out_path3 = ROOT / "paper" / "plot_heatmap_norm.png"
plt.savefig(out_path3, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path3}")
plt.close()
