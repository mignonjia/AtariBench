import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ── Load data ─────────────────────────────────────────────────────────────────

with open("runs/model_summary_30s.json") as f:
    data = json.load(f)

RANK_MODELS = [
    "gemini-2.5-flash",
    "gpt-5.4-mini",
    "deepseek-ai/deepseek-v3.1",
    "zai-org/glm-5.1",
]

ALL_MODELS = RANK_MODELS

MODEL_LABELS = [
    "Gemini 2.5 Flash",
    "GPT-5.4 Mini",
    "DeepSeek V3.1",
    "GLM 5.1",
]

# Collect avg_total_reward per (model, game)
reward = {}
for e in data["entries"]:
    if e["model_name"] in RANK_MODELS:
        reward[(e["model_name"], e["game"])] = e["avg_total_reward"]

# Games with complete data for the 3 ranked models
all_games = sorted(
    set(g for (m, g) in reward.keys()),
    key=lambda g: g.replace("_", " ").title(),
)
games = [g for g in all_games if all((m, g) in reward for m in RANK_MODELS)]

n_models = len(ALL_MODELS)
n_games = len(games)
game_labels = [g.replace("_", " ").title() for g in games]

# ── Build reward matrix (NaN for missing gemini-3 games) ─────────────────────
matrix = np.full((n_models, n_games), np.nan)
for i, m in enumerate(ALL_MODELS):
    for j, g in enumerate(games):
        if (m, g) in reward:
            matrix[i, j] = reward[(m, g)]

# ── Rank plot ─────────────────────────────────────────────────────────────────
# Rank only among the 3 complete models (rows 0-2), always 1-3.
# Gemini-3-flash (row 3) is not ranked; its cell shows NA or its value color-coded.

RANK_N = len(RANK_MODELS)  # 4

# Rank all models that have data per game (higher score = rank 1).
# Ties share the same rank; next rank skips (standard competition style).
# Games with gemini-3-flash data: ranks 1-5; without: ranks 1-4.
rank_matrix = np.full((n_models, n_games), np.nan)
for j in range(n_games):
    rows_with_data = [i for i in range(n_models) if not np.isnan(matrix[i, j])]
    col = matrix[rows_with_data, j]
    sorted_order = np.argsort(-col)   # descending
    rank = 1
    for k, ki in enumerate(sorted_order):
        idx = rows_with_data[ki]
        if k > 0 and col[ki] == col[sorted_order[k - 1]]:
            rank_matrix[idx, j] = rank_matrix[rows_with_data[sorted_order[k - 1]], j]
        else:
            rank_matrix[idx, j] = rank
        rank = k + 2

# Color: green=1, red=3 (rank among 3 reference models)
cmap_r = mcolors.LinearSegmentedColormap.from_list(
    "gr", ["#1a9850", "#fee08b", "#d73027"]
)
grey_color = "#cccccc"

fig_width = max(8, n_games * 0.42)
fig_height = max(2.2, n_models * 0.55 + 1.4)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Draw background grey for all cells, then overwrite with color where data exists
ax.set_facecolor(grey_color)

# Draw colored cells using imshow for the ranked rows
rank_display = np.copy(rank_matrix)

im = ax.imshow(rank_display, aspect="auto", cmap=cmap_r, vmin=1, vmax=n_models,
               interpolation="nearest")

# Grey out NaN cells (gemini-3-flash missing games)
for i in range(n_models):
    for j in range(n_games):
        if np.isnan(rank_matrix[i, j]):
            ax.add_patch(mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                color=grey_color, zorder=2
            ))

# Annotate cells
for i in range(n_models):
    for j in range(n_games):
        rv = rank_matrix[i, j]
        if np.isnan(rv):
            ax.text(j, i, "NA", ha="center", va="center",
                    fontsize=8, color="#888888", fontstyle="italic", zorder=3)
        else:
            norm_val = (rv - 1) / max(n_models - 1, 1)
            text_color = "black" if 0.25 < norm_val < 0.75 else "white"
            label = str(int(rv))
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=7, color=text_color, fontweight="bold", zorder=3)

ax.set_xticks(range(n_games))
ax.set_xticklabels(game_labels, rotation=40, ha="right", fontsize=7)
ax.set_yticks(range(n_models))
ax.set_yticklabels(MODEL_LABELS, fontsize=8)
ax.set_title("Individual Benchmark Scores by Model", fontsize=10, fontweight="bold", pad=8)

cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
cbar.set_label("Rank (1 = best)", fontsize=7)
cbar.ax.tick_params(labelsize=7)
cbar.set_ticks(list(range(1, n_models + 1)))
cbar.set_ticklabels([str(r) for r in range(1, n_models + 1)])

plt.tight_layout()
out_path = "paper/plot_heatmap_rank.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close()

# ── Raw value plot ─────────────────────────────────────────────────────────────
cmap_raw = mcolors.LinearSegmentedColormap.from_list(
    "rg", ["#d73027", "#fee08b", "#1a9850"]
)

# Normalize per game among the 3 ranked models; gemini-3 colored relative to them
col_min = matrix[:RANK_N, :].min(axis=0)
shift = np.where(col_min < 0, -col_min, 0)
shifted_ref = matrix[:RANK_N, :] + shift
col_max = shifted_ref.max(axis=0)
col_max[col_max == 0] = 1

norm_matrix = np.full((n_models, n_games), np.nan)
for i in range(n_models):
    valid = ~np.isnan(matrix[i, :])
    norm_matrix[i, valid] = (matrix[i, valid] + shift[valid]) / col_max[valid]
norm_matrix = np.clip(norm_matrix, 0, 1)

fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
ax2.set_facecolor(grey_color)

im2 = ax2.imshow(norm_matrix, aspect="auto", cmap=cmap_raw, vmin=0, vmax=1,
                 interpolation="nearest")

for i in range(n_models):
    for j in range(n_games):
        if np.isnan(norm_matrix[i, j]):
            ax2.add_patch(mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, color=grey_color, zorder=2
            ))
            ax2.text(j, i, "NA", ha="center", va="center",
                     fontsize=8, color="#888888", fontstyle="italic", zorder=3)
        else:
            nv = norm_matrix[i, j]
            text_color = "black" if 0.35 < nv < 0.85 else "white"
            ax2.text(j, i, f"{matrix[i, j]:.1f}",
                     ha="center", va="center", fontsize=6.5,
                     color=text_color, fontweight="bold", zorder=3)

ax2.set_xticks(range(n_games))
ax2.set_xticklabels(game_labels, rotation=40, ha="right", fontsize=7)
ax2.set_yticks(range(n_models))
ax2.set_yticklabels(MODEL_LABELS, fontsize=8)
ax2.set_title("Individual Benchmark Scores by Model (Raw)", fontsize=10, fontweight="bold", pad=8)

cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.02, pad=0.01)
cbar2.set_label("Normalized Score (per game)", fontsize=7)
cbar2.ax.tick_params(labelsize=7)
cbar2.set_ticks([0, 0.5, 1])
cbar2.set_ticklabels(["Low", "Mid", "High"])

plt.tight_layout()
out_path2 = "paper/plot_heatmap_raw.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path2}")
plt.close()

# ── Normalized value plot ──────────────────────────────────────────────────────
# Same as raw but cells show the normalized score (0–1) instead of raw values.

fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height))
ax3.set_facecolor(grey_color)

im3 = ax3.imshow(norm_matrix, aspect="auto", cmap=cmap_raw, vmin=0, vmax=1,
                 interpolation="nearest")

for i in range(n_models):
    for j in range(n_games):
        if np.isnan(norm_matrix[i, j]):
            ax3.add_patch(mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, color=grey_color, zorder=2
            ))
            ax3.text(j, i, "NA", ha="center", va="center",
                     fontsize=8, color="#888888", fontstyle="italic", zorder=3)
        else:
            nv = norm_matrix[i, j]
            text_color = "black" if 0.35 < nv < 0.85 else "white"
            ax3.text(j, i, f"{nv:.2f}",
                     ha="center", va="center", fontsize=6.5,
                     color=text_color, fontweight="bold", zorder=3)

ax3.set_xticks(range(n_games))
ax3.set_xticklabels(game_labels, rotation=40, ha="right", fontsize=7)
ax3.set_yticks(range(n_models))
ax3.set_yticklabels(MODEL_LABELS, fontsize=8)
ax3.set_title("Individual Benchmark Scores by Model (Normalized)", fontsize=10, fontweight="bold", pad=8)

cbar3 = fig3.colorbar(im3, ax=ax3, fraction=0.02, pad=0.01)
cbar3.set_label("Normalized Score (per game)", fontsize=7)
cbar3.ax.tick_params(labelsize=7)
cbar3.set_ticks([0, 0.5, 1])
cbar3.set_ticklabels(["Low", "Mid", "High"])

plt.tight_layout()
out_path3 = "paper/plot_heatmap_norm.png"
plt.savefig(out_path3, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path3}")
plt.close()
