import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load data
with open("runs/model_summary_30s.json") as f:
    data = json.load(f)

entries = data["entries"]

TARGET_MODELS = [
    "gemini-2.5-flash",
    "gpt-5.4-mini",
    "deepseek-ai/deepseek-v3.1",
]

MODEL_LABELS = [
    "Gemini 2.5 Flash",
    "GPT-5.4 Mini",
    "DeepSeek V3.1",
]

# Collect avg_total_reward per (model, game)
reward = {}
for e in entries:
    model = e["model_name"]
    if model in TARGET_MODELS:
        game = e["game"]
        reward[(model, game)] = e["avg_total_reward"]

# All games that appear for all 3 models
all_games = sorted(
    set(g for (m, g) in reward.keys()),
    key=lambda g: g.replace("_", " ").title(),
)
# Keep only games present for all 3 target models
games = [g for g in all_games if all((m, g) in reward for m in TARGET_MODELS)]

# Build matrix: rows = models, cols = games
matrix = np.array(
    [[reward[(m, g)] for g in games] for m in TARGET_MODELS], dtype=float
)

# Rank per game (column): higher score = rank 1 (best).
# Ties get the same rank; the next rank skips (standard competition / "1224" style).
rank_matrix = np.zeros_like(matrix, dtype=float)
for j in range(matrix.shape[1]):
    col = matrix[:, j]
    # argsort descending to get order; assign ranks with tie-skip
    sorted_idx = np.argsort(-col)   # indices from highest to lowest
    rank = 1
    for k, idx in enumerate(sorted_idx):
        if k > 0 and col[idx] == col[sorted_idx[k - 1]]:
            rank_matrix[idx, j] = rank_matrix[sorted_idx[k - 1], j]  # same rank as previous
        else:
            rank_matrix[idx, j] = rank
        rank = k + 2  # next rank = position + 1 (1-based)

# Pretty game labels
game_labels = [g.replace("_", " ").title() for g in games]

n_models = len(TARGET_MODELS)
n_games = len(games)

fig_width = max(14, n_games * 0.72)
fig_height = max(3, n_models * 0.9 + 1.6)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Color map: green (rank 1/best) -> yellow -> red (worst rank)
cmap_r = mcolors.LinearSegmentedColormap.from_list(
    "gr", ["#1a9850", "#fee08b", "#d73027"]
)

im = ax.imshow(rank_matrix, aspect="auto", cmap=cmap_r, vmin=1, vmax=n_models)

# Annotate cells with rank
for i in range(n_models):
    for j in range(n_games):
        rv = rank_matrix[i, j]
        norm_val = (rv - 1) / max(n_models - 1, 1)   # 0=rank1(green), 1=last(red)
        text_color = "black" if 0.25 < norm_val < 0.75 else "white"
        ax.text(
            j, i,
            f"{int(rv)}",
            ha="center", va="center",
            fontsize=9,
            color=text_color,
            fontweight="bold",
        )

# Axes ticks
ax.set_xticks(range(n_games))
ax.set_xticklabels(game_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(n_models))
ax.set_yticklabels(MODEL_LABELS, fontsize=10)

ax.set_title("Individual Benchmark Scores by Model", fontsize=13, fontweight="bold", pad=12)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cbar.set_label("Rank (1 = best)", fontsize=9)
cbar.set_ticks(list(range(1, n_models + 1)))
cbar.set_ticklabels([str(r) for r in range(1, n_models + 1)])

plt.tight_layout()
out_path = "paper/plot_heatmap_rank.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close()

# ── Second plot: raw avg_total_reward with shift-then-normalize coloring ──────
fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))

cmap_raw = mcolors.LinearSegmentedColormap.from_list(
    "rg", ["#d73027", "#fee08b", "#1a9850"]
)

col_min = matrix.min(axis=0)
shift = np.where(col_min < 0, -col_min, 0)
shifted = matrix + shift
col_max_shifted = shifted.max(axis=0)
col_max_shifted[col_max_shifted == 0] = 1
norm_matrix = shifted / col_max_shifted

im2 = ax2.imshow(norm_matrix, aspect="auto", cmap=cmap_raw, vmin=0, vmax=1)

for i in range(n_models):
    for j in range(n_games):
        nv = norm_matrix[i, j]
        text_color = "black" if 0.35 < nv < 0.85 else "white"
        ax2.text(
            j, i,
            f"{matrix[i, j]:.0f}",
            ha="center", va="center",
            fontsize=7.5,
            color=text_color,
            fontweight="bold",
        )

ax2.set_xticks(range(n_games))
ax2.set_xticklabels(game_labels, rotation=45, ha="right", fontsize=9)
ax2.set_yticks(range(n_models))
ax2.set_yticklabels(MODEL_LABELS, fontsize=10)
ax2.set_title("Individual Benchmark Scores by Model (Raw)", fontsize=13, fontweight="bold", pad=12)

cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.025, pad=0.01)
cbar2.set_label("Normalized Score (per game)", fontsize=9)
cbar2.set_ticks([0, 0.5, 1])
cbar2.set_ticklabels(["Low", "Mid", "High"])

plt.tight_layout()
out_path2 = "paper/plot_heatmap_raw.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path2}")
plt.close()
