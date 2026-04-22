"""
Cluster games by model-rank profile.

Each game is a 4-D vector of model ranks (rank 1 = best model on that game).
Games that cluster together are ones where the same models tend to rank
high/low — i.e. they appear to test the same underlying capability.

Produces:
  1. K-means cluster heatmap  — rank profile per cluster, games annotated
  2. PCA scatter              — 2-D projection of games, colored by cluster
                                and shaped by taxonomy group
  3. Silhouette sweep         — choose k automatically
  4. Positive-correlation subgraph — edges only where r > threshold
"""
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

# ── config (mirrors plot_game_correlation.py) ────────────────────────────────
MODEL_ALIASES = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "deepseek-ai/deepseek-v3.1": "deepseek-ai/deepseek-v3.1",
    "zai-org/glm-5.1": "zai-org/glm-5.1",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "google:gemini-3-flash-preview": "gemini-3-flash-preview",
    "google:gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "random": "random",
}
EXCLUDE_ROW_KEYS = {
    "gemini-3.1-pro-preview|structured_history|high",
    "gemini-3.1-pro-preview|append_only|high",
}
CORR_MODELS = {
    "gemini-2.5-flash|structured_history|none",
    "deepseek-ai/deepseek-v3.1|structured_history|none",
    "zai-org/glm-5.1|structured_history|none",
    "gpt-5.4-mini|structured_history|none",
    "gemini-3-flash-preview|structured_history|low",
    "gemini-3-flash-preview|structured_history|high",
}
MODEL_LABELS = {
    "gemini-2.5-flash|structured_history|none":          "Gemini 2.5 Flash",
    "deepseek-ai/deepseek-v3.1|structured_history|none": "DeepSeek V3.1",
    "zai-org/glm-5.1|structured_history|none":           "GLM 5.1",
    "gpt-5.4-mini|structured_history|none":              "GPT-5.4 Mini",
    "gemini-3-flash-preview|structured_history|low":     "Gemini 3 Flash (Low)",
    "gemini-3-flash-preview|structured_history|high":    "Gemini 3 Flash (High)",
}

TAXONOMY = {
    "Shooter": ["air_raid", "assault", "beam_rider", "demon_attack",
                "laser_gates", "name_this_game", "phoenix", "riverraid",
                "seaquest", "time_pilot", "robotank"],
    "Sports":  ["boxing", "ice_hockey", "fishing_derby", "tennis"],
    "Action":  ["breakout", "freeway", "gopher", "journey_escape"],
    "Maze":    ["pacman", "qbert"],
}
GAME_TO_GROUP = {g: grp for grp, gs in TAXONOMY.items() for g in gs}
TAX_COLORS = {"Shooter": "#4393c3", "Sports": "#d6604d",
              "Action": "#74c476",  "Maze": "#e08214"}
TAX_MARKERS = {"Shooter": "o", "Sports": "s", "Action": "^", "Maze": "D"}

SKILL_GROUPS = {
    "Shoot & Dodge":    ["air_raid", "assault", "beam_rider", "demon_attack",
                         "name_this_game", "phoenix"],
    "Navigate & Shoot": ["laser_gates", "riverraid", "robotank", "seaquest",
                         "time_pilot"],
    "Track & React":    ["boxing", "breakout", "fishing_derby", "ice_hockey",
                         "tennis"],
    "Evasion":          ["freeway", "gopher", "journey_escape"],
    "Route Planning":   ["pacman", "qbert"],
}
GAME_TO_SKILL = {g: sk for sk, gs in SKILL_GROUPS.items() for g in gs}


def make_row_key(entry):
    canonical = MODEL_ALIASES.get(entry.get("model_name", ""))
    if canonical is None:
        return None
    pm = entry.get("prompt_mode") or "structured_history"
    tl = entry.get("thinking_level")
    tl = tl if tl and tl != "none" else "none"
    rk = f"{canonical}|{pm}|{tl}"
    return None if rk in EXCLUDE_ROW_KEYS else rk


SNR_THRESHOLD = 1.0  # keep games where (best-worst gap) / pooled_stderr >= this
EXCLUDE_GAMES = {"boxing", "ice_hockey", "demon_attack", "fishing_derby"}

# ── data loading — collect avg + stderr per (model, game) ────────────────────
entries_data = {}  # (rk, game) -> (avg, stderr, run_count)
for p in (ROOT / "runs").rglob("*.json"):
    if "30s" not in p.name:
        continue
    with p.open() as f:
        payload = json.load(f)
    for e in payload.get("entries", []):
        rk = make_row_key(e)
        game = e.get("game")
        avg = e.get("avg_total_reward")
        se  = e.get("stderr_total_reward") or 0.0
        n   = e.get("run_count") or 1
        if rk not in CORR_MODELS or game is None or avg is None:
            continue
        key = (rk, game)
        if key not in entries_data or n > entries_data[key][2]:
            entries_data[key] = (avg, se, n)

reward = {k: v[0] for k, v in entries_data.items()}

sh_keys   = sorted({rk for rk, _ in entries_data.keys()})
all_games = sorted({g  for _, g in entries_data.keys()})
n_models  = len(sh_keys)
model_labels = [MODEL_LABELS[rk] for rk in sh_keys]

# ── per-game separability: (best_mean - worst_mean) / pooled_stderr ──────────
# Keeps a game if the best and worst model are distinguishable above noise.
game_snr = {}
for game in all_games:
    pairs = [(entries_data[(rk, game)][0], entries_data[(rk, game)][1])
             for rk in sh_keys if (rk, game) in entries_data]
    if len(pairs) < 2:
        game_snr[game] = 0.0
        continue
    avgs = [a for a, _ in pairs]
    best_se  = min(pairs, key=lambda x: -x[0])[1]   # stderr of top model
    worst_se = min(pairs, key=lambda x:  x[0])[1]   # stderr of bottom model
    gap = max(avgs) - min(avgs)
    pooled_se = np.sqrt(best_se**2 + worst_se**2)
    game_snr[game] = gap / pooled_se if pooled_se > 0 else 99.0

# ── Figure 0 — separability bar chart ────────────────────────────────────────
snr_games  = sorted(all_games, key=lambda g: game_snr[g], reverse=True)
snr_values = [game_snr[g] for g in snr_games]
colors_bar = ["#1a9850" if v >= SNR_THRESHOLD else "#d73027" for v in snr_values]

fig0s, ax0s = plt.subplots(figsize=(7, 6))
bars = ax0s.barh(range(len(snr_games)), snr_values, color=colors_bar,
                 edgecolor="white", height=0.7)
ax0s.axvline(SNR_THRESHOLD, color="#333333", lw=1.5, ls="--",
             label=f"Threshold = {SNR_THRESHOLD}")
ax0s.set_yticks(range(len(snr_games)))
ax0s.set_yticklabels([g.replace("_", " ").title() for g in snr_games], fontsize=11)
ax0s.invert_yaxis()
ax0s.set_xlabel("Separability  =  (best − worst mean) / pooled stderr", fontsize=11)
ax0s.set_title("Per-game separability\n(green = kept, red = dropped)",
               fontsize=12, fontweight="bold")
ax0s.legend(fontsize=10)
for i, v in enumerate(snr_values):
    ax0s.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=10)
ax0s.set_xlim(0, max(snr_values) * 1.18)
ax0s.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
p0s = ROOT / "paper" / "correlation" / "plot_clusters_separability.png"
plt.savefig(p0s, dpi=150, bbox_inches="tight"); print(f"Saved to {p0s}"); plt.close()

# filter to separable games
all_games = [g for g in all_games if game_snr[g] >= SNR_THRESHOLD and g not in EXCLUDE_GAMES]
n_games   = len(all_games)
game_labels = [g.replace("_", " ").title() for g in all_games]
print(f"Kept {n_games}/21 games (SNR >= {SNR_THRESHOLD}):")
print(f"  Kept:    {all_games}")
dropped = [g for g in sorted(game_snr) if game_snr[g] < SNR_THRESHOLD]
print(f"  Dropped: {dropped}")

# ── raw score matrix [n_models × n_games] ────────────────────────────────────
matrix = np.full((n_models, n_games), np.nan)
for i, rk in enumerate(sh_keys):
    for j, game in enumerate(all_games):
        if (rk, game) in reward:
            matrix[i, j] = reward[(rk, game)]

rank_matrix = np.full_like(matrix, np.nan)
for j in range(n_games):
    col = matrix[:, j]
    valid_idx = np.where(~np.isnan(col))[0]
    if len(valid_idx) < 2:
        continue
    scores = col[valid_idx]
    order  = np.argsort(-scores)
    i2 = 0
    while i2 < len(order):
        j2 = i2
        while j2 + 1 < len(order) and scores[order[j2 + 1]] == scores[order[i2]]:
            j2 += 1
        avg_rank = (i2 + j2) / 2.0 + 1
        for kk in range(i2, j2 + 1):
            rank_matrix[valid_idx[order[kk]], j] = avg_rank
        i2 = j2 + 1

# Game feature matrix X: shape [n_games × n_models]
X = rank_matrix.T  # each row = one game, each col = one model's rank

# ── Figure 1 — silhouette sweep to choose k ──────────────────────────────────
K_RANGE = range(2, 8)
sil_scores = []
inertias = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels_k = km.fit_predict(X)
    sil_scores.append(silhouette_score(X, labels_k))
    inertias.append(km.inertia_)

fig0, (ax0a, ax0b) = plt.subplots(1, 2, figsize=(9, 3.5))
ax0a.plot(list(K_RANGE), sil_scores, "o-", color="#4393c3", lw=2)
ax0a.set_xlabel("k", fontsize=9); ax0a.set_ylabel("Silhouette score", fontsize=9)
ax0a.set_title("Silhouette score vs k", fontsize=10, fontweight="bold")
ax0a.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax0b.plot(list(K_RANGE), inertias, "o-", color="#d6604d", lw=2)
ax0b.set_xlabel("k", fontsize=9); ax0b.set_ylabel("Inertia (within-cluster SS)", fontsize=9)
ax0b.set_title("Elbow curve", fontsize=10, fontweight="bold")
ax0b.xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.tight_layout()
p0 = ROOT / "paper" / "correlation" / "plot_clusters_sweep.png"
plt.savefig(p0, dpi=150, bbox_inches="tight"); print(f"Saved to {p0}"); plt.close()

best_k = min(int(K_RANGE[int(np.argmax(sil_scores))]), 5)  # cap at 5 to avoid singletons
print(f"Best k by silhouette: {best_k}  (scores: {dict(zip(K_RANGE, [f'{s:.3f}' for s in sil_scores]))})")

# ── run final k-means with best k ────────────────────────────────────────────
K = best_k
km_final = KMeans(n_clusters=K, random_state=42, n_init=20)
cluster_ids = km_final.fit_predict(X)  # shape [n_games]

# Sort clusters by mean rank of best model (so cluster 0 = easiest for top model)
cluster_mean_rank = [X[cluster_ids == c].mean() for c in range(K)]
cluster_order = np.argsort(cluster_mean_rank)  # low mean rank = models agree on ordering
relabel = {old: new for new, old in enumerate(cluster_order)}
cluster_ids = np.array([relabel[c] for c in cluster_ids])

CLUSTER_COLORS = plt.cm.Set2(np.linspace(0, 0.9, K))

# ── Figure 2 — rank profile heatmap per cluster ──────────────────────────────
# Sort games within each cluster by their mean rank across models
sorted_game_idx = []
for c in range(K):
    in_c = np.where(cluster_ids == c)[0]
    in_c_sorted = in_c[np.argsort(X[in_c].mean(axis=1))]
    sorted_game_idx.extend(in_c_sorted)

sorted_games  = [all_games[i]  for i in sorted_game_idx]
sorted_labels = [game_labels[i] for i in sorted_game_idx]
sorted_X      = X[sorted_game_idx]          # [n_games × n_models]
sorted_ids    = cluster_ids[sorted_game_idx]

fig1, ax1 = plt.subplots(figsize=(7, 9))
im = ax1.imshow(sorted_X, aspect="auto",
                cmap="RdYlGn_r", vmin=1, vmax=n_models, interpolation="nearest")

for i in range(n_games):
    for j in range(n_models):
        rv = sorted_X[i, j]
        if not np.isnan(rv):
            tc = "white" if rv in (1, n_models) else "black"
            ax1.text(j, i, f"{rv:.0f}", ha="center", va="center",
                     fontsize=8, color=tc, fontweight="bold")

ax1.set_xticks(range(n_models))
ax1.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=8)
ax1.set_yticks(range(n_games))
ax1.set_yticklabels(sorted_labels, fontsize=7.5)

# Color y-tick labels by cluster; draw separator lines between clusters
boundaries = []
for i in range(1, n_games):
    if sorted_ids[i] != sorted_ids[i - 1]:
        boundaries.append(i - 0.5)
for b in boundaries:
    ax1.axhline(b, color="black", lw=1.5)
for i, g in enumerate(sorted_games):
    c = sorted_ids[i]
    ax1.get_yticklabels()[i].set_color(CLUSTER_COLORS[c])

# Cluster labels on right side
prev = 0
for b_idx, b in enumerate(boundaries + [n_games - 0.5]):
    end = int(b + 0.5)
    mid = (prev + end - 1) / 2.0
    c   = sorted_ids[prev]
    ax1.text(n_models + 0.15, mid, f"C{c+1}",
             ha="left", va="center", fontsize=9, fontweight="bold",
             color=CLUSTER_COLORS[c], clip_on=False)
    prev = end

# Taxonomy group markers on left
for i, g in enumerate(sorted_games):
    grp = GAME_TO_GROUP.get(g, "")
    ax1.text(-0.6, i, grp[0], ha="center", va="center",
             fontsize=6.5, color=TAX_COLORS.get(grp, "gray"),
             fontweight="bold", clip_on=False)
ax1.text(-0.6, -1.0, "Tax", ha="center", va="center",
         fontsize=6, color="gray", clip_on=False)

cbar = fig1.colorbar(im, ax=ax1, fraction=0.025, pad=0.08)
cbar.set_label("Model rank (1=best)", fontsize=8)
cbar.set_ticks(range(1, n_models + 1))
cbar.set_ticklabels([str(r) for r in range(1, n_models + 1)])
cbar.ax.tick_params(labelsize=7)

ax1.set_title(f"Game clusters by rank profile  (k={K}, k-means)",
              fontsize=10, fontweight="bold", pad=10)
plt.tight_layout()
p1 = ROOT / "paper" / "correlation" / "plot_clusters_heatmap.png"
plt.savefig(p1, dpi=150, bbox_inches="tight"); print(f"Saved to {p1}"); plt.close()

# ── Figure 3 — PCA scatter ───────────────────────────────────────────────────
pca  = PCA(n_components=2)
X_pc = pca.fit_transform(X)
var  = pca.explained_variance_ratio_

fig2, ax2 = plt.subplots(figsize=(8, 6))
for i, g in enumerate(all_games):
    c   = cluster_ids[i]
    grp = GAME_TO_GROUP.get(g, "")
    ax2.scatter(X_pc[i, 0], X_pc[i, 1],
                color=CLUSTER_COLORS[c], marker=TAX_MARKERS.get(grp, "o"),
                s=90, edgecolors="white", linewidths=0.6, zorder=3)
    ax2.annotate(g.replace("_", " ").title(),
                 (X_pc[i, 0], X_pc[i, 1]),
                 fontsize=6.5, textcoords="offset points", xytext=(5, 3),
                 color=CLUSTER_COLORS[c])

# Draw cluster convex hulls
from scipy.spatial import ConvexHull
for c in range(K):
    pts = X_pc[cluster_ids == c]
    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            verts = np.append(hull.vertices, hull.vertices[0])
            ax2.fill(pts[hull.vertices, 0], pts[hull.vertices, 1],
                     alpha=0.10, color=CLUSTER_COLORS[c])
            ax2.plot(pts[verts, 0], pts[verts, 1],
                     color=CLUSTER_COLORS[c], lw=1.2, ls="--", alpha=0.6)
        except Exception:
            pass

ax2.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=9)
ax2.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", fontsize=9)
ax2.set_title("PCA of games by rank profile", fontsize=10, fontweight="bold")

# PCA loadings as arrows
scale = 1.5
for j, ml in enumerate(model_labels):
    dx, dy = pca.components_[0, j] * scale, pca.components_[1, j] * scale
    ax2.annotate("", xy=(dx, dy), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))
    ax2.text(dx * 1.1, dy * 1.1, ml, fontsize=7, color="#333333", ha="center")

# Legends
cluster_handles = [mpatches.Patch(color=CLUSTER_COLORS[c], label=f"Cluster {c+1}")
                   for c in range(K)]
tax_handles = [plt.scatter([], [], marker=TAX_MARKERS[g], color="gray",
                           s=50, label=g) for g in TAXONOMY]
ax2.legend(handles=cluster_handles + tax_handles,
           fontsize=7, loc="upper left", ncol=2, framealpha=0.85)
ax2.axhline(0, color="#cccccc", lw=0.8); ax2.axvline(0, color="#cccccc", lw=0.8)
plt.tight_layout()
p2 = ROOT / "paper" / "correlation" / "plot_clusters_pca.png"
plt.savefig(p2, dpi=150, bbox_inches="tight"); print(f"Saved to {p2}"); plt.close()

# ── Figure 4 — positive-correlation subgraph ─────────────────────────────────
# Recompute correlation matrix (same logic as plot_game_correlation.py)
corr = np.full((n_games, n_games), np.nan)
for i in range(n_games):
    for j in range(n_games):
        xi, xj = rank_matrix[:, i], rank_matrix[:, j]
        both = ~np.isnan(xi) & ~np.isnan(xj)
        if both.sum() < 2:
            continue
        a, b = xi[both], xj[both]
        if a.std() == 0 or b.std() == 0:
            continue
        corr[i, j] = np.corrcoef(a, b)[0, 1]

THRESHOLD = 0.5
fig3, ax3 = plt.subplots(figsize=(7, 6))
ax3.set_xlim(-1.45, 1.45); ax3.set_ylim(-1.45, 1.45)
ax3.set_aspect("equal"); ax3.axis("off")
ax3.set_title(f"Games connected when Pearson r > {THRESHOLD}",
              fontsize=12, fontweight="bold")

# Circular layout ordered by cluster then by game name
angles = np.linspace(0, 2 * np.pi, n_games, endpoint=False)
node_order = sorted(range(n_games),
                    key=lambda i: (cluster_ids[i], all_games[i]))
pos = {node_order[k]: (np.cos(angles[k]), np.sin(angles[k]))
       for k in range(n_games)}

# Draw edges
for i in range(n_games):
    for j in range(i + 1, n_games):
        r = corr[i, j]
        if np.isnan(r) or r <= THRESHOLD:
            continue
        xi, yi = pos[i]; xj, yj = pos[j]
        alpha = min(1.0, (r - THRESHOLD) / (1.0 - THRESHOLD))
        ax3.plot([xi, xj], [yi, yj], color="#888888",
                 lw=1.0 + 2.5 * alpha, alpha=0.35 + 0.45 * alpha, zorder=1)

# Draw nodes — uniform circle marker, color by cluster only
for i, g in enumerate(all_games):
    xi, yi = pos[i]
    c = cluster_ids[i]
    ax3.scatter(xi, yi, s=160, color=CLUSTER_COLORS[c],
                marker="o", edgecolors="white", linewidths=1.0, zorder=3)
    ha  = "left" if xi > 0 else "right"
    off = 0.09 if xi > 0 else -0.09
    ax3.text(xi + off, yi, g.replace("_", " ").title(),
             ha=ha, va="center", fontsize=9,
             color=CLUSTER_COLORS[c], fontweight="bold")

plt.tight_layout()
p3 = ROOT / "paper" / "correlation" / "plot_clusters_graph.png"
plt.savefig(p3, dpi=150, bbox_inches="tight"); print(f"Saved to {p3}"); plt.close()

# ── Figure 5 — model deviation from global rank per cluster ─────────────────
# For each (model, cluster) cell: show rank delta vs that model's global avg.
# Red = model underperforms its average here; green = overperforms.
global_mean_rank = rank_matrix.mean(axis=1)  # [n_models]

delta_matrix = np.zeros((n_models, K))
cluster_sizes = []
for c in range(K):
    idx = [i for i in range(n_games) if cluster_ids[i] == c]
    cluster_sizes.append(len(idx))
    for mi in range(n_models):
        cluster_ranks = rank_matrix[mi, idx]
        delta_matrix[mi, c] = np.nanmean(cluster_ranks) - global_mean_rank[mi]

GAME_ATTRS_LOCAL = {
    "air_raid":       ("Shoot&Dodge",   "Dense",  "Low",   "High"),
    "assault":        ("Shoot&Dodge",   "Dense",  "Low",   "High"),
    "beam_rider":     ("Shoot&Dodge",   "Dense",  "High",  "High"),
    "demon_attack":   ("Shoot&Dodge",   "Dense",  "Low",   "High"),
    "name_this_game": ("Shoot&Dodge",   "Dense",  "Low",   "High"),
    "phoenix":        ("Shoot&Dodge",   "Dense",  "Low",   "High"),
    "laser_gates":    ("Nav&Shoot",     "Dense",  "High",  "High"),
    "riverraid":      ("Nav&Shoot",     "Dense",  "High",  "High"),
    "robotank":       ("Nav&Shoot",     "Dense",  "High",  "High"),
    "seaquest":       ("Nav&Shoot",     "Dense",  "High",  "High"),
    "time_pilot":     ("Nav&Shoot",     "Sparse", "High",  "High"),
    "boxing":         ("Track&React",   "Dense",  "Low",   "High"),
    "breakout":       ("Track&React",   "Dense",  "Low",   "Med"),
    "fishing_derby":  ("Track&React",   "Dense",  "High",  "High"),
    "ice_hockey":     ("Track&React",   "Sparse", "Low",   "Med"),
    "tennis":         ("Track&React",   "Sparse", "Low",   "Med"),
    "freeway":        ("Evasion",       "Sparse", "High",  "High"),
    "gopher":         ("Evasion",       "Dense",  "High",  "High"),
    "journey_escape": ("Evasion",       "Dense",  "High",  "High"),
    "pacman":         ("RoutePlan",     "Dense",  "High",  "High"),
    "qbert":          ("RoutePlan",     "Dense",  "High",  "Low"),
}

def attr_summary(games, attr_idx):
    vals = [GAME_ATTRS_LOCAL[g][attr_idx] for g in games if g in GAME_ATTRS_LOCAL]
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return ", ".join(f"{v}×{n}" for v, n in sorted(counts.items(), key=lambda x: -x[1]))

CLUSTER_INTERPRETATIONS = [
    "Patience & restraint\n(sparse reward or careful timing)",
    "Precise target alignment\n(Gemini collapses here)",
    "Fast multi-threat tracking\n(GLM & GPT-mini fall behind)",
    "Multi-dir scrolling shooter\n(GLM-specific strength)",
    "Structured spatial navigation\n(DeepSeek struggles)",
]

fig5, axes5 = plt.subplots(1, 2, figsize=(13, 5),
                            gridspec_kw={"width_ratios": [1, 1.6]})

# Left: delta heatmap
ax_d = axes5[0]
vmax = max(abs(delta_matrix).max(), 0.5)
im5 = ax_d.imshow(delta_matrix, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax,
                  aspect="auto", interpolation="nearest")
cluster_xlabels = [f"C{c+1}\n(n={cluster_sizes[c]})" for c in range(K)]
ax_d.set_xticks(range(K))
ax_d.set_xticklabels(cluster_xlabels, fontsize=8)
ax_d.set_yticks(range(n_models))
ax_d.set_yticklabels(model_labels, fontsize=8)
ax_d.set_title("Rank deviation from global avg\n(red = worse than usual, green = better)",
               fontsize=9, fontweight="bold")
for mi in range(n_models):
    for c in range(K):
        v = delta_matrix[mi, c]
        tc = "white" if abs(v) > vmax * 0.6 else "black"
        ax_d.text(c, mi, f"{v:+.2f}", ha="center", va="center",
                  fontsize=8.5, color=tc, fontweight="bold")
cbar5 = fig5.colorbar(im5, ax=ax_d, fraction=0.04, pad=0.03)
cbar5.set_label("Δ rank", fontsize=7); cbar5.ax.tick_params(labelsize=7)

# Right: cluster attribute summary + interpretation
ax_t = axes5[1]
ax_t.axis("off")
col_headers = ["Cluster", "Games", "Reward\ndensity", "Skill\ngroups", "Interpretation"]
rows = []
for c in range(K):
    members = [all_games[i] for i in range(n_games) if cluster_ids[i] == c]
    rdensity  = attr_summary(members, 1)
    skills    = attr_summary(members, 0)
    rows.append([f"C{c+1}", "\n".join(g.replace("_"," ").title() for g in members),
                 rdensity, skills,
                 CLUSTER_INTERPRETATIONS[c] if c < len(CLUSTER_INTERPRETATIONS) else ""])

col_widths = [0.05, 0.22, 0.12, 0.18, 0.30]
x_positions = [sum(col_widths[:i]) + col_widths[i]/2 for i in range(len(col_widths))]
row_height = 1.0 / (len(rows) + 1.5)
header_y = 1.0 - row_height * 0.5

for xi, (hdr, xp) in enumerate(zip(col_headers, x_positions)):
    ax_t.text(xp, header_y, hdr, ha="center", va="center",
              fontsize=8, fontweight="bold",
              transform=ax_t.transAxes)

for ri, row in enumerate(rows):
    y = header_y - row_height * (ri + 1)
    c = ri
    bg = mpatches.FancyBboxPatch((0.0, y - row_height*0.48), 1.0, row_height*0.95,
                                  boxstyle="round,pad=0.01",
                                  facecolor=(*CLUSTER_COLORS[c][:3], 0.12),
                                  edgecolor=(*CLUSTER_COLORS[c][:3], 0.5),
                                  transform=ax_t.transAxes, clip_on=False)
    ax_t.add_patch(bg)
    for xi, (cell, xp) in enumerate(zip(row, x_positions)):
        ax_t.text(xp, y, cell, ha="center", va="center",
                  fontsize=6.8 if xi == 1 else 7.5,
                  color=CLUSTER_COLORS[c] if xi == 0 else "black",
                  fontweight="bold" if xi in (0, 4) else "normal",
                  transform=ax_t.transAxes, linespacing=1.3)

ax_t.set_title("Cluster attribute summary & interpretation",
               fontsize=9, fontweight="bold", pad=6)

plt.suptitle("Why do the clusters form? — model capability gaps per game group",
             fontsize=11, fontweight="bold")
plt.tight_layout()
p5 = ROOT / "paper" / "correlation" / "plot_clusters_explanation.png"
plt.savefig(p5, dpi=150, bbox_inches="tight"); print(f"Saved to {p5}"); plt.close()

# ── console summary ───────────────────────────────────────────────────────────
print(f"\n=== K-means clusters (k={K}) ===")
for c in range(K):
    members = [all_games[i] for i in range(n_games) if cluster_ids[i] == c]
    centroid = X[cluster_ids == c].mean(axis=0)
    skill_breakdown = {}
    for g in members:
        sk = GAME_TO_SKILL.get(g, "?")
        skill_breakdown[sk] = skill_breakdown.get(sk, 0) + 1
    best_model  = model_labels[int(np.argmin(centroid))]
    worst_model = model_labels[int(np.argmax(centroid))]
    print(f"  C{c+1} ({len(members)} games): {members}")
    print(f"       centroid ranks: { {ml: f'{r:.1f}' for ml, r in zip(model_labels, centroid)} }")
    print(f"       best model: {best_model}  worst: {worst_model}")
    print(f"       skill groups: {skill_breakdown}")
