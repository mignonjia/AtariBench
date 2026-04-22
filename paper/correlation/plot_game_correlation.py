"""
Game-vs-game correlation map (21x21).

Each game is represented as a vector of 7 models' normalized scores.
Pairwise Pearson correlation uses only the model entries where both games
have non-NaN data, so missing entries (e.g. Gemini 3.1 Pro Low on 3 games)
are handled gracefully.

Produces three figures:
  1. Correlation heatmap ordered by taxonomy groups
  2. Correlation heatmap ordered by hierarchical clustering (+ dendrogram)
  3. Summary bar chart: within-group vs between-group avg correlation
"""
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[2]

# ── taxonomy (from method.tex) ───────────────────────────────────────────────
TAXONOMY = {
    "Shooter": [
        "air_raid", "assault", "beam_rider", "demon_attack",
        "laser_gates", "name_this_game", "phoenix", "riverraid",
        "seaquest", "time_pilot", "robotank",
    ],
    "Sports": ["boxing", "ice_hockey", "fishing_derby", "tennis"],
    "Action": ["breakout", "freeway", "gopher", "journey_escape"],
    "Maze":   ["pacman", "qbert"],
}
GROUP_COLORS = {
    "Shooter": "#4393c3",
    "Sports":  "#d6604d",
    "Action":  "#74c476",
    "Maze":    "#e08214",
}
GAME_TO_GROUP = {g: grp for grp, games in TAXONOMY.items() for g in games}

# ── model config (must match plot_heatmap.py) ────────────────────────────────
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


def make_row_key(entry):
    canonical = MODEL_ALIASES.get(entry.get("model_name", ""))
    if canonical is None:
        return None
    pm = entry.get("prompt_mode") or "structured_history"
    tl = entry.get("thinking_level")
    tl = tl if tl and tl != "none" else "none"
    rk = f"{canonical}|{pm}|{tl}"
    return None if rk in EXCLUDE_ROW_KEYS else rk


# ── data loading ─────────────────────────────────────────────────────────────
reward = {}
for p in (ROOT / "runs").rglob("*.json"):
    if "30s" not in p.name:
        continue
    with p.open() as f:
        payload = json.load(f)
    for e in payload.get("entries", []):
        rk = make_row_key(e)
        game = e.get("game")
        val = e.get("avg_total_reward")
        if rk is None or game is None or val is None:
            continue
        key = (rk, game)
        if key not in reward or val > reward[key]:
            reward[key] = val

CORR_MODELS = {
    "gemini-2.5-flash|structured_history|none",
    "deepseek-ai/deepseek-v3.1|structured_history|none",
    "zai-org/glm-5.1|structured_history|none",
    "gpt-5.4-mini|structured_history|none",
    "gemini-3-flash-preview|structured_history|low",
    "gemini-3-flash-preview|structured_history|high",
}

sh_keys = sorted(
    {rk for rk, _ in reward.keys() if rk in CORR_MODELS}
)
all_games = sorted({g for _, g in reward.keys()})
n_models = len(sh_keys)
n_games  = len(all_games)
print(f"Models ({n_models}): {sh_keys}")
print(f"Games  ({n_games}):  {all_games}")

# ── build raw score matrix [n_models × n_games] ─────────────────────────────
matrix = np.full((n_models, n_games), np.nan)
for i, rk in enumerate(sh_keys):
    for j, game in enumerate(all_games):
        if (rk, game) in reward:
            matrix[i, j] = reward[(rk, game)]

# ── rank matrix: rank models within each game (1 = best, ties share avg rank)
rank_matrix = np.full_like(matrix, np.nan)
for j in range(n_games):
    col = matrix[:, j]
    valid_idx = np.where(~np.isnan(col))[0]
    if len(valid_idx) < 2:
        continue
    scores = col[valid_idx]
    # scipy-style average rank: argsort descending, assign avg rank for ties
    order = np.argsort(-scores)
    ranks = np.empty(len(scores))
    i2 = 0
    while i2 < len(order):
        j2 = i2
        while j2 + 1 < len(order) and scores[order[j2 + 1]] == scores[order[i2]]:
            j2 += 1
        avg_rank = (i2 + j2) / 2.0 + 1  # 1-based
        for k in range(i2, j2 + 1):
            ranks[order[k]] = avg_rank
        i2 = j2 + 1
    rank_matrix[valid_idx, j] = ranks

# ── pairwise Pearson correlation between games (on ranks) ────────────────────
corr = np.full((n_games, n_games), np.nan)
n_overlap = np.zeros((n_games, n_games), dtype=int)
for i in range(n_games):
    for j in range(n_games):
        xi, xj = rank_matrix[:, i], rank_matrix[:, j]
        both = ~np.isnan(xi) & ~np.isnan(xj)
        n = both.sum()
        n_overlap[i, j] = n
        if n < 2:
            continue
        a, b = xi[both], xj[both]
        if a.std() == 0 or b.std() == 0:
            continue
        corr[i, j] = np.corrcoef(a, b)[0, 1]


# ── helpers ──────────────────────────────────────────────────────────────────
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "bwr_clipped", ["#2166ac", "#f7f7f7", "#d6604d"]
)


def draw_heatmap(ax, ordered_games, title, show_values=True):
    idx = [all_games.index(g) for g in ordered_games]
    sub = corr[np.ix_(idx, idx)]
    n = len(ordered_games)
    im = ax.imshow(sub, cmap=CMAP, vmin=-1, vmax=1, aspect="auto",
                   interpolation="nearest")
    labels = [g.replace("_", " ").title() for g in ordered_games]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    if show_values:
        for i in range(n):
            for j in range(n):
                v = sub[i, j]
                if np.isnan(v):
                    ax.text(j, i, "NA", ha="center", va="center",
                            fontsize=4.5, color="#888888", fontstyle="italic")
                else:
                    tc = "white" if abs(v) > 0.6 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=4.5, color=tc, fontweight="bold")
    return im, sub


def draw_group_separators(ax, ordered_games):
    """Draw lines between taxonomy groups and color y-tick labels."""
    boundaries = []
    prev_grp = GAME_TO_GROUP.get(ordered_games[0])
    for k, g in enumerate(ordered_games):
        grp = GAME_TO_GROUP.get(g)
        if grp != prev_grp:
            boundaries.append(k - 0.5)
            prev_grp = grp
    for b in boundaries:
        ax.axhline(b, color="#333333", lw=1.2, zorder=5)
        ax.axvline(b, color="#333333", lw=1.2, zorder=5)
    # Color tick labels by group
    for tick, g in zip(ax.get_yticklabels(), ordered_games):
        tick.set_color(GROUP_COLORS.get(GAME_TO_GROUP.get(g, ""), "black"))
    for tick, g in zip(ax.get_xticklabels(), ordered_games):
        tick.set_color(GROUP_COLORS.get(GAME_TO_GROUP.get(g, ""), "black"))


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — taxonomy-ordered heatmap
# ════════════════════════════════════════════════════════════════════════════
taxonomy_order = [g for grp in ["Shooter", "Sports", "Action", "Maze"]
                    for g in TAXONOMY[grp] if g in all_games]

fig1, ax1 = plt.subplots(figsize=(9, 8.5))
im1, _ = draw_heatmap(ax1, taxonomy_order,
                      "Game–Game Correlation (taxonomy order)")
draw_group_separators(ax1, taxonomy_order)

legend_patches = [mpatches.Patch(color=c, label=g)
                  for g, c in GROUP_COLORS.items()]
ax1.legend(handles=legend_patches, loc="upper right", fontsize=7,
           framealpha=0.9, title="Taxonomy", title_fontsize=7)

cbar1 = fig1.colorbar(im1, ax=ax1, fraction=0.03, pad=0.02)
cbar1.set_label("Pearson r", fontsize=8)
cbar1.ax.tick_params(labelsize=7)

plt.tight_layout()
p1 = ROOT / "paper" / "correlation" / "plot_corr_taxonomy.png"
plt.savefig(p1, dpi=150, bbox_inches="tight")
print(f"Saved to {p1}")
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — hierarchical clustering heatmap + dendrogram
# ════════════════════════════════════════════════════════════════════════════
# Convert correlation to distance; fill NaN diagonal→1 for distance
corr_filled = np.where(np.isnan(corr), 0.0, corr)
np.fill_diagonal(corr_filled, 1.0)
dist = np.clip(1.0 - corr_filled, 0.0, 2.0)
np.fill_diagonal(dist, 0.0)
condensed = squareform(dist, checks=False)

Z = linkage(condensed, method="average")
dend_order = dendrogram(Z, no_plot=True)["leaves"]
cluster_games = [all_games[i] for i in dend_order]

fig2 = plt.figure(figsize=(12, 9.5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4], wspace=0.04)
ax_dend = fig2.add_subplot(gs[0])
ax_heat = fig2.add_subplot(gs[1])

# Dendrogram (rotated left)
dendrogram(
    Z, orientation="left", ax=ax_dend,
    labels=[g.replace("_", " ").title() for g in all_games],
    color_threshold=0.6 * max(Z[:, 2]),
    leaf_font_size=6.5,
)
ax_dend.invert_yaxis()
ax_dend.set_xlabel("Distance", fontsize=7)
ax_dend.tick_params(axis="y", labelsize=6.5)
# Color dendrogram leaf labels by group
for tick in ax_dend.get_yticklabels():
    raw = tick.get_text().lower().replace(" ", "_")
    tick.set_color(GROUP_COLORS.get(GAME_TO_GROUP.get(raw, ""), "black"))
ax_dend.spines[["top", "right", "left"]].set_visible(False)

im2, _ = draw_heatmap(ax_heat, cluster_games,
                      "Game–Game Correlation (hierarchical clustering order)",
                      show_values=True)
# Color heat tick labels by group (no separator lines here)
for tick, g in zip(ax_heat.get_yticklabels(), cluster_games):
    tick.set_color(GROUP_COLORS.get(GAME_TO_GROUP.get(g, ""), "black"))
for tick, g in zip(ax_heat.get_xticklabels(), cluster_games):
    tick.set_color(GROUP_COLORS.get(GAME_TO_GROUP.get(g, ""), "black"))

cbar2 = fig2.colorbar(im2, ax=ax_heat, fraction=0.025, pad=0.02)
cbar2.set_label("Pearson r", fontsize=8)
cbar2.ax.tick_params(labelsize=7)

legend_patches2 = [mpatches.Patch(color=c, label=g)
                   for g, c in GROUP_COLORS.items()]
ax_heat.legend(handles=legend_patches2, loc="upper right", fontsize=7,
               framealpha=0.9, title="Taxonomy", title_fontsize=7)

plt.suptitle("Hierarchical Clustering of Games by Model-Performance Correlation",
             fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()
p2 = ROOT / "paper" / "correlation" / "plot_corr_clustered.png"
plt.savefig(p2, dpi=150, bbox_inches="tight")
print(f"Saved to {p2}")
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — within-group vs between-group average correlation
# ════════════════════════════════════════════════════════════════════════════
groups = list(TAXONOMY.keys())
present = {grp: [g for g in gs if g in all_games] for grp, gs in TAXONOMY.items()}

# Per-group within correlation
within_vals = {}
for grp, gs in present.items():
    idx = [all_games.index(g) for g in gs]
    vals = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            v = corr[idx[a], idx[b]]
            if not np.isnan(v):
                vals.append(v)
    within_vals[grp] = vals

# All between-group pairs
between_all = []
group_labels_bt = []
for i, g1 in enumerate(groups):
    for j, g2 in enumerate(groups):
        if j <= i:
            continue
        idx1 = [all_games.index(g) for g in present[g1]]
        idx2 = [all_games.index(g) for g in present[g2]]
        vals = []
        for a in idx1:
            for b in idx2:
                v = corr[a, b]
                if not np.isnan(v):
                    vals.append(v)
        between_all.extend(vals)
        group_labels_bt.append(f"{g1}\nvs\n{g2}")

fig3, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

# Left: violin / box per group for within-group
ax_w = axes[0]
positions = range(len(groups))
parts = ax_w.violinplot(
    [within_vals[g] for g in groups],
    positions=list(positions), widths=0.6, showmedians=True,
)
for pc, grp in zip(parts["bodies"], groups):
    pc.set_facecolor(GROUP_COLORS[grp])
    pc.set_alpha(0.7)
for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
    parts[part].set_color("#333333")
    parts[part].set_linewidth(1.2)
ax_w.axhline(0, color="#aaaaaa", lw=0.8, ls="--")
ax_w.set_xticks(list(positions))
ax_w.set_xticklabels(groups, fontsize=9)
ax_w.set_ylabel("Pearson r", fontsize=9)
ax_w.set_title("Within-group pairwise correlation", fontsize=10, fontweight="bold")
ax_w.set_ylim(-1.05, 1.05)
for i, grp in enumerate(groups):
    m = np.mean(within_vals[grp]) if within_vals[grp] else np.nan
    ax_w.text(i, -0.95, f"μ={m:.2f}", ha="center", fontsize=7.5, color="#333333")

# Right: grouped bar of mean ± std for all cross-group pairs
ax_b = axes[1]
all_cross_groups = [(g1, g2) for i, g1 in enumerate(groups)
                             for j, g2 in enumerate(groups) if j > i]
cross_means, cross_stds, cross_colors, cross_labels = [], [], [], []
for g1, g2 in all_cross_groups:
    idx1 = [all_games.index(g) for g in present[g1]]
    idx2 = [all_games.index(g) for g in present[g2]]
    vals = [corr[a, b] for a in idx1 for b in idx2 if not np.isnan(corr[a, b])]
    cross_means.append(np.mean(vals) if vals else np.nan)
    cross_stds.append(np.std(vals) if vals else 0.0)
    cross_colors.append(GROUP_COLORS[g1])
    cross_labels.append(f"{g1}\nvs {g2}")

x = np.arange(len(cross_means))
bars = ax_b.bar(x, cross_means, yerr=cross_stds, capsize=4,
                color=cross_colors, alpha=0.75, edgecolor="#555555", width=0.6)
ax_b.axhline(0, color="#aaaaaa", lw=0.8, ls="--")
ax_b.set_xticks(x)
ax_b.set_xticklabels(cross_labels, fontsize=7.5)
ax_b.set_ylabel("Mean Pearson r", fontsize=9)
ax_b.set_title("Between-group mean correlation (±std)", fontsize=10, fontweight="bold")
ax_b.set_ylim(-1.05, 1.05)

# Reference line: global within-group mean
all_within = [v for vals in within_vals.values() for v in vals]
global_within_mean = np.mean(all_within)
ax_b.axhline(global_within_mean, color="#222222", lw=1.2, ls=":",
             label=f"Global within-group mean ({global_within_mean:.2f})")
ax_b.legend(fontsize=7.5)

plt.suptitle("Correlation Structure: Within vs Between Taxonomy Groups",
             fontsize=11, fontweight="bold")
plt.tight_layout()
p3 = ROOT / "paper" / "correlation" / "plot_corr_groups.png"
plt.savefig(p3, dpi=150, bbox_inches="tight")
print(f"Saved to {p3}")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — capability-axis analysis
# Tests whether games sharing the same attribute (reward density, visual load,
# tempo, action topology, primary skill) are more correlated than games that
# differ on that attribute.
# ════════════════════════════════════════════════════════════════════════════

# Capability attributes from the taxonomy table in method.tex.
# Primary skill groups are consolidated into 5 categories (no singletons):
#   Shoot & Dodge  — fixed-position shooter, move laterally, dodge projectiles
#                    (air_raid, assault, beam_rider, demon_attack, phoenix, name_this_game)
#   Navigate & Shoot — free 2-D/3-D navigation combined with combat
#                    (riverraid, seaquest, robotank, time_pilot, laser_gates)
#   Track & React  — intercept or time a moving object / opponent
#                    (breakout, tennis, fishing_derby, ice_hockey, boxing)
#   Evasion        — survive by avoiding threats; no primary shooting
#                    (freeway, journey_escape, gopher)
#   Route Planning — strategic pathfinding through a fixed environment
#                    (pacman, qbert)
GAME_ATTRS = {
    #                  primary_skill_group   action_topo  reward   visual  tempo
    "air_raid":       ("Shoot & Dodge",      "H+A",       "Dense",  "Low",  "High"),
    "assault":        ("Shoot & Dodge",      "H+A",       "Dense",  "Low",  "High"),
    "beam_rider":     ("Shoot & Dodge",      "H+A",       "Dense",  "High", "High"),
    "demon_attack":   ("Shoot & Dodge",      "H+A",       "Dense",  "Low",  "High"),
    "name_this_game": ("Shoot & Dodge",      "H+A",       "Dense",  "Low",  "High"),
    "phoenix":        ("Shoot & Dodge",      "H+A",       "Dense",  "Low",  "High"),
    "laser_gates":    ("Navigate & Shoot",   "B+A",       "Dense",  "High", "High"),
    "riverraid":      ("Navigate & Shoot",   "H+A",       "Dense",  "High", "High"),
    "robotank":       ("Navigate & Shoot",   "B+A",       "Dense",  "High", "High"),
    "seaquest":       ("Navigate & Shoot",   "B+A",       "Dense",  "High", "High"),
    "time_pilot":     ("Navigate & Shoot",   "B+A",       "Sparse", "High", "High"),
    "boxing":         ("Track & React",      "B+A",       "Dense",  "Low",  "High"),
    "breakout":       ("Track & React",      "H",         "Dense",  "Low",  "Med"),
    "fishing_derby":  ("Track & React",      "B+A",       "Dense",  "High", "High"),
    "ice_hockey":     ("Track & React",      "B+A",       "Sparse", "Low",  "Med"),
    "tennis":         ("Track & React",      "B",         "Sparse", "Low",  "Med"),
    "freeway":        ("Evasion",            "V",         "Sparse", "High", "High"),
    "gopher":         ("Evasion",            "H+A",       "Dense",  "High", "High"),
    "journey_escape": ("Evasion",            "B",         "Dense",  "High", "High"),
    "pacman":         ("Route Planning",     "B",         "Dense",  "High", "High"),
    "qbert":          ("Route Planning",     "B",         "Dense",  "High", "Low"),
}
ATTR_NAMES = ["Primary Skill", "Action Topology", "Reward Density", "Visual Load", "Tempo"]
ATTR_INDICES = list(range(5))


def same_vs_diff_corr(attr_idx):
    """Return (same_attr_pairs, diff_attr_pairs) correlation lists."""
    same, diff = [], []
    games_with_attr = [g for g in all_games if g in GAME_ATTRS]
    for i in range(len(games_with_attr)):
        for j in range(i + 1, len(games_with_attr)):
            gi, gj = games_with_attr[i], games_with_attr[j]
            ii, jj = all_games.index(gi), all_games.index(gj)
            v = corr[ii, jj]
            if np.isnan(v):
                continue
            if GAME_ATTRS[gi][attr_idx] == GAME_ATTRS[gj][attr_idx]:
                same.append(v)
            else:
                diff.append(v)
    return same, diff


fig4, axes4 = plt.subplots(1, len(ATTR_INDICES), figsize=(14, 4.5), sharey=True)
fig4.suptitle("Does sharing a capability attribute predict higher correlation?",
              fontsize=11, fontweight="bold")

print("\n=== Capability-axis same vs different correlation ===")
for ax, attr_idx in zip(axes4, ATTR_INDICES):
    same, diff = same_vs_diff_corr(attr_idx)
    data = [same, diff]
    vp = ax.violinplot(data, positions=[0, 1], widths=0.55, showmedians=True)
    vp["bodies"][0].set_facecolor("#4393c3")
    vp["bodies"][0].set_alpha(0.7)
    vp["bodies"][1].set_facecolor("#d6604d")
    vp["bodies"][1].set_alpha(0.7)
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        vp[part].set_color("#333333")
        vp[part].set_linewidth(1.2)
    ax.axhline(0, color="#aaaaaa", lw=0.8, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Same", "Diff"], fontsize=9)
    ax.set_title(ATTR_NAMES[attr_idx], fontsize=9, fontweight="bold")
    ax.set_ylim(-1.05, 1.05)
    ms, md = np.mean(same), np.mean(diff)
    ax.text(0, -0.92, f"μ={ms:.2f}", ha="center", fontsize=7.5)
    ax.text(1, -0.92, f"μ={md:.2f}", ha="center", fontsize=7.5)
    # Annotate Δ
    ax.text(0.5, 0.96, f"Δ={ms - md:+.2f}", ha="center", va="top",
            transform=ax.transAxes, fontsize=8,
            color="#1a9850" if ms > md else "#d73027", fontweight="bold")
    print(f"  {ATTR_NAMES[attr_idx]:20s}: same μ={ms:.3f} (n={len(same)}), "
          f"diff μ={md:.3f} (n={len(diff)}), Δ={ms-md:+.3f}")

axes4[0].set_ylabel("Pearson r", fontsize=9)
legend_h = [
    mpatches.Patch(color="#4393c3", alpha=0.7, label="Same attribute"),
    mpatches.Patch(color="#d6604d", alpha=0.7, label="Different attribute"),
]
axes4[-1].legend(handles=legend_h, fontsize=8, loc="upper right")

plt.tight_layout()
p4 = ROOT / "paper" / "correlation" / "plot_corr_capability.png"
plt.savefig(p4, dpi=150, bbox_inches="tight")
print(f"Saved to {p4}")
plt.close()

# ── print cluster vs taxonomy summary ────────────────────────────────────────
n_clusters = 4
cluster_ids = fcluster(Z, n_clusters, criterion="maxclust")
print("\n=== Hierarchical clusters (k=4) vs taxonomy ===")
for k in range(1, n_clusters + 1):
    members = [all_games[i] for i, c in enumerate(cluster_ids) if c == k]
    tax_breakdown = {}
    for g in members:
        grp = GAME_TO_GROUP.get(g, "?")
        tax_breakdown[grp] = tax_breakdown.get(grp, 0) + 1
    print(f"  Cluster {k}: {members}")
    print(f"             taxonomy: {tax_breakdown}")

print(f"\nGlobal within-group mean r = {global_within_mean:.3f}")
for g1, g2 in all_cross_groups:
    idx1 = [all_games.index(g) for g in present[g1]]
    idx2 = [all_games.index(g) for g in present[g2]]
    vals = [corr[a, b] for a in idx1 for b in idx2 if not np.isnan(corr[a, b])]
    print(f"  {g1:8s} vs {g2:8s}: mean r = {np.mean(vals):.3f}")
