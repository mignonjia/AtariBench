import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
APPEND_ONLY_SUMMARY_PATH = ROOT / "runs" / "model_summary_30s_gemini_append_only.json"

MODEL_ALIASES = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "openai:gpt-5.4": "gpt-5.4",
    "openai:gpt-5.4-mini": "gpt-5.4-mini",
    "anthropic:claude-opus-4-6": "claude-opus-4-6",
    "anthropic:claude-sonnet-4-6": "claude-sonnet-4-6",
    "deepseek-ai/deepseek-v3.1": "deepseek-ai/deepseek-v3.1",
    "zai-org/glm-5.1": "zai-org/glm-5.1",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "google:gemini-3-flash-preview": "gemini-3-flash-preview",
    "google:gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "random": "random",
}

MODEL_DISPLAY_NAMES = {
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "deepseek-ai/deepseek-v3.1": "DeepSeek V3.1",
    "zai-org/glm-5.1": "GLM 5.1",
    "random": "Random",
}

MODEL_CANONICAL_ORDER = [
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gpt-5.4",
    "gpt-5.4-mini",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "deepseek-ai/deepseek-v3.1",
    "zai-org/glm-5.1",
    "random",
]
PROMPT_MODE_ORDER = {"structured_history": 0, "append_only": 1}
THINKING_MODE_ORDER = {"default": 0, "high": 1, "low": 2, "off": 3, "none": 3}
THINKING_LEVEL_ORDER = {"high": 0, "low": 1, "none": 2}

EXCLUDE_ROW_KEYS = set()
EXCLUDE_MODELS = set()
EXCLUDE_GAMES = {"gopher"}

# Games must have data for all REFERENCE_ROW_KEYS to be included in the plot
REFERENCE_ROW_KEYS = [
    "gemini-3-flash-preview|structured_history|default|high",
    "gemini-3-flash-preview|structured_history|default|low",
    "gemini-2.5-flash|structured_history|off|none",
    "gpt-5.4-mini|structured_history|none|none",
]
APPEND_ONLY_COMPARISON_STRUCTURED_KEYS = [
    "gemini-3.1-pro-preview|structured_history|default|high",
    "gemini-3.1-pro-preview|structured_history|default|low",
    "gemini-3-flash-preview|structured_history|default|low",
    "gemini-3-flash-preview|structured_history|default|high",
]


def find_json_files():
    return [p for p in (ROOT / "runs").rglob("*.json") if "30s" in p.name]


def canonical_model_name(raw_name):
    return MODEL_ALIASES.get(raw_name)


def make_row_key(entry):
    canonical = canonical_model_name(entry.get("model_name", ""))
    if canonical is None:
        return None
    prompt_mode = entry.get("prompt_mode") or "structured_history"
    thinking_mode = entry.get("thinking_mode") or "none"
    thinking_level = entry.get("thinking_level")
    tl = thinking_level if thinking_level and thinking_level != "none" else "none"
    return f"{canonical}|{prompt_mode}|{thinking_mode}|{tl}"


def make_label(row_key):
    canonical, prompt_mode, thinking_mode, thinking_level = row_key.split("|")
    base = MODEL_DISPLAY_NAMES.get(canonical, canonical)
    parts = []
    if thinking_level not in ("none", ""):
        parts.append(thinking_level.capitalize())
    elif thinking_mode in ("off", "none"):
        parts.append("None")
    if thinking_mode not in ("default", "off", "none", ""):
        parts.append(f"{thinking_mode.capitalize()} Mode")
    if prompt_mode == "append_only":
        parts.append("AO")
    return f"{base} ({', '.join(parts)})" if parts else base


def row_sort_key(row_key):
    canonical, prompt_mode, thinking_mode, thinking_level = row_key.split("|")
    ci = MODEL_CANONICAL_ORDER.index(canonical) if canonical in MODEL_CANONICAL_ORDER else 999
    pi = PROMPT_MODE_ORDER.get(prompt_mode, 99)
    mi = THINKING_MODE_ORDER.get(thinking_mode, 99)
    ti = THINKING_LEVEL_ORDER.get(thinking_level, 99)
    return (ci, pi, mi, ti)


def load_entries(paths):
    entries = []
    for path in paths:
        with path.open() as f:
            payload = json.load(f)
        entries.extend(payload.get("entries", []))
    return entries


def collect_best_rewards(entries):
    rewards = {}
    run_counts = {}
    for entry in entries:
        row_key = make_row_key(entry)
        game = entry.get("game")
        value = entry.get("avg_total_reward")
        if row_key is None or row_key.split("|")[0] in EXCLUDE_MODELS:
            continue
        if game is None or game in EXCLUDE_GAMES or value is None:
            continue
        key = (row_key, game)
        if key not in rewards or value > rewards[key]:
            rewards[key] = value
            run_counts[key] = entry.get("run_count")
    return rewards, run_counts


def add_row_average_column(ax, values, label, fmt, x_pos, header_y):
    ax.text(
        x_pos, header_y, label,
        ha="center", va="bottom", fontsize=8, fontweight="bold", clip_on=False,
    )
    for i, value in enumerate(values):
        text = "NA" if np.isnan(value) else format(value, fmt)
        ax.text(
            x_pos, i, text,
            ha="center", va="center", fontsize=7, color="#333333",
            fontweight="bold", clip_on=False,
        )
    ax.set_xlim(-0.5, x_pos + 0.55)


def build_sorted_matrix(candidate_keys, reward, games):
    """Build reward matrix sorted by avg normalized score (descending = best first)."""
    n_prov = len(candidate_keys)
    n_games = len(games)
    prov_matrix = np.full((n_prov, n_games), np.nan)
    for i, model in enumerate(candidate_keys):
        for j, game in enumerate(games):
            if (model, game) in reward:
                prov_matrix[i, j] = reward[(model, game)]

    # normalize per game: shift negatives, scale to [0, 1]
    norm = np.full_like(prov_matrix, np.nan)
    for j in range(n_games):
        col = prov_matrix[:, j]
        valid = ~np.isnan(col)
        if not valid.any():
            continue
        col_min = col[valid].min()
        shifted = col + max(0.0, -col_min)
        col_max = shifted[valid].max()
        if col_max == 0:
            col_max = 1.0
        norm[valid, j] = np.clip(shifted[valid] / col_max, 0.0, 1.0)

    prov_avg_norm = np.nanmean(norm, axis=1)
    # sort descending by avg norm (higher = better); break ties with canonical order
    sorted_indices = sorted(
        range(n_prov),
        key=lambda i: (-prov_avg_norm[i], row_sort_key(candidate_keys[i])),
    )
    return [candidate_keys[i] for i in sorted_indices]


def compute_rank_matrix(matrix):
    n_models, n_games = matrix.shape
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
    return rank_matrix


def generate_plots(row_keys, reward, run_counts, games, game_labels, file_suffix, title_suffix=""):
    n_models = len(row_keys)
    n_games = len(games)
    model_labels = [make_label(rk) for rk in row_keys]
    summary_col_x = n_games + 0.55
    summary_header_y = -0.9

    matrix = np.full((n_models, n_games), np.nan)
    count_matrix = np.full((n_models, n_games), np.nan)
    for i, model in enumerate(row_keys):
        for j, game in enumerate(games):
            if (model, game) in reward:
                matrix[i, j] = reward[(model, game)]
            count = run_counts.get((model, game))
            if count is not None:
                count_matrix[i, j] = count

    rank_matrix = compute_rank_matrix(matrix)
    avg_rank_by_model = np.nanmean(rank_matrix, axis=1)

    cmap_r = mcolors.LinearSegmentedColormap.from_list("gr", ["#1a9850", "#fee08b", "#d73027"])
    cmap_raw = mcolors.LinearSegmentedColormap.from_list("rg", ["#d73027", "#fee08b", "#1a9850"])
    grey_color = "#cccccc"
    fig_width = max(8, n_games * 0.42)
    fig_height = max(2.6, n_models * 0.38 + 1.2)
    title_tag = f" ({title_suffix})" if title_suffix else ""

    # ── rank heatmap ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor(grey_color)
    im = ax.imshow(rank_matrix, aspect="auto", cmap=cmap_r,
                   vmin=1, vmax=n_models, interpolation="nearest")
    for i in range(n_models):
        for j in range(n_games):
            if np.isnan(rank_matrix[i, j]):
                ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, color=grey_color, zorder=2))
    for i in range(n_models):
        for j in range(n_games):
            rv = rank_matrix[i, j]
            if np.isnan(rv):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=8,
                        color="#888888", fontstyle="italic", zorder=3)
            else:
                norm_val = (rv - 1) / max(n_models - 1, 1)
                text_color = "black" if 0.25 < norm_val < 0.75 else "white"
                ax.text(j, i, str(int(rv)), ha="center", va="center", fontsize=7,
                        color=text_color, fontweight="bold", zorder=3)
    ax.set_xticks(range(n_games))
    ax.set_xticklabels(game_labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_labels, fontsize=8)
    ax.set_title(f"Individual Benchmark Scores by Model{title_tag}",
                 fontsize=10, fontweight="bold", pad=8)
    add_row_average_column(ax, avg_rank_by_model, "Avg Rank", ".2f", summary_col_x, summary_header_y)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Rank (1 = best)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks(list(range(1, n_models + 1)))
    cbar.set_ticklabels([str(r) for r in range(1, n_models + 1)])
    plt.tight_layout()
    p = ROOT / "paper" / "heatmap" / f"plot_heatmap_rank{file_suffix}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved to {p}")
    plt.close()

    # ── raw score heatmap ─────────────────────────────────────────────────────
    all_nan_cols = np.all(np.isnan(matrix), axis=0)
    col_min = np.zeros(n_games)
    col_max = np.ones(n_games)
    for j in range(n_games):
        if not all_nan_cols[j]:
            col_min[j] = np.nanmin(matrix[:, j])
            col_max[j] = max(np.nanmax(matrix[:, j] + max(0, -col_min[j])), 1e-9)
    shift = np.where(col_min < 0, -col_min, 0.0)
    col_max[col_max == 0] = 1
    norm_matrix = np.full((n_models, n_games), np.nan)
    for i in range(n_models):
        valid = ~np.isnan(matrix[i, :])
        norm_matrix[i, valid] = (matrix[i, valid] + shift[valid]) / col_max[valid]
    norm_matrix = np.clip(norm_matrix, 0, 1)
    avg_norm_by_model = np.nanmean(norm_matrix, axis=1)

    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
    ax2.set_facecolor(grey_color)
    im2 = ax2.imshow(norm_matrix, aspect="auto", cmap=cmap_raw,
                     vmin=0, vmax=1, interpolation="nearest")
    for i in range(n_models):
        for j in range(n_games):
            if np.isnan(norm_matrix[i, j]):
                ax2.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, color=grey_color, zorder=2))
                ax2.text(j, i, "NA", ha="center", va="center", fontsize=8,
                         color="#888888", fontstyle="italic", zorder=3)
            else:
                nv = norm_matrix[i, j]
                text_color = "black" if 0.35 < nv < 0.85 else "white"
                ax2.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                         fontsize=6.5, color=text_color, fontweight="bold", zorder=3)
    ax2.set_xticks(range(n_games))
    ax2.set_xticklabels(game_labels, rotation=40, ha="right", fontsize=7)
    ax2.set_yticks(range(n_models))
    ax2.set_yticklabels(model_labels, fontsize=8)
    ax2.set_title(f"Individual Benchmark Scores by Model (Raw){title_tag}",
                  fontsize=10, fontweight="bold", pad=8)
    cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.02, pad=0.01)
    cbar2.set_label("Normalized Score (per game)", fontsize=7)
    cbar2.ax.tick_params(labelsize=7)
    cbar2.set_ticks([0, 0.5, 1])
    cbar2.set_ticklabels(["Low", "Mid", "High"])
    plt.tight_layout()
    p2 = ROOT / "paper" / "heatmap" / f"plot_heatmap_raw{file_suffix}.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"Saved to {p2}")
    plt.close()

    # ── run-count heatmap ────────────────────────────────────────────────────
    fig_count, ax_count = plt.subplots(figsize=(fig_width, fig_height))
    ax_count.set_facecolor(grey_color)
    cmap_count = mcolors.LinearSegmentedColormap.from_list(
        "run_counts", ["#d73027", "#fee08b", "#1a9850"]
    )
    count_vmax = max(9.0, np.nanmax(count_matrix) if not np.all(np.isnan(count_matrix)) else 9.0)
    im_count = ax_count.imshow(
        count_matrix, aspect="auto", cmap=cmap_count,
        vmin=0, vmax=count_vmax, interpolation="nearest",
    )
    for i in range(n_models):
        for j in range(n_games):
            cv = count_matrix[i, j]
            if np.isnan(cv):
                ax_count.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, color=grey_color, zorder=2))
                ax_count.text(j, i, "NA", ha="center", va="center", fontsize=8,
                              color="#888888", fontstyle="italic", zorder=3)
            else:
                text_color = "white" if cv < 5 or cv > count_vmax * 0.75 else "black"
                ax_count.text(j, i, str(int(cv)), ha="center", va="center",
                              fontsize=6.5, color=text_color, fontweight="bold", zorder=3)
                if cv < 9:
                    ax_count.add_patch(
                        mpatches.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, edgecolor="#111111", linewidth=1.2, zorder=4,
                        )
                    )
    ax_count.set_xticks(range(n_games))
    ax_count.set_xticklabels(game_labels, rotation=40, ha="right", fontsize=7)
    ax_count.set_yticks(range(n_models))
    ax_count.set_yticklabels(model_labels, fontsize=8)
    ax_count.set_title(f"Run Counts by Model and Game{title_tag}",
                       fontsize=10, fontweight="bold", pad=8)
    cbar_count = fig_count.colorbar(im_count, ax=ax_count, fraction=0.02, pad=0.01)
    cbar_count.set_label("Runs", fontsize=7)
    cbar_count.ax.tick_params(labelsize=7)
    cbar_count.set_ticks([0, 9, count_vmax])
    cbar_count.set_ticklabels(["0", "9", str(int(count_vmax))])
    plt.tight_layout()
    p_count = ROOT / "paper" / "heatmap" / f"plot_heatmap_runs{file_suffix}.png"
    plt.savefig(p_count, dpi=150, bbox_inches="tight")
    print(f"Saved to {p_count}")
    plt.close()

    # ── normalized score heatmap ──────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height))
    ax3.set_facecolor(grey_color)
    im3 = ax3.imshow(norm_matrix, aspect="auto", cmap=cmap_raw,
                     vmin=0, vmax=1, interpolation="nearest")
    for i in range(n_models):
        for j in range(n_games):
            if np.isnan(norm_matrix[i, j]):
                ax3.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, color=grey_color, zorder=2))
                ax3.text(j, i, "NA", ha="center", va="center", fontsize=8,
                         color="#888888", fontstyle="italic", zorder=3)
            else:
                nv = norm_matrix[i, j]
                text_color = "black" if 0.35 < nv < 0.85 else "white"
                ax3.text(j, i, f"{nv:.2f}", ha="center", va="center",
                         fontsize=6.5, color=text_color, fontweight="bold", zorder=3)
    ax3.set_xticks(range(n_games))
    ax3.set_xticklabels(game_labels, rotation=40, ha="right", fontsize=7)
    ax3.set_yticks(range(n_models))
    ax3.set_yticklabels(model_labels, fontsize=8)
    ax3.set_title(f"Individual Benchmark Scores by Model (Normalized){title_tag}",
                  fontsize=10, fontweight="bold", pad=8)
    add_row_average_column(ax3, avg_norm_by_model, "Avg Norm", ".2f", summary_col_x, summary_header_y)
    cbar3 = fig3.colorbar(im3, ax=ax3, fraction=0.02, pad=0.01)
    cbar3.set_label("Normalized Score (per game)", fontsize=7)
    cbar3.ax.tick_params(labelsize=7)
    cbar3.set_ticks([0, 0.5, 1])
    cbar3.set_ticklabels(["Low", "Mid", "High"])
    plt.tight_layout()
    p3 = ROOT / "paper" / "heatmap" / f"plot_heatmap_norm{file_suffix}.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    print(f"Saved to {p3}")
    plt.close()


# ── data loading ─────────────────────────────────────────────────────────────
json_paths = find_json_files()
print(f"Found {len(json_paths)} JSON files with '30s' in name")
entries = load_entries(json_paths)
reward, run_counts = collect_best_rewards(entries)
ao_entries = load_entries([APPEND_ONLY_SUMMARY_PATH])
ao_reward, ao_run_counts = collect_best_rewards(ao_entries)
print(f"Append-only data source: {APPEND_ONLY_SUMMARY_PATH}")

all_games = sorted(
    {game for _, game in reward.keys()},
    key=lambda g: g.replace("_", " ").title(),
)
games = [g for g in all_games if all((rk, g) in reward for rk in REFERENCE_ROW_KEYS)]
game_labels = [game.replace("_", " ").title() for game in games]

all_row_keys = sorted(
    {rk for rk, _ in reward.keys() if rk not in EXCLUDE_ROW_KEYS},
    key=row_sort_key,
)
sh_keys = [rk for rk in all_row_keys if rk.split("|")[1] == "structured_history"]
ao_keys = sorted(
    {rk for rk, _ in ao_reward.keys() if rk not in EXCLUDE_ROW_KEYS and rk.split("|")[1] == "append_only"},
    key=row_sort_key,
)
ao_comparison_reward = dict(ao_reward)
ao_comparison_run_counts = dict(ao_run_counts)
for rk in APPEND_ONLY_COMPARISON_STRUCTURED_KEYS:
    for game in games:
        key = (rk, game)
        if key in reward:
            ao_comparison_reward[key] = reward[key]
        if key in run_counts:
            ao_comparison_run_counts[key] = run_counts[key]
ao_comparison_keys = [
    rk for rk in APPEND_ONLY_COMPARISON_STRUCTURED_KEYS + ao_keys
    if any((rk, game) in ao_comparison_reward for game in games)
]

# Main figures: structured_history only, rows ordered by avg rank
sh_sorted = build_sorted_matrix(sh_keys, reward, games)
generate_plots(sh_sorted, reward, run_counts, games, game_labels, file_suffix="", title_suffix="")

# Separate figures: append_only plus matching structured-history Gemini rows.
ao_sorted = build_sorted_matrix(ao_comparison_keys, ao_comparison_reward, games)
generate_plots(
    ao_sorted,
    ao_comparison_reward,
    ao_comparison_run_counts,
    games,
    game_labels,
    file_suffix="_ao",
    title_suffix="Append-Only vs Structured History",
)

# ── breakout append-only bar chart ───────────────────────────────────────────
breakout_scores = [
    (make_label(rk), ao_reward.get((rk, "breakout"), float("nan")))
    for rk in ao_keys
]
breakout_scores = [(lbl, v) for lbl, v in breakout_scores if not np.isnan(v)]
breakout_scores.sort(key=lambda x: x[1], reverse=True)

if breakout_scores:
    labels_br, values_br = zip(*breakout_scores)
    fig_br, ax_br = plt.subplots(figsize=(6, max(2.5, len(labels_br) * 0.45 + 1.0)))
    colors = plt.cm.RdYlGn(np.linspace(0.85, 0.15, len(values_br)))
    bars = ax_br.barh(range(len(labels_br)), values_br, color=colors, edgecolor="white", height=0.6)
    ax_br.set_yticks(range(len(labels_br)))
    ax_br.set_yticklabels(labels_br[::-1] if False else labels_br, fontsize=8)
    ax_br.invert_yaxis()
    ax_br.set_xlabel("Avg Total Reward", fontsize=8)
    ax_br.set_title("Breakout — Append-Only Settings", fontsize=10, fontweight="bold")
    for bar, val in zip(bars, values_br):
        ax_br.text(bar.get_width() + max(values_br) * 0.01, bar.get_y() + bar.get_height() / 2,
                   f"{val:.1f}", va="center", fontsize=7)
    ax_br.set_xlim(0, max(values_br) * 1.18)
    ax_br.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p_br = ROOT / "paper" / "heatmap" / "plot_heatmap_breakout_ao.png"
    plt.savefig(p_br, dpi=150, bbox_inches="tight")
    print(f"Saved to {p_br}")
    plt.close()
