"""
Per-(model, game) action analysis:
  1. Average planning length  — actions planned per turn
  2. Attack percentage        — fraction of actions containing fire/shoot/punch/bonk

Also correlates both metrics with avg reward to see how strategy relates to score.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]

# ── config ───────────────────────────────────────────────────────────────────
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
EXCLUDE_ROW_KEYS = {
    "gemini-3.1-pro-preview|structured_history|default|high",
    "gemini-3.1-pro-preview|append_only|default|high",
}
EXCLUDE_GAMES = {"gopher"}

ATTACK_KEYWORDS = ("fire", "shoot", "punch", "bonk")

ALL_GAMES = [
    "air_raid", "assault", "beam_rider", "boxing", "breakout", "demon_attack",
    "fishing_derby", "freeway", "ice_hockey", "journey_escape",
    "laser_gates", "name_this_game", "pacman", "phoenix", "qbert",
    "riverraid", "robotank", "seaquest", "tennis", "time_pilot",
]


def make_row_key(summary: dict) -> str | None:
    raw = summary.get("model_name", "")
    canonical = MODEL_ALIASES.get(raw)
    if canonical is None:
        return None
    pm = summary.get("prompt_mode") or "structured_history"
    tm = summary.get("thinking_mode") or "none"
    tl = summary.get("thinking_level")
    tl = tl if tl and tl != "none" else "none"
    rk = f"{canonical}|{pm}|{tm}|{tl}"
    return None if rk in EXCLUDE_ROW_KEYS else rk


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


def infer_summary_from_path(run_dir: Path, game: str, model_name: str) -> dict:
    thinking_mode = "none" if model_name == "gpt-5.4-mini" else "off"
    return {
        "game": game,
        "model_name": model_name,
        "prompt_mode": "structured_history",
        "thinking_mode": thinking_mode,
        "thinking_level": None,
        "stop_reason": "frame_budget",
    }


def is_attack(action: str) -> bool:
    a = action.lower()
    return any(kw in a for kw in ATTACK_KEYWORDS)


# ── scan run directories ─────────────────────────────────────────────────────
# acc[(rk, game)] = {"plan_lengths": [...], "attack_fracs": [...], "rewards": [...]}
acc: dict[tuple, dict] = {}

for game in ALL_GAMES:
    if game in EXCLUDE_GAMES:
        continue
    game_dir = ROOT / "runs" / game
    if not game_dir.exists():
        continue
    for model_dir in game_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
            turns_path   = run_dir / "turns.jsonl"
            if not turns_path.exists():
                continue

            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                with summary_path.open() as f:
                    summary = json.load(f)
            else:
                summary = infer_summary_from_path(run_dir, game, model_dir.name)
            rk = make_row_key(summary)
            if rk is None:
                continue
            if summary.get("stop_reason") != "frame_budget":
                continue  # skip incomplete runs

            key = (rk, game)
            if key not in acc:
                acc[key] = {"plan_lengths": [], "attack_fracs": [], "rewards": []}

            # read turns
            plan_lengths, attack_fracs = [], []
            total_reward = 0.0
            with turns_path.open() as f:
                for line in f:
                    turn = json.loads(line)
                    total_reward += float(turn.get("reward_delta") or 0.0)
                    actions = turn.get("planned_action_strings") or []
                    if not actions:
                        continue
                    plan_lengths.append(len(actions))
                    n_attack = sum(1 for a in actions if is_attack(a))
                    attack_fracs.append(n_attack / len(actions))

            if plan_lengths:
                if summary_path.exists():
                    total_reward = float(summary.get("total_reward", total_reward))
                acc[key]["rewards"].append(total_reward)
                acc[key]["plan_lengths"].append(np.mean(plan_lengths))
                acc[key]["attack_fracs"].append(np.mean(attack_fracs))

# ── build matrices ────────────────────────────────────────────────────────────
sh_keys   = sorted({rk for rk, _ in acc.keys()}, key=row_sort_key)
all_games = [g for g in ALL_GAMES if g not in EXCLUDE_GAMES and any((rk, g) in acc for rk in sh_keys)]
n_models, n_games = len(sh_keys), len(all_games)
model_labels = [make_label(rk) for rk in sh_keys]
game_labels  = [g.replace("_", " ").title() for g in all_games]

if n_models == 0 or n_games == 0:
    raise RuntimeError("No action data found. Need local turns.jsonl files to run action analysis.")

plan_matrix   = np.full((n_models, n_games), np.nan)
attack_matrix = np.full((n_models, n_games), np.nan)
reward_matrix = np.full((n_models, n_games), np.nan)

for i, rk in enumerate(sh_keys):
    for j, game in enumerate(all_games):
        key = (rk, game)
        if key not in acc or not acc[key]["plan_lengths"]:
            continue
        plan_matrix[i, j]   = np.mean(acc[key]["plan_lengths"])
        attack_matrix[i, j] = np.mean(acc[key]["attack_fracs"]) * 100  # percent
        reward_matrix[i, j] = np.mean(acc[key]["rewards"])

# rank models within each game (1 = best); used instead of normalized reward
# so game-level score scales don't confound the analysis
rank_matrix = np.full_like(reward_matrix, np.nan)
for j in range(n_games):
    col = reward_matrix[:, j]
    valid_idx = np.where(~np.isnan(col))[0]
    if len(valid_idx) < 2:
        continue
    scores = col[valid_idx]
    order = np.argsort(-scores)
    i2 = 0
    while i2 < len(order):
        j2 = i2
        while j2 + 1 < len(order) and scores[order[j2+1]] == scores[order[i2]]:
            j2 += 1
        avg_rank = (i2 + j2) / 2.0 + 1
        for kk in range(i2, j2 + 1):
            rank_matrix[valid_idx[order[kk]], j] = avg_rank
        i2 = j2 + 1

print("Models:", model_labels)
print("Games with data:", sum(~np.all(np.isnan(plan_matrix), axis=0)))


def draw_heatmap(ax, data, title, fmt, cmap, vmin, vmax, unit=""):
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    for i in range(n_models):
        for j in range(n_games):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "NA", ha="center", va="center",
                        fontsize=5, color="#aaaaaa", fontstyle="italic")
            else:
                nv = (v - vmin) / max(vmax - vmin, 1e-9)
                tc = "white" if nv < 0.35 or nv > 0.75 else "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=5.5, color=tc, fontweight="bold")
    ax.set_xticks(range(n_games))
    ax.set_xticklabels(game_labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_labels, fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    return im


fig_w = max(10, n_games * 0.44)
fig_h = max(3, n_models * 0.52 + 1.2)

# ── Figure 1: planning length heatmap ────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h))
im1 = draw_heatmap(ax1, plan_matrix, "Avg planning length (actions / turn)",
                   "{:.1f}", "YlOrRd",
                   np.nanmin(plan_matrix), np.nanmax(plan_matrix))
cbar1 = fig1.colorbar(im1, ax=ax1, fraction=0.02, pad=0.01)
cbar1.set_label("actions / turn", fontsize=7); cbar1.ax.tick_params(labelsize=7)
plt.tight_layout()
p1 = ROOT / "paper" / "action" / "plot_action_plan_length.png"
plt.savefig(p1, dpi=150, bbox_inches="tight"); print(f"Saved {p1}"); plt.close()

# ── Figure 2: attack percentage heatmap ──────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
im2 = draw_heatmap(ax2, attack_matrix, "Attack action % (fire / shoot / punch / bonk)",
                   "{:.0f}%", "RdYlGn_r", 0, 100)
cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.02, pad=0.01)
cbar2.set_label("% attack actions", fontsize=7); cbar2.ax.tick_params(labelsize=7)
plt.tight_layout()
p2 = ROOT / "paper" / "action" / "plot_action_attack_pct.png"
plt.savefig(p2, dpi=150, bbox_inches="tight"); print(f"Saved {p2}"); plt.close()

# ── Figure 3: planning length vs rank — raw (confounded) + deconfounded ───────
# Two-panel story:
#   Left:  raw data — apparent correlation driven by model identity
#   Right: model-demeaned — flat trend reveals no true within-model signal
COLORS_M = plt.cm.tab10(np.linspace(0, 0.9, n_models))
COLORS_G = plt.cm.tab20(np.linspace(0, 1, n_games))

plan_model_mean = np.nanmean(plan_matrix, axis=1, keepdims=True)
rank_model_mean = np.nanmean(rank_matrix, axis=1, keepdims=True)
plan_dm = plan_matrix - plan_model_mean
rank_dm = rank_matrix - rank_model_mean
attack_model_mean = np.nanmean(attack_matrix, axis=1, keepdims=True)
attack_dm = attack_matrix - attack_model_mean

fig3, (ax_raw, ax_dm) = plt.subplots(1, 2, figsize=(13, 5.5))

def scatter_per_game_trend(ax, xmat, ymat, xlabel, ylabel, title, invert_y=True):
    """Points colored by model; one regression line per game."""
    all_xs, all_ys = [], []

    # draw per-game regression lines first (background)
    for j, game in enumerate(all_games):
        xs, ys = [], []
        for i in range(n_models):
            if np.isnan(xmat[i, j]) or np.isnan(ymat[i, j]):
                continue
            xs.append(xmat[i, j]); ys.append(ymat[i, j])
        if len(xs) >= 3 and np.std(xs) > 0 and np.std(ys) > 0:
            z = np.polyfit(xs, ys, 1)
            xl = np.linspace(min(xs), max(xs), 40)
            r, _ = pearsonr(xs, ys)
            ax.plot(xl, np.polyval(z, xl), color=COLORS_G[j],
                    lw=1.2, alpha=0.5, zorder=2)
            # label only games with |r| > 0.85 to avoid clutter
            if abs(r) > 0.85:
                ax.text(xl[-1], np.polyval(z, xl[-1]),
                        f" {game.replace('_',' ').title()} r={r:.2f}",
                        fontsize=5.5, color=COLORS_G[j], va="center", zorder=5)

    # draw points on top, colored by model
    for i in range(n_models):
        xs, ys = [], []
        for j in range(n_games):
            if np.isnan(xmat[i, j]) or np.isnan(ymat[i, j]):
                continue
            xs.append(xmat[i, j]); ys.append(ymat[i, j])
            all_xs.append(xmat[i, j]); all_ys.append(ymat[i, j])
        if xs:
            ax.scatter(xs, ys, color=COLORS_M[i], s=50, alpha=0.85,
                       label=model_labels[i], edgecolors="white",
                       linewidths=0.5, zorder=3)

    # overall trend line
    if len(all_xs) >= 4:
        z = np.polyfit(all_xs, all_ys, 1)
        xl = np.linspace(min(all_xs), max(all_xs), 100)
        ax.plot(xl, np.polyval(z, xl), color="black", lw=2.2, zorder=6)
        r, p = pearsonr(all_xs, all_ys)
        sig = "p < 0.05" if p < 0.05 else f"p = {p:.2f}  n.s."
        ax.text(0.97, 0.05, f"overall  r = {r:+.2f}\n{sig}",
                ha="right", va="bottom", transform=ax.transAxes,
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888", alpha=0.9))

    if invert_y:
        ax.invert_yaxis()
    ax.axhline(0, color="#dddddd", lw=0.8, ls="--")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

scatter_per_game_trend(
    ax_raw, plan_matrix, rank_matrix,
    "Avg actions / turn",
    "Rank  (1 = best)",
    "Raw: apparent overall correlation\n(confounded by model identity)",
)
scatter_per_game_trend(
    ax_dm, plan_dm, rank_dm,
    "Δ actions / turn  (model mean removed)",
    "Δ rank  (model mean removed)",
    "After removing model effect:\nno overall signal",
)
ax_dm.axvline(0, color="#dddddd", lw=0.8, ls="--")

handles, labels = ax_raw.get_legend_handles_labels()
fig3.legend(handles, labels, fontsize=8, loc="lower center", ncol=3,
            bbox_to_anchor=(0.5, -0.08), framealpha=0.9,
            title="Model (point color)  —  thin lines = per-game regression",
            title_fontsize=8)
fig3.suptitle("Planning length vs rank: spurious overall correlation, no within-model signal",
              fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()
p3 = ROOT / "paper" / "action" / "plot_action_vs_reward.png"
plt.savefig(p3, dpi=150, bbox_inches="tight"); print(f"Saved {p3}"); plt.close()

# ── Figure 4: per-game avg attack % (all models) vs avg reward ───────────────
game_attack_avg = np.nanmean(attack_matrix, axis=0)
game_reward_avg = np.nanmean(rank_matrix, axis=0)

fig4, ax4 = plt.subplots(figsize=(8, 5))
sc = ax4.scatter(game_attack_avg, game_reward_avg, s=80,
                 c=game_attack_avg, cmap="RdYlGn_r", vmin=0, vmax=100,
                 edgecolors="white", linewidths=0.5, zorder=3)
for j, g in enumerate(all_games):
    if not np.isnan(game_attack_avg[j]):
        ax4.annotate(g.replace("_", " ").title(),
                     (game_attack_avg[j], game_reward_avg[j]),
                     fontsize=6.5, textcoords="offset points", xytext=(5, 3))
valid = ~(np.isnan(game_attack_avg) | np.isnan(game_reward_avg))
if valid.sum() >= 3:
    z = np.polyfit(game_attack_avg[valid], game_reward_avg[valid], 1)
    xl = np.linspace(game_attack_avg[valid].min(), game_attack_avg[valid].max(), 50)
    ax4.plot(xl, np.polyval(z, xl), "k--", lw=1.2, alpha=0.5)
    r, p = pearsonr(game_attack_avg[valid], game_reward_avg[valid])
    ax4.set_title(f"Avg attack % vs avg reward per game  (r={r:.2f}, p={p:.3f})",
                  fontsize=10, fontweight="bold")
fig4.colorbar(sc, ax=ax4, fraction=0.03).set_label("Avg attack %", fontsize=7)
ax4.set_xlabel("Avg attack action % (across models)", fontsize=9)
ax4.set_ylabel("Avg rank (lower = better)", fontsize=9)
ax4.invert_yaxis()  # rank 1 = best, so lower y = better → flip so better is up
ax4.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
p4 = ROOT / "paper" / "action" / "plot_attack_vs_reward_pergame.png"
plt.savefig(p4, dpi=150, bbox_inches="tight"); print(f"Saved {p4}"); plt.close()

# ── console summary ───────────────────────────────────────────────────────────
print("\n=== Planning length (avg actions/turn) ===")
print(f"{'Model':30s}", end="")
for g in all_games: print(f"  {g[:8]:8s}", end="")
print()
for i, (rk, ml) in enumerate(zip(sh_keys, model_labels)):
    print(f"{ml:30s}", end="")
    for j in range(n_games):
        v = plan_matrix[i, j]
        print(f"  {'NA':>8s}" if np.isnan(v) else f"  {v:8.1f}", end="")
    print()

print("\n=== Attack % ===")
print(f"{'Model':30s}", end="")
for g in all_games: print(f"  {g[:8]:8s}", end="")
print()
for i, (rk, ml) in enumerate(zip(sh_keys, model_labels)):
    print(f"{ml:30s}", end="")
    for j in range(n_games):
        v = attack_matrix[i, j]
        print(f"  {'NA':>8s}" if np.isnan(v) else f"  {v:7.1f}%", end="")
    print()
