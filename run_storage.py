"""Canonical storage helpers for AtariBench runs."""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import math
import re
from pathlib import Path

_HELPER_PROMPT_MODULES = frozenset({"__init__", "common_prompt", "game_clip", "termination"})
MODEL_SUMMARY_FILENAME = "model_summary.json"
FULL_RUN_MODEL_SUMMARY_FILENAME = "model_summary_30s.json"


def _discover_canonical_game_keys() -> frozenset[str]:
    prompts_dir = Path(__file__).resolve().parent / "games" / "prompts"
    return frozenset(
        path.stem
        for path in prompts_dir.glob("*.py")
        if path.stem not in _HELPER_PROMPT_MODULES
    )


CANONICAL_GAME_KEYS = _discover_canonical_game_keys()


def sanitize_model_label(model_name: str) -> str:
    """Return a filesystem-safe label for a model name."""

    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-") or "model"


def uses_canonical_game_storage(game: str) -> bool:
    """Return whether a game uses the canonical per-model storage layout."""

    return game in CANONICAL_GAME_KEYS


def game_root(project_dir: str | Path, game: str) -> Path:
    """Return the canonical run root for one game."""

    return Path(project_dir).resolve() / "runs" / game


def runs_root(project_dir: str | Path) -> Path:
    """Return the root directory for all stored runs."""

    return Path(project_dir).resolve() / "runs"


def runs_batch_root(project_dir: str | Path) -> Path:
    """Return the root for shared batch metadata across games."""

    return runs_root(project_dir) / "_batches"


def game_model_dir(project_dir: str | Path, game: str, model_name: str) -> Path:
    """Return the canonical directory for one model under a game."""

    return game_root(project_dir, game) / sanitize_model_label(model_name)


def game_batch_root(project_dir: str | Path, game: str) -> Path:
    """Return the root for batch metadata related to one game."""

    return game_root(project_dir, game) / "_batches"


def resolve_output_layout(
    project_dir: str | Path,
    game: str,
    model_name: str,
    requested_output_dir: str | Path,
) -> tuple[str | Path, bool]:
    """Resolve the on-disk run layout for the requested game/model."""

    if uses_canonical_game_storage(game):
        return game_model_dir(project_dir, game, model_name), False
    return requested_output_dir, True


def update_game_model_summary(project_dir: str | Path, game: str) -> Path:
    """Recompute one game's per-model summaries for all-success and 30-second runs."""

    root = game_root(project_dir, game)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / MODEL_SUMMARY_FILENAME
    full_run_summary_path = root / FULL_RUN_MODEL_SUMMARY_FILENAME
    lock_path = root / ".model_summary.lock"

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        summary_path.write_text(
            json.dumps(
                _build_game_summary_payload(
                    root=root,
                    game=game,
                    include_run=_is_successful_run,
                    run_filter="all_successful",
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        full_run_summary_path.write_text(
            json.dumps(
                _build_game_summary_payload(
                    root=root,
                    game=game,
                    include_run=_is_full_canonical_run,
                    run_filter="full_30s",
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    update_runs_model_summary(project_dir)
    return summary_path


def update_runs_model_summary(project_dir: str | Path) -> Path:
    """Recompute the flat cross-game summaries for all-success and 30-second runs."""

    root = runs_root(project_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / MODEL_SUMMARY_FILENAME
    full_run_summary_path = root / FULL_RUN_MODEL_SUMMARY_FILENAME
    lock_path = root / ".model_summary.lock"

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        summary_path.write_text(
            json.dumps(
                _build_runs_summary_payload(
                    root=root,
                    per_game_summary_filename=MODEL_SUMMARY_FILENAME,
                    run_filter="all_successful",
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        full_run_summary_path.write_text(
            json.dumps(
                _build_runs_summary_payload(
                    root=root,
                    per_game_summary_filename=FULL_RUN_MODEL_SUMMARY_FILENAME,
                    run_filter="full_30s",
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return summary_path


def _build_game_summary_payload(
    root: Path,
    game: str,
    *,
    include_run,
    run_filter: str,
) -> dict[str, object]:
    models: dict[str, list[dict[str, object]]] = {}

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue

        run_summaries: list[tuple[Path, dict[str, object]]] = []
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            with contextlib.suppress(json.JSONDecodeError):
                run_summaries.append(
                    (
                        run_dir,
                        json.loads(summary_path.read_text(encoding="utf-8")),
                    )
                )

        eligible_runs = [
            (run_dir, summary)
            for run_dir, summary in run_summaries
            if include_run(summary)
        ]
        if not eligible_runs:
            continue

        model_name = _coerce_string(
            eligible_runs[0][1].get("model_name"),
            default=model_dir.name,
        )
        setting_groups: dict[str, list[tuple[Path, dict[str, object]]]] = {}
        for run_dir, summary in eligible_runs:
            setting_key = _build_setting_key(summary)
            setting_groups.setdefault(setting_key, []).append((run_dir, summary))

        models[model_name] = [
            _build_setting_summary(
                model_name=model_name,
                setting_key=setting_key,
                eligible_runs=group_runs,
            )
            for setting_key, group_runs in sorted(setting_groups.items())
        ]

    return {
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "game": game,
        "run_filter": run_filter,
        "models": models,
    }


def _build_runs_summary_payload(
    root: Path,
    *,
    per_game_summary_filename: str,
    run_filter: str,
) -> dict[str, object]:
    entries: list[dict[str, object]] = []

    for game_dir in sorted(root.iterdir()):
        if not game_dir.is_dir() or game_dir.name.startswith("_"):
            continue
        per_game_summary_path = game_dir / per_game_summary_filename
        if not per_game_summary_path.exists():
            continue
        with contextlib.suppress(json.JSONDecodeError):
            payload = json.loads(per_game_summary_path.read_text(encoding="utf-8"))
            models = payload.get("models", {})
            if not isinstance(models, dict):
                continue
            for model_name, model_entries in sorted(models.items()):
                if isinstance(model_entries, list):
                    iterable = model_entries
                elif isinstance(model_entries, dict):
                    iterable = [model_entries]
                else:
                    continue
                for model_summary in iterable:
                    if not isinstance(model_summary, dict):
                        continue
                    entry = {
                        "game": game_dir.name,
                        "model_name": _coerce_string(
                            model_summary.get("model_name"),
                            default=model_name,
                        ),
                    }
                    entry.update(model_summary)
                    entries.append(entry)

    entries.sort(
        key=lambda entry: (
            _coerce_string(entry.get("game"), default=""),
            _coerce_string(entry.get("model_name"), default=""),
            _coerce_string(entry.get("setting_key"), default=""),
        )
    )

    return {
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_filter": run_filter,
        "entries": entries,
    }


def _is_successful_run(summary: dict[str, object]) -> bool:
    return _coerce_string(summary.get("stop_reason"), default="") == "frame_budget"


def _is_full_canonical_run(summary: dict[str, object]) -> bool:
    duration_seconds = summary.get("duration_seconds")
    stop_reason = _coerce_string(summary.get("stop_reason"), default="")
    frame_count = _coerce_float(summary.get("frame_count"))

    if duration_seconds is not None:
        return int(duration_seconds) == 30 and stop_reason == "frame_budget"
    return stop_reason == "frame_budget" and frame_count >= 901.0


def _coerce_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _coerce_string(value: object, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _build_setting_summary(
    model_name: str,
    setting_key: str,
    eligible_runs: list[tuple[Path, dict[str, object]]],
) -> dict[str, object]:
    latest_run_dir, latest_summary = max(eligible_runs, key=lambda item: item[0].name)

    total_rewards = [_coerce_float(summary.get("total_reward")) for _, summary in eligible_runs]
    total_lost_lives = [
        _coerce_float(summary.get("total_lost_lives")) for _, summary in eligible_runs
    ]
    turn_counts = [_coerce_float(summary.get("turn_count")) for _, summary in eligible_runs]
    frame_counts = [_coerce_float(summary.get("frame_count")) for _, summary in eligible_runs]
    input_tokens = [_coerce_float(summary.get("input_tokens")) for _, summary in eligible_runs]
    output_tokens = [_coerce_float(summary.get("output_tokens")) for _, summary in eligible_runs]
    total_tokens = [_coerce_float(summary.get("total_tokens")) for _, summary in eligible_runs]
    thinking_tokens = [_coerce_float(summary.get("thinking_tokens")) for _, summary in eligible_runs]
    cached_input_tokens = [
        _coerce_float(summary.get("cached_input_tokens")) for _, summary in eligible_runs
    ]
    token_usage_reported_turns = [
        _coerce_float(summary.get("token_usage_reported_turns")) for _, summary in eligible_runs
    ]
    token_usage_missing_turns = [
        _coerce_float(summary.get("token_usage_missing_turns")) for _, summary in eligible_runs
    ]

    run_count = len(eligible_runs)
    return {
        "model_name": model_name,
        "setting_key": setting_key,
        "run_count": run_count,
        "avg_total_reward": sum(total_rewards) / run_count,
        "stderr_total_reward": _standard_error(total_rewards),
        "avg_total_lost_lives": sum(total_lost_lives) / run_count,
        "stderr_total_lost_lives": _standard_error(total_lost_lives),
        "avg_turn_count": sum(turn_counts) / run_count,
        "avg_frame_count": sum(frame_counts) / run_count,
        "avg_input_tokens": sum(input_tokens) / run_count,
        "avg_output_tokens": sum(output_tokens) / run_count,
        "avg_total_tokens": sum(total_tokens) / run_count,
        "avg_thinking_tokens": sum(thinking_tokens) / run_count,
        "avg_cached_input_tokens": sum(cached_input_tokens) / run_count,
        "avg_token_usage_reported_turns": sum(token_usage_reported_turns) / run_count,
        "avg_token_usage_missing_turns": sum(token_usage_missing_turns) / run_count,
        "best_total_reward": max(total_rewards),
        "worst_total_reward": min(total_rewards),
        "latest_run_dir": str(latest_run_dir.resolve()),
        "latest_timestamp": latest_run_dir.name,
        "latest_total_reward": _coerce_float(latest_summary.get("total_reward")),
        "latest_input_tokens": _coerce_float(latest_summary.get("input_tokens")),
        "latest_output_tokens": _coerce_float(latest_summary.get("output_tokens")),
        "latest_total_tokens": _coerce_float(latest_summary.get("total_tokens")),
        "latest_thinking_tokens": _coerce_float(latest_summary.get("thinking_tokens")),
        "latest_cached_input_tokens": _coerce_float(latest_summary.get("cached_input_tokens")),
        "latest_token_usage_reported_turns": _coerce_float(
            latest_summary.get("token_usage_reported_turns")
        ),
        "latest_token_usage_missing_turns": _coerce_float(
            latest_summary.get("token_usage_missing_turns")
        ),
        "thinking_mode": _coerce_string(latest_summary.get("thinking_mode"), default="default"),
        "prompt_mode": _coerce_string(
            latest_summary.get("prompt_mode"),
            default="structured_history",
        ),
        "context_cache": bool(latest_summary.get("context_cache", False)),
        "frames_per_action": _extract_frames_per_action(latest_summary),
        "thinking_level": latest_summary.get("thinking_level"),
        "thinking_budget": latest_summary.get("thinking_budget"),
        "history_clips": _extract_history_clips(latest_summary),
        "non_zero_reward_clips": _extract_non_zero_reward_clips(latest_summary),
    }


def _build_setting_key(summary: dict[str, object]) -> str:
    return "|".join(
        (
            f"prompt_mode={_coerce_string(summary.get('prompt_mode'), default='structured_history')}",
            f"context_cache={_stringify_setting_value(bool(summary.get('context_cache', False)))}",
            f"thinking_mode={_coerce_string(summary.get('thinking_mode'), default='default')}",
            f"frames_per_action={_extract_frames_per_action(summary)}",
            f"thinking_level={_stringify_setting_value(summary.get('thinking_level'))}",
            f"thinking_budget={_stringify_setting_value(summary.get('thinking_budget'))}",
            f"history_clips={_extract_history_clips(summary)}",
            f"non_zero_reward_clips={_extract_non_zero_reward_clips(summary)}",
        )
    )


def _stringify_setting_value(value: object) -> str:
    if value is None:
        return "null"
    return str(value)


def _extract_history_clips(summary: dict[str, object]) -> int:
    if _coerce_string(summary.get("prompt_mode"), default="structured_history") == "append_only":
        return -1
    value = summary.get("history_clips")
    if value is None:
        return 3
    return int(value)


def _extract_non_zero_reward_clips(summary: dict[str, object]) -> int:
    if _coerce_string(summary.get("prompt_mode"), default="structured_history") == "append_only":
        return -1
    value = summary.get("non_zero_reward_clips")
    if value is not None:
        return int(value)
    history_clips = summary.get("history_clips")
    if history_clips is not None:
        return int(history_clips)
    return 3


def _extract_frames_per_action(summary: dict[str, object]) -> int:
    value = summary.get("frames_per_action")
    if value is None:
        return 3
    return int(value)


def _standard_error(values: list[float]) -> float:
    count = len(values)
    if count <= 1:
        return 0.0

    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return math.sqrt(variance) / math.sqrt(count)
