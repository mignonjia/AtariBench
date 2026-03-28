"""Batch runner for launching multiple AtariBench jobs with bounded concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path


def _bootstrap_local_paths() -> None:
    project_dir = Path(__file__).resolve().parent
    candidate = str(project_dir)
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


if __package__ in {None, ""}:
    _bootstrap_local_paths()
    from core.trajectory import apply_minimal_logging_policy
    from games import list_game_keys, resolve_game_selection
    from llm import infer_model_provider, validate_model_thinking_mode
    from run_storage import (
        game_batch_root,
        game_root,
        runs_batch_root,
        sanitize_model_label,
        update_game_model_summary,
        uses_canonical_game_storage,
    )
    from viz import render_run_video
else:
    from .core.trajectory import apply_minimal_logging_policy
    from .games import list_game_keys, resolve_game_selection
    from .llm import infer_model_provider, validate_model_thinking_mode
    from .run_storage import (
        game_batch_root,
        game_root,
        runs_batch_root,
        sanitize_model_label,
        update_game_model_summary,
        uses_canonical_game_storage,
    )
    from .viz import render_run_video

_SUPPORTED_COMPANIES = ("gemini", "openai", "anthropic")
_INTERNAL_REQUEST_ENV = "ATARIBENCH_INTERNAL_RUN_REQUEST"


@dataclasses.dataclass(frozen=True)
class BatchJobSpec:
    """One logical batch job before expansion into individual runs."""

    model_name: str
    run_count: int
    thinking_mode: str
    label: str
    games_label: str = ""
    games: list[str] = dataclasses.field(default_factory=list)
    duration_seconds: int = 30
    max_actions_per_turn: int = 10
    frames_per_action: int = 3
    history_clips: int = 3
    non_zero_reward_clips: int = 3
    prompt_mode: str = "structured_history"
    seed: int | None = None
    seed_start: int | None = None
    minimal_logging: bool = False


@dataclasses.dataclass(frozen=True)
class RunRequest:
    """One concrete run request for the subprocess-backed batch runner."""

    game: str
    job_label: str
    run_index: int
    total_num_runs: int
    model_name: str
    company: str
    thinking_mode: str
    games_label: str
    duration_seconds: int
    max_actions_per_turn: int
    frames_per_action: int
    history_clips: int
    non_zero_reward_clips: int
    prompt_mode: str
    seed: int | None
    output_dir: str
    log_path: str
    minimal_logging: bool = False
    run_label: str | None = None


@dataclasses.dataclass(frozen=True)
class RunResult:
    """Final status for one batch run."""

    game: str
    job_label: str
    run_index: int
    model_name: str
    requested_thinking_mode: str
    final_thinking_mode: str
    success: bool
    return_code: int
    run_dir: str | None
    stop_reason: str | None
    log_path: str
    attempts: int
    summary: dict[str, object] | None
    video_path: str | None
    video_error: str | None
    error_type: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiple AtariBench jobs.")
    parser.add_argument(
        "--common-config",
        required=True,
        help="Path to a YAML file with shared batch/run settings.",
    )
    parser.add_argument(
        "--runs-config",
        "--config",
        dest="runs_config",
        required=True,
        help="Path to a YAML file listing run settings.",
    )
    parser.add_argument(
        "--minimal-logging",
        action="store_true",
        help=(
            "After rendering completes, keep only summary.json, turns.jsonl, and "
            "visualization.mp4 in each run directory."
        ),
    )
    return parser


def parse_job_spec(
    raw_spec: str,
    *,
    games: list[str] | None = None,
    duration_seconds: int = 30,
    max_actions_per_turn: int = 10,
    frames_per_action: int = 3,
    history_clips: int = 3,
    non_zero_reward_clips: int = 3,
    prompt_mode: str = "structured_history",
    seed: int | None = None,
    minimal_logging: bool = False,
    label: str | None = None,
    games_label: str | None = None,
) -> BatchJobSpec:
    parts = [part.strip() for part in raw_spec.split(":")]
    if len(parts) not in {2, 3}:
        raise ValueError(
            f"Invalid job spec '{raw_spec}'. Expected MODEL:COUNT[:THINKING]."
        )

    model_name = parts[0]
    if not model_name:
        raise ValueError(f"Invalid job spec '{raw_spec}': missing model name.")

    try:
        run_count = int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"Invalid job spec '{raw_spec}': COUNT must be an integer."
        ) from exc
    if run_count < 1:
        raise ValueError(f"Invalid job spec '{raw_spec}': COUNT must be >= 1.")

    thinking_mode = parts[2] if len(parts) == 3 else "default"
    validate_model_thinking_mode(model_name, thinking_mode)
    resolved_games = list(games) if games is not None else []
    label = label or sanitize_model_label(model_name)
    return BatchJobSpec(
        model_name=model_name,
        run_count=run_count,
        thinking_mode=thinking_mode,
        label=label,
        games_label=games_label or ",".join(resolved_games),
        games=resolved_games,
        duration_seconds=duration_seconds,
        max_actions_per_turn=max_actions_per_turn,
        frames_per_action=frames_per_action,
        history_clips=history_clips,
        non_zero_reward_clips=non_zero_reward_clips,
        prompt_mode=prompt_mode,
        seed=seed,
        minimal_logging=minimal_logging,
    )


def load_yaml_config(path: str | Path) -> object:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "PyYAML is required to load YAML config files. Install it in the active environment."
        ) from exc

    class _ConfigLoader(yaml.SafeLoader):
        pass

    for first_char, resolvers in list(_ConfigLoader.yaml_implicit_resolvers.items()):
        _ConfigLoader.yaml_implicit_resolvers[first_char] = [
            (tag, regexp)
            for tag, regexp in resolvers
            if tag != "tag:yaml.org,2002:bool"
        ]

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=_ConfigLoader)


def build_jobs_from_config(
    common_settings: dict[str, object] | None,
    setting_entries: list[dict[str, object]],
) -> tuple[dict[str, object], list[BatchJobSpec]]:
    common = dict(common_settings or {})
    max_concurrency_by_company = _normalize_company_concurrency_map(
        _require_config_key(common, "max_concurrency_by_company", context="common")
    )
    game_selections = _normalize_config_game_selections(common.get("games"))
    batch_options = {
        "output_dir": str(
            common.get(
                "output_dir",
                str(runs_batch_root(Path(__file__).resolve().parent)),
            )
        ),
        "max_concurrency_by_company": max_concurrency_by_company,
        "max_retries": int(_require_config_key(common, "max_retries", context="common")),
        "retry_backoff_seconds": float(
            _require_config_key(common, "retry_backoff_seconds", context="common")
        ),
        "render_video_fps": int(_require_config_key(common, "render_video_fps", context="common")),
    }
    common_duration_seconds = int(_require_config_key(common, "duration_seconds", context="common"))
    common_max_actions_per_turn = int(
        _require_config_key(common, "max_actions_per_turn", context="common")
    )
    common_frames_per_action = int(common.get("frames_per_action", 3))
    common_minimal_logging = _coerce_config_bool(
        common.get("minimal_logging", False),
        key="minimal_logging",
        context="common",
    )

    jobs: list[BatchJobSpec] = []
    for index, entry in enumerate(setting_entries, start=1):
        model_name = str(_require_config_key(entry, "model_name", context=f"setting #{index}"))
        thinking_mode = str(_require_config_key(entry, "thinking_mode", context=f"setting #{index}"))
        validate_model_thinking_mode(model_name, thinking_mode)
        prompt_mode = str(_require_config_key(entry, "prompt_mode", context=f"setting #{index}"))
        if prompt_mode not in {"structured_history", "append_only"}:
            raise ValueError(f"Unsupported prompt_mode '{prompt_mode}'.")
        games_value = _require_config_key(entry, "games", context=f"setting #{index}")
        games = resolve_games_value(games_value, game_selections)
        history_clips = -1
        non_zero_reward_clips = -1
        if prompt_mode == "structured_history":
            history_clips = int(
                _require_config_key(entry, "history_clips", context=f"setting #{index}")
            )
            non_zero_reward_clips = int(
                _require_config_key(entry, "non_zero_reward_clips", context=f"setting #{index}")
            )
        label = f"{sanitize_model_label(model_name)}_cfg_{index:03d}"
        jobs.append(
            BatchJobSpec(
                model_name=model_name,
                run_count=int(_require_config_key(entry, "num_runs", context=f"setting #{index}")),
                thinking_mode=thinking_mode,
                label=label,
                games_label=_stringify_games_label(games_value),
                games=games,
                duration_seconds=common_duration_seconds,
                max_actions_per_turn=common_max_actions_per_turn,
                frames_per_action=common_frames_per_action,
                history_clips=history_clips,
                non_zero_reward_clips=non_zero_reward_clips,
                prompt_mode=prompt_mode,
                seed=None if entry.get("seed") is None else int(entry["seed"]),
                seed_start=(
                    None
                    if entry.get("seed_start") is None
                    else int(entry["seed_start"])
                ),
                minimal_logging=_coerce_config_bool(
                    entry.get("minimal_logging", common_minimal_logging),
                    key="minimal_logging",
                    context=f"setting #{index}",
                ),
            )
        )
    return batch_options, jobs


def resolve_games_value(
    raw_games: object,
    config_game_selections: dict[str, list[str]] | None = None,
) -> list[str]:
    if isinstance(raw_games, str):
        raw_values = [raw_games]
    elif isinstance(raw_games, list):
        raw_values = [str(value) for value in raw_games]
    else:
        raise ValueError("games must be a string or list of strings.")

    resolved: list[str] = []
    for value in raw_values:
        for game in _resolve_games_token(value, config_game_selections):
            if game not in resolved:
                resolved.append(game)
    return resolved


def _stringify_games_label(raw_games: object) -> str:
    if isinstance(raw_games, str):
        return raw_games
    if isinstance(raw_games, list):
        return ",".join(str(value) for value in raw_games)
    return str(raw_games)


def _require_config_key(mapping: dict[str, object], key: str, *, context: str) -> object:
    if key not in mapping:
        raise ValueError(f"Missing required key '{key}' in {context}.")
    return mapping[key]


def _normalize_config_game_selections(raw_value: object) -> dict[str, list[str]] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("games in common.yaml must be a mapping of selection -> game list.")

    normalized: dict[str, list[str]] = {}
    for selection_name, selection_values in raw_value.items():
        normalized_name = str(selection_name).strip().lower()
        if isinstance(selection_values, str):
            values = [selection_values]
        elif isinstance(selection_values, list):
            values = [str(value) for value in selection_values]
        else:
            raise ValueError(
                f"games.{selection_name} must be a string or list of strings."
            )
        normalized[normalized_name] = values
    return normalized


def _coerce_config_bool(value: object, *, key: str, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"Invalid boolean value for '{key}' in {context}: {value!r}.")


def _resolve_games_token(
    token: str,
    config_game_selections: dict[str, list[str]] | None,
    _seen: set[str] | None = None,
) -> list[str]:
    normalized_token = token.strip().lower()
    if normalized_token == "all":
        return list_game_keys()

    if _seen is None:
        _seen = set()
    if normalized_token in _seen:
        raise ValueError(f"Cyclic game selection detected for '{token}'.")

    if config_game_selections and normalized_token in config_game_selections:
        _seen.add(normalized_token)
        resolved: list[str] = []
        for value in config_game_selections[normalized_token]:
            for game in _resolve_games_token(value, config_game_selections, _seen):
                if game not in resolved:
                    resolved.append(game)
        _seen.remove(normalized_token)
        return resolved

    return resolve_game_selection(normalized_token)


def _normalize_company_concurrency_map(raw_value: object) -> dict[str, int] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("max_concurrency_by_company must be a mapping of company -> limit.")

    normalized: dict[str, int] = {}
    for company, limit in raw_value.items():
        normalized_company = str(company).strip().lower()
        if normalized_company not in _SUPPORTED_COMPANIES:
            raise ValueError(
                f"Unsupported company '{company}' in max_concurrency_by_company. "
                "Use gemini, openai, or anthropic."
            )
        limit_value = int(limit)
        if limit_value < 1:
            raise ValueError(
                f"Concurrency limit for company '{company}' must be >= 1."
            )
        normalized[normalized_company] = limit_value
    return normalized


def expand_run_requests(
    jobs: list[BatchJobSpec],
    project_dir: str | Path,
    base_output_dir: str | Path,
    log_dir: str | Path,
    batch_timestamp: str | None = None,
) -> list[RunRequest]:
    requests: list[RunRequest] = []
    project_dir = Path(project_dir)
    base_output_dir = Path(base_output_dir)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        for game in job.games:
            for run_index in range(1, job.run_count + 1):
                run_slug = f"run_{run_index:03d}"
                if uses_canonical_game_storage(game):
                    output_dir = game_root(project_dir, game) / sanitize_model_label(job.model_name)
                else:
                    output_dir = base_output_dir / game / job.label / run_slug
                requests.append(
                    RunRequest(
                        game=game,
                        job_label=job.label,
                        run_index=run_index,
                        total_num_runs=job.run_count,
                        model_name=job.model_name,
                        company=infer_model_provider(job.model_name),
                        thinking_mode=job.thinking_mode,
                        games_label=job.games_label,
                        duration_seconds=job.duration_seconds,
                        max_actions_per_turn=job.max_actions_per_turn,
                        frames_per_action=job.frames_per_action,
                        history_clips=job.history_clips,
                        non_zero_reward_clips=job.non_zero_reward_clips,
                        prompt_mode=job.prompt_mode,
                        seed=(
                            job.seed
                            if job.seed is not None
                            else (
                                None
                                if job.seed_start is None
                                else job.seed_start + run_index - 1
                            )
                        ),
                        minimal_logging=job.minimal_logging,
                        output_dir=str(output_dir),
                        log_path=str(log_dir / f"{game}_{job.label}_{run_slug}.log"),
                        run_label=_build_run_label(
                            batch_timestamp=batch_timestamp,
                            job_label=job.label,
                            run_index=run_index,
                        ),
                    )
                )
    return requests


def _extract_cfg_run_label(job_label: str) -> str | None:
    match = re.search(r"(cfg_\d+)$", job_label)
    if match:
        return match.group(1)
    return None


def _build_run_label(
    *,
    batch_timestamp: str | None,
    job_label: str,
    run_index: int,
) -> str | None:
    cfg_label = _extract_cfg_run_label(job_label)
    if not cfg_label and batch_timestamp is None:
        return None

    parts: list[str] = []
    if batch_timestamp:
        parts.append(batch_timestamp)
    if cfg_label:
        parts.append(cfg_label)
    parts.append(f"run_{run_index:03d}")
    return "_".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_dir = Path(__file__).resolve().parent
    common_settings = load_yaml_config(args.common_config)
    setting_entries = load_yaml_config(args.runs_config)
    if not isinstance(common_settings, dict):
        parser.error("--common-config must contain a mapping.")
    if not isinstance(setting_entries, list):
        parser.error("--runs-config must contain a list of settings.")
    batch_options, jobs = build_jobs_from_config(
        common_settings=common_settings,
        setting_entries=setting_entries,
    )
    if args.minimal_logging:
        jobs = [
            dataclasses.replace(job, minimal_logging=True)
            for job in jobs
        ]

    selected_games = list(dict.fromkeys(game for job in jobs for game in job.games))
    all_games = list_game_keys()

    timestamp = dt.datetime.now().strftime("%m%d_%H%M%S")
    batch_label = Path(str(args.runs_config)).stem
    if len(selected_games) == 1 and uses_canonical_game_storage(selected_games[0]):
        batch_root = game_batch_root(project_dir, selected_games[0]) / timestamp
        base_output_dir = game_root(project_dir, selected_games[0])
    else:
        batch_root = Path(str(batch_options["output_dir"])) / f"{batch_label}_{timestamp}"
        base_output_dir = batch_root / "runs"
    logs_dir = batch_root / "logs"
    requests = expand_run_requests(
        jobs=jobs,
        project_dir=project_dir,
        base_output_dir=base_output_dir,
        log_dir=logs_dir,
        batch_timestamp=timestamp,
    )

    print(f"batch_root={batch_root}")
    print(f"all_games={','.join(all_games)}")
    print(f"selected_games={','.join(selected_games)}")
    print(
        "max_concurrency_by_company="
        f"{json.dumps(batch_options['max_concurrency_by_company'], sort_keys=True)}"
    )
    print(f"total_runs={len(requests)}")

    results = execute_requests(
        requests=requests,
        max_retries=int(batch_options["max_retries"]),
        retry_backoff_seconds=float(batch_options["retry_backoff_seconds"]),
        render_video_fps=int(batch_options["render_video_fps"]),
        max_concurrency_by_company=batch_options.get("max_concurrency_by_company"),
        default_max_concurrency=1,
    )
    refresh_model_summaries(project_dir, results)

    batch_summary = {
        "batch_root": str(batch_root),
        "game_selection": None,
        "selected_games": selected_games,
        "all_games": all_games,
        "max_concurrency_by_company": batch_options.get("max_concurrency_by_company"),
        "common_config_path": str(Path(args.common_config).resolve()) if args.common_config else None,
        "runs_config_path": (
            str(Path(args.runs_config).resolve())
            if args.runs_config
            else None
        ),
        "total_runs": len(results),
        "successful_runs": sum(1 for result in results if result.success),
        "failed_runs": sum(1 for result in results if not result.success),
        "results": [dataclasses.asdict(result) for result in sorted(results, key=_sort_key)],
    }
    summary_path = batch_root / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(batch_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"batch_summary={summary_path}")
    return 0 if batch_summary["failed_runs"] == 0 else 1


def refresh_model_summaries(project_dir: str | Path, results: list[RunResult]) -> None:
    """Rebuild per-game and cross-game summaries after a batch finishes."""

    refreshed_games = sorted(
        {
            result.game
            for result in results
            if result.success and uses_canonical_game_storage(result.game)
        }
    )
    for game in refreshed_games:
        update_game_model_summary(project_dir, game)


def execute_run(
    request: RunRequest,
    max_retries: int,
    retry_backoff_seconds: float,
    render_video_fps: int,
) -> RunResult:
    print(_format_run_start_line(request), flush=True)
    current_thinking = request.thinking_mode
    attempts = 0
    combined_output = ""
    while True:
        attempts += 1
        completed = _run_subprocess(
            request=request,
            thinking_mode=current_thinking,
        )
        combined_output = completed.stdout or ""
        _write_log(
            request.log_path,
            header=(
                f"game={request.game}\n"
                f"model={request.model_name}\n"
                f"company={request.company}\n"
                f"requested_thinking_mode={request.thinking_mode}\n"
                f"final_thinking_mode={current_thinking}\n"
                f"prompt_mode={request.prompt_mode}\n"
                f"duration_seconds={request.duration_seconds}\n"
                f"frames_per_action={request.frames_per_action}\n"
                f"history_clips={request.history_clips}\n"
                f"non_zero_reward_clips={request.non_zero_reward_clips}\n"
                f"minimal_logging={str(request.minimal_logging).lower()}\n"
                f"attempt={attempts}\n"
                f"return_code={completed.returncode}\n\n"
            ),
            content=combined_output,
        )

        if completed.returncode == 0:
            run_dir = normalize_run_dir(extract_run_dir(combined_output))
            summary = load_run_summary(run_dir)
            is_full_duration_run = _is_full_duration_run(summary, request.duration_seconds)
            if not is_full_duration_run and attempts <= max_retries:
                sleep_seconds = compute_retry_sleep_seconds(
                    attempt=attempts,
                    base_backoff_seconds=retry_backoff_seconds,
                )
                time.sleep(sleep_seconds)
                continue
            video_path = None
            video_error = None
            if run_dir:
                try:
                    rendered_path = render_run_video(run_dir=run_dir, fps=render_video_fps)
                    video_path = str(rendered_path)
                    summary = _attach_video_metadata(summary, video_path, None, run_dir)
                except Exception as exc:  # pragma: no cover
                    video_error = str(exc)
                    summary = _attach_video_metadata(summary, None, video_error, run_dir)
                if request.minimal_logging:
                    apply_minimal_logging_policy(run_dir)
            return RunResult(
                game=request.game,
                job_label=request.job_label,
                run_index=request.run_index,
                model_name=request.model_name,
                requested_thinking_mode=request.thinking_mode,
                final_thinking_mode=current_thinking,
                success=is_full_duration_run,
                return_code=0 if is_full_duration_run else 1,
                run_dir=run_dir,
                stop_reason=_extract_stop_reason(summary, combined_output),
                log_path=request.log_path,
                attempts=attempts,
                summary=summary,
                video_path=video_path,
                video_error=video_error,
                error_type=None if is_full_duration_run else "incomplete_run",
            )

        error_type = classify_error_output(combined_output)
        if (error_type == "transient" or error_type is None) and attempts <= max_retries:
            sleep_seconds = compute_retry_sleep_seconds(
                attempt=attempts,
                base_backoff_seconds=retry_backoff_seconds,
            )
            time.sleep(sleep_seconds)
            continue

        return RunResult(
            game=request.game,
            job_label=request.job_label,
            run_index=request.run_index,
            model_name=request.model_name,
            requested_thinking_mode=request.thinking_mode,
            final_thinking_mode=current_thinking,
            success=False,
            return_code=completed.returncode,
            run_dir=normalize_run_dir(extract_run_dir(combined_output)),
            stop_reason=None,
            log_path=request.log_path,
            attempts=attempts,
            summary=None,
            video_path=None,
            video_error=None,
            error_type=error_type,
        )


def execute_requests(
    requests: list[RunRequest],
    *,
    max_retries: int,
    retry_backoff_seconds: float,
    render_video_fps: int,
    max_concurrency_by_company: dict[str, int] | None,
    default_max_concurrency: int,
) -> list[RunResult]:
    max_workers = _resolve_executor_worker_count(
        max_concurrency=default_max_concurrency,
        max_concurrency_by_company=max_concurrency_by_company,
    )
    company_limits = _resolve_company_limits(
        max_concurrency_by_company=max_concurrency_by_company,
        default_limit=default_max_concurrency,
    )
    active_counts = {company: 0 for company in _SUPPORTED_COMPANIES}
    pending_requests = list(requests)
    results: list[RunResult] = []
    in_flight: dict[concurrent.futures.Future[RunResult], RunRequest] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending_requests or in_flight:
            made_progress = False
            while len(in_flight) < max_workers:
                next_index = _find_next_schedulable_request_index(
                    pending_requests=pending_requests,
                    active_counts=active_counts,
                    company_limits=company_limits,
                )
                if next_index is None:
                    break
                request = pending_requests.pop(next_index)
                future = executor.submit(
                    execute_run,
                    request,
                    max_retries,
                    retry_backoff_seconds,
                    render_video_fps,
                )
                in_flight[future] = request
                active_counts[request.company] += 1
                made_progress = True

            if not in_flight:
                if pending_requests:
                    raise RuntimeError("No schedulable requests remain, but pending requests still exist.")
                break

            done, _ = concurrent.futures.wait(
                in_flight.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                request = in_flight.pop(future)
                active_counts[request.company] -= 1
                result = future.result()
                results.append(result)
                status = "OK" if result.success else "FAIL"
                suffix = f" stop_reason={result.stop_reason}" if result.stop_reason else ""
                print(
                    f"[{status}] {result.game} {result.job_label} run={result.run_index} "
                    f"thinking={result.final_thinking_mode} attempts={result.attempts}"
                    f"{suffix}"
                )
                made_progress = True

            if not made_progress and pending_requests:
                raise RuntimeError("Scheduler made no progress while requests were pending.")

    return results


def _resolve_company_limits(
    *,
    max_concurrency_by_company: dict[str, int] | None,
    default_limit: int,
) -> dict[str, int]:
    if not max_concurrency_by_company:
        return {company: default_limit for company in _SUPPORTED_COMPANIES}
    return {
        company: max_concurrency_by_company.get(company, default_limit)
        for company in _SUPPORTED_COMPANIES
    }


def _find_next_schedulable_request_index(
    *,
    pending_requests: list[RunRequest],
    active_counts: dict[str, int],
    company_limits: dict[str, int],
) -> int | None:
    for index, request in enumerate(pending_requests):
        if active_counts[request.company] < company_limits[request.company]:
            return index
    return None


def _resolve_executor_worker_count(
    *,
    max_concurrency: int,
    max_concurrency_by_company: dict[str, int] | None,
) -> int:
    if not max_concurrency_by_company:
        return max_concurrency
    return sum(max_concurrency_by_company.get(company, max_concurrency) for company in _SUPPORTED_COMPANIES)


def _run_subprocess(
    request: RunRequest,
    thinking_mode: str,
) -> subprocess.CompletedProcess[str]:
    script_path = Path(__file__).resolve().with_name("main.py")
    env = os.environ.copy()
    env[_INTERNAL_REQUEST_ENV] = json.dumps(
        {
            "game": request.game,
            "model": request.model_name,
            "thinking": thinking_mode,
            "duration_seconds": request.duration_seconds,
            "output_dir": request.output_dir,
            "seed": request.seed,
            "max_actions_per_turn": request.max_actions_per_turn,
            "frames_per_action": request.frames_per_action,
            "history_clips": request.history_clips,
            "non_zero_reward_clips": request.non_zero_reward_clips,
            "prompt_mode": request.prompt_mode,
            "run_label": request.run_label,
            "minimal_logging": request.minimal_logging,
        },
        sort_keys=True,
    )
    command = [
        sys.executable,
        str(script_path),
    ]

    return subprocess.run(
        command,
        cwd=_subprocess_cwd(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def classify_error_output(output: str) -> str | None:
    normalized = output.lower()
    if "resource_exhausted" in normalized or "429" in normalized:
        return "transient"
    if "503 unavailable" in normalized or "high demand" in normalized:
        return "transient"
    if "rate limit" in normalized or "too many requests" in normalized:
        return "transient"
    if "httpx.connecterror" in normalized:
        return "transient"
    if "httpx.readtimeout" in normalized or "httpx.timeoutexception" in normalized:
        return "transient"
    if "readtimeout" in normalized or "timed out" in normalized:
        return "transient"
    if "no route to host" in normalized:
        return "transient"
    if "nodename nor servname provided" in normalized:
        return "transient"
    if "temporary failure in name resolution" in normalized:
        return "transient"
    if "budget 0 is invalid" in normalized or "only works in thinking mode" in normalized:
        return "thinking_required"
    return None


def compute_retry_sleep_seconds(
    attempt: int,
    base_backoff_seconds: float,
    jitter_ratio: float = 0.2,
    max_backoff_seconds: float = 300.0,
) -> float:
    """Return exponential backoff with bounded jitter for transient retries."""

    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    exponential = base_backoff_seconds * (2 ** (attempt - 1))
    capped = min(exponential, max_backoff_seconds)
    jitter_window = capped * jitter_ratio
    if jitter_window <= 0:
        return capped
    return capped + random.uniform(0.0, jitter_window)


def extract_run_dir(output: str) -> str | None:
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped.startswith("runs/") and Path(stripped).name:
            return stripped
        if "/runs/" in stripped and Path(stripped).name:
            return stripped
    return None


def normalize_run_dir(run_dir: str | None) -> str | None:
    if not run_dir:
        return None
    path = Path(run_dir)
    if path.is_absolute():
        return str(path)
    return str((_subprocess_cwd() / path).resolve())


def load_run_summary(run_dir: str | None) -> dict[str, object] | None:
    if not run_dir:
        return None
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _attach_video_metadata(
    summary: dict[str, object] | None,
    video_path: str | None,
    video_error: str | None,
    run_dir: str | None,
) -> dict[str, object] | None:
    if not summary or not run_dir:
        return summary
    summary["video_path"] = video_path
    summary["video_error"] = video_error
    summary_path = Path(run_dir) / "summary.json"
    if summary_path.exists():
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def _extract_stop_reason(
    summary: dict[str, object] | None,
    output: str,
) -> str | None:
    if summary and "stop_reason" in summary:
        stop_reason = summary["stop_reason"]
        return str(stop_reason) if stop_reason is not None else None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines[-1]
    return None


def _is_full_duration_run(
    summary: dict[str, object] | None,
    expected_duration_seconds: int,
) -> bool:
    if not summary:
        return False
    stop_reason = summary.get("stop_reason")
    if stop_reason != "frame_budget":
        return False
    duration_seconds = summary.get("duration_seconds")
    if duration_seconds is None:
        return True
    return int(duration_seconds) == expected_duration_seconds


def _write_log(path: str | Path, header: str, content: str) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(header + content, encoding="utf-8")


def _format_run_start_line(request: RunRequest) -> str:
    seed_value = "null" if request.seed is None else str(request.seed)
    return (
        "[START] "
        f"model_name={request.model_name} "
        f"thinking_mode={request.thinking_mode} "
        f"prompt_mode={request.prompt_mode} "
        f"frames_per_action={request.frames_per_action} "
        f"history_clips={request.history_clips} "
        f"non_zero_reward_clips={request.non_zero_reward_clips} "
        f"minimal_logging={str(request.minimal_logging).lower()} "
        f"games={request.games_label} "
        f"selected_game={request.game} "
        f"seed={seed_value} "
        f"current_num_run={request.run_index} "
        f"total_num_runs={request.total_num_runs} "
        f"output_dir={request.output_dir}"
    )


def _sort_key(result: RunResult) -> tuple[str, int]:
    return (result.game, result.job_label, result.run_index)


def _subprocess_cwd() -> Path:
    return Path(__file__).resolve().parent


if __name__ == "__main__":
    raise SystemExit(main())
