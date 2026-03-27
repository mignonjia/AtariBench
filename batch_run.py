"""Batch runner for launching multiple AtariBench jobs with bounded concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import json
import random
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
    from games import list_game_keys
    from llm import validate_model_thinking_mode
    from run_storage import game_batch_root, game_root, sanitize_model_label, uses_canonical_game_storage
    from viz import render_run_video
else:
    from .games import list_game_keys
    from .llm import validate_model_thinking_mode
    from .run_storage import game_batch_root, game_root, sanitize_model_label, uses_canonical_game_storage
    from .viz import render_run_video


@dataclasses.dataclass(frozen=True)
class BatchJobSpec:
    """One logical batch job before expansion into individual runs."""

    model_name: str
    run_count: int
    thinking_mode: str
    label: str


@dataclasses.dataclass(frozen=True)
class RunRequest:
    """One concrete run request for the subprocess-backed batch runner."""

    job_label: str
    run_index: int
    model_name: str
    thinking_mode: str
    output_dir: str
    log_path: str


@dataclasses.dataclass(frozen=True)
class RunResult:
    """Final status for one batch run."""

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
    parser.add_argument("--game", required=True, choices=list_game_keys())
    parser.add_argument(
        "--job",
        action="append",
        required=True,
        help=(
            "Batch job spec in the form MODEL:COUNT[:THINKING]. "
            "Example: gemini-2.5-flash:5:off"
        ),
    )
    parser.add_argument("--duration-seconds", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "runs" / "batches"),
    )
    parser.add_argument("--max-actions-per-turn", type=int, default=10)
    parser.add_argument("--history-clips", type=int, default=3)
    parser.add_argument("--non-zero-reward-clips", type=int, default=3)
    parser.add_argument(
        "--prompt-mode",
        default="structured_history",
        choices=["structured_history", "append_only"],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument(
        "--fallback-thinking",
        choices=["minimal", "default", "on"],
        default="minimal",
        help="Fallback thinking mode when a model rejects budget 0.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Retry count for transient provider errors such as 429/503.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=5.0,
        help="Base backoff in seconds between transient retries.",
    )
    parser.add_argument("--render-video-fps", type=int, default=30)
    return parser


def parse_job_spec(raw_spec: str) -> BatchJobSpec:
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
    label = sanitize_model_label(model_name)
    return BatchJobSpec(
        model_name=model_name,
        run_count=run_count,
        thinking_mode=thinking_mode,
        label=label,
    )


def expand_run_requests(
    jobs: list[BatchJobSpec],
    base_output_dir: str | Path,
    log_dir: str | Path,
    canonical_output_root: str | Path | None = None,
) -> list[RunRequest]:
    requests: list[RunRequest] = []
    base_output_dir = Path(base_output_dir)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    canonical_output_root = (
        Path(canonical_output_root) if canonical_output_root is not None else None
    )

    for job in jobs:
        for run_index in range(1, job.run_count + 1):
            run_slug = f"run_{run_index:03d}"
            output_dir = (
                canonical_output_root / job.label
                if canonical_output_root is not None
                else base_output_dir / job.label / run_slug
            )
            requests.append(
                RunRequest(
                    job_label=job.label,
                    run_index=run_index,
                    model_name=job.model_name,
                    thinking_mode=job.thinking_mode,
                    output_dir=str(output_dir),
                    log_path=str(log_dir / f"{job.label}_{run_slug}.log"),
                )
            )
    return requests


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_dir = Path(__file__).resolve().parent

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if uses_canonical_game_storage(args.game):
        batch_root = game_batch_root(project_dir, args.game) / timestamp
        base_output_dir = game_root(project_dir, args.game)
        canonical_output_root = base_output_dir
    else:
        batch_root = Path(args.output_dir) / f"{args.game}_{timestamp}"
        base_output_dir = batch_root / "runs"
        canonical_output_root = None
    logs_dir = batch_root / "logs"
    jobs = [parse_job_spec(raw_spec) for raw_spec in args.job]
    requests = expand_run_requests(
        jobs=jobs,
        base_output_dir=base_output_dir,
        log_dir=logs_dir,
        canonical_output_root=canonical_output_root,
    )

    print(f"batch_root={batch_root}")
    print(f"max_concurrency={args.max_concurrency}")
    print(f"total_runs={len(requests)}")

    results: list[RunResult] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_concurrency
    ) as executor:
        future_map = {
            executor.submit(
                execute_run,
                request,
                args.game,
                args.duration_seconds,
                args.max_actions_per_turn,
                args.history_clips,
                args.non_zero_reward_clips,
                args.prompt_mode,
                args.seed,
                args.fallback_thinking,
                args.max_retries,
                args.retry_backoff_seconds,
                args.render_video_fps,
            ): request
            for request in requests
        }
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            results.append(result)
            status = "OK" if result.success else "FAIL"
            suffix = f" stop_reason={result.stop_reason}" if result.stop_reason else ""
            print(
                f"[{status}] {result.job_label} run={result.run_index} "
                f"thinking={result.final_thinking_mode} attempts={result.attempts}"
                f"{suffix}"
            )

    batch_summary = {
        "batch_root": str(batch_root),
        "game": args.game,
        "max_concurrency": args.max_concurrency,
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


def execute_run(
    request: RunRequest,
    game: str,
    duration_seconds: int,
    max_actions_per_turn: int,
    history_clips: int,
    non_zero_reward_clips: int,
    prompt_mode: str,
    seed: int | None,
    fallback_thinking: str,
    max_retries: int,
    retry_backoff_seconds: float,
    render_video_fps: int,
) -> RunResult:
    current_thinking = request.thinking_mode
    attempts = 0
    combined_output = ""
    while True:
        attempts += 1
        completed = _run_subprocess(
            request=request,
            game=game,
            duration_seconds=duration_seconds,
            max_actions_per_turn=max_actions_per_turn,
            history_clips=history_clips,
            non_zero_reward_clips=non_zero_reward_clips,
            prompt_mode=prompt_mode,
            seed=seed,
            thinking_mode=current_thinking,
        )
        combined_output = completed.stdout or ""
        _write_log(
            request.log_path,
            header=(
                f"model={request.model_name}\n"
                f"requested_thinking_mode={request.thinking_mode}\n"
                f"final_thinking_mode={current_thinking}\n"
                f"attempt={attempts}\n"
                f"return_code={completed.returncode}\n\n"
            ),
            content=combined_output,
        )

        if completed.returncode == 0:
            run_dir = normalize_run_dir(extract_run_dir(combined_output))
            summary = load_run_summary(run_dir)
            is_full_duration_run = _is_full_duration_run(summary, duration_seconds)
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
            return RunResult(
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

        if _should_fallback_to_thinking(combined_output, current_thinking):
            current_thinking = fallback_thinking
            continue

        error_type = classify_error_output(combined_output)
        if (error_type == "transient" or error_type is None) and attempts <= max_retries:
            sleep_seconds = compute_retry_sleep_seconds(
                attempt=attempts,
                base_backoff_seconds=retry_backoff_seconds,
            )
            time.sleep(sleep_seconds)
            continue

        return RunResult(
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


def _run_subprocess(
    request: RunRequest,
    game: str,
    duration_seconds: int,
    max_actions_per_turn: int,
    history_clips: int,
    non_zero_reward_clips: int,
    prompt_mode: str,
    seed: int | None,
    thinking_mode: str,
) -> subprocess.CompletedProcess[str]:
    script_path = Path(__file__).resolve().with_name("main.py")
    command = [
        sys.executable,
        str(script_path),
        "--game",
        game,
        "--model",
        request.model_name,
        "--thinking",
        thinking_mode,
        "--duration-seconds",
        str(duration_seconds),
        "--output-dir",
        request.output_dir,
        "--max-actions-per-turn",
        str(max_actions_per_turn),
        "--history-clips",
        str(history_clips),
        "--non-zero-reward-clips",
        str(non_zero_reward_clips),
        "--prompt-mode",
        prompt_mode,
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])

    return subprocess.run(
        command,
        cwd=_subprocess_cwd(),
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
        if "/breakout/" in stripped or stripped.startswith("runs/"):
            if Path(stripped).name and stripped.startswith("runs/"):
                return stripped
        if "/AtariBench/runs/" in stripped:
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


def _should_fallback_to_thinking(output: str, current_thinking: str) -> bool:
    if current_thinking != "off":
        return False
    return classify_error_output(output) == "thinking_required"


def _write_log(path: str | Path, header: str, content: str) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(header + content, encoding="utf-8")


def _sort_key(result: RunResult) -> tuple[str, int]:
    return (result.job_label, result.run_index)


def _subprocess_cwd() -> Path:
    return Path(__file__).resolve().parent


if __name__ == "__main__":
    raise SystemExit(main())
