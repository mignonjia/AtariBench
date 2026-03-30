"""CLI entrypoint for a single AtariBench run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap_local_paths() -> None:
    project_dir = Path(__file__).resolve().parent
    candidate = str(project_dir)
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


if __package__ in {None, ""}:
    _bootstrap_local_paths()
    from core.pipeline import PipelineConfig, PipelineRunner
    from core.trajectory import apply_minimal_logging_policy
    from games import get_game_spec, list_game_keys
    from llm import build_model_client, validate_model_thinking_mode
    from run_storage import resolve_output_layout, update_game_model_summary, uses_canonical_game_storage
    from viz import render_run_video
else:
    from .core.pipeline import PipelineConfig, PipelineRunner
    from .core.trajectory import apply_minimal_logging_policy
    from .games import get_game_spec, list_game_keys
    from .llm import build_model_client, validate_model_thinking_mode
    from .run_storage import resolve_output_layout, update_game_model_summary, uses_canonical_game_storage
    from .viz import render_run_video


_INTERNAL_REQUEST_ENV = "ATARIBENCH_INTERNAL_RUN_REQUEST"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an AtariBench model pipeline.")
    parser.add_argument("--game", required=True, choices=list_game_keys())
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument(
        "--thinking",
        default="default",
        choices=["default", "auto", "on", "off", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
        help="Control provider thinking/reasoning mode.",
    )
    parser.add_argument("--duration-seconds", type=int, default=30)
    parser.add_argument(
        "--prompt-mode",
        default="structured_history",
        choices=["structured_history", "append_only"],
    )
    parser.add_argument(
        "--context-cache",
        action="store_true",
        help=(
            "Enable explicit provider context/prompt cache hints for append_only. "
            "Structured_history requests stay unchanged."
        ),
    )
    parser.add_argument(
        "--minimal-logging",
        action="store_true",
        help=(
            "After rendering completes, keep only summary.json, turns.jsonl, and "
            "visualization.mp4 in the run directory."
        ),
    )
    return parser


def _args_from_internal_request(project_dir: Path, raw_payload: str) -> argparse.Namespace:
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise ValueError(f"{_INTERNAL_REQUEST_ENV} must contain a JSON object.")
    return argparse.Namespace(
        game=payload["game"],
        model=payload["model"],
        thinking=payload["thinking"],
        duration_seconds=int(payload.get("duration_seconds", 30)),
        prompt_mode=payload.get("prompt_mode", "structured_history"),
        context_cache=bool(payload.get("context_cache", False)),
        minimal_logging=bool(payload.get("minimal_logging", False)),
        provider=payload.get("provider", "auto"),
        output_dir=str(payload.get("output_dir", project_dir / "runs")),
        seed=None if payload.get("seed") is None else int(payload["seed"]),
        max_actions_per_turn=int(payload.get("max_actions_per_turn", 10)),
        frames_per_action=int(payload.get("frames_per_action", 3)),
        history_clips=int(payload.get("history_clips", 3)),
        non_zero_reward_clips=int(payload.get("non_zero_reward_clips", 3)),
        run_label=payload.get("run_label"),
    )


def main(argv: list[str] | None = None) -> int:
    project_dir = Path(__file__).resolve().parent
    raw_internal_request = os.environ.get(_INTERNAL_REQUEST_ENV)
    if raw_internal_request:
        args = _args_from_internal_request(project_dir, raw_internal_request)
    else:
        parser = build_parser()
        parsed_args = parser.parse_args(argv)
        args = argparse.Namespace(
            game=parsed_args.game,
            model=parsed_args.model,
            thinking=parsed_args.thinking,
            duration_seconds=parsed_args.duration_seconds,
            prompt_mode=parsed_args.prompt_mode,
            context_cache=parsed_args.context_cache,
            minimal_logging=parsed_args.minimal_logging,
            provider="auto",
            output_dir=str(project_dir / "runs"),
            seed=None,
            max_actions_per_turn=10,
            frames_per_action=3,
            history_clips=3,
            non_zero_reward_clips=3,
            run_label=None,
        )
    validate_model_thinking_mode(args.model, args.thinking)
    history_clips = args.history_clips
    non_zero_reward_clips = args.non_zero_reward_clips
    if args.prompt_mode == "append_only":
        history_clips = -1
        non_zero_reward_clips = -1
    else:
        args.context_cache = False

    game_spec = get_game_spec(args.game)
    output_dir, nest_output_by_game = resolve_output_layout(
        project_dir=project_dir,
        game=args.game,
        model_name=args.model,
        requested_output_dir=args.output_dir,
    )
    config = PipelineConfig(
        duration_seconds=args.duration_seconds,
        max_actions_per_turn=args.max_actions_per_turn,
        frames_per_action=args.frames_per_action,
        history_clips=history_clips,
        non_zero_reward_clips=non_zero_reward_clips,
        prompt_mode=args.prompt_mode,
        context_cache=args.context_cache,
        model_name=args.model,
        thinking_mode=args.thinking,
        seed=args.seed,
        output_dir=output_dir,
        nest_output_by_game=nest_output_by_game,
        run_label=args.run_label,
        minimal_logging=args.minimal_logging,
    )
    client = build_model_client(model_name=args.model, provider=args.provider)
    runner = PipelineRunner(
        game_spec=game_spec,
        model_client=client,
        config=config,
    )
    summary = runner.run()
    video_path = None
    video_error = None
    try:
        rendered_path = render_run_video(run_dir=summary["run_dir"], fps=game_spec.fps)
        video_path = str(rendered_path)
    except Exception as exc:  # pragma: no cover
        video_error = str(exc)
    summary = _attach_video_metadata(summary, video_path=video_path, video_error=video_error)
    if args.minimal_logging:
        apply_minimal_logging_policy(summary["run_dir"])
    if uses_canonical_game_storage(args.game):
        update_game_model_summary(project_dir, args.game)
    print(summary["run_dir"])
    print(summary["stop_reason"])
    return 0


def _attach_video_metadata(
    summary: dict[str, object],
    video_path: str | None,
    video_error: str | None,
) -> dict[str, object]:
    summary["video_path"] = video_path
    summary["video_error"] = video_error
    summary_path = Path(str(summary["run_dir"])) / "summary.json"
    if summary_path.exists():
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
