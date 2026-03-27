"""CLI entrypoint for a single AtariBench run."""

from __future__ import annotations

import argparse
import json
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
    from games import get_game_spec, list_game_keys
    from llm import build_model_client, validate_model_thinking_mode
    from run_storage import resolve_output_layout, update_game_model_summary, uses_canonical_game_storage
    from viz import render_run_video
else:
    from .core.pipeline import PipelineConfig, PipelineRunner
    from .games import get_game_spec, list_game_keys
    from .llm import build_model_client, validate_model_thinking_mode
    from .run_storage import resolve_output_layout, update_game_model_summary, uses_canonical_game_storage
    from .viz import render_run_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an AtariBench model pipeline.")
    parser.add_argument("--game", required=True, choices=list_game_keys())
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "gemini", "openai", "anthropic"],
        help="Backend provider. Defaults to auto-detect from --model.",
    )
    parser.add_argument(
        "--thinking",
        default="default",
        choices=["default", "auto", "on", "off", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
        help="Control provider thinking/reasoning mode.",
    )
    parser.add_argument("--duration-seconds", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "runs"),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-actions-per-turn", type=int, default=10)
    parser.add_argument("--history-clips", type=int, default=3)
    parser.add_argument("--non-zero-reward-clips", type=int, default=3)
    parser.add_argument(
        "--prompt-mode",
        default="structured_history",
        choices=["structured_history", "append_only"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_dir = Path(__file__).resolve().parent
    validate_model_thinking_mode(args.model, args.thinking)

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
        history_clips=args.history_clips,
        non_zero_reward_clips=args.non_zero_reward_clips,
        prompt_mode=args.prompt_mode,
        model_name=args.model,
        thinking_mode=args.thinking,
        seed=args.seed,
        output_dir=output_dir,
        nest_output_by_game=nest_output_by_game,
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
