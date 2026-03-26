"""CLI entrypoint for a single AtariBench run."""

from __future__ import annotations

import argparse
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
    from llm import build_model_client
    from run_storage import resolve_output_layout, update_game_model_summary, uses_canonical_game_storage
else:
    from .core.pipeline import PipelineConfig, PipelineRunner
    from .games import get_game_spec, list_game_keys
    from .llm import build_model_client
    from .run_storage import resolve_output_layout, update_game_model_summary, uses_canonical_game_storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an AtariBench model pipeline.")
    parser.add_argument("--game", required=True, choices=list_game_keys())
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "gemini", "openai"],
        help="Backend provider. Defaults to auto-detect from --model.",
    )
    parser.add_argument(
        "--thinking",
        default="default",
        choices=["default", "on", "off", "minimal", "low"],
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_dir = Path(__file__).resolve().parent

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
    if uses_canonical_game_storage(args.game):
        update_game_model_summary(project_dir, args.game)
    print(summary["run_dir"])
    print(summary["stop_reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
