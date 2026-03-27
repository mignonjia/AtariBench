"""Remove invalid stored runs and refresh per-game summaries."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from run_storage import update_game_model_summary


@dataclass(frozen=True)
class RemovalCandidate:
    game: str
    run_dir: Path
    reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove incomplete, invalid, or non-30-second run directories "
            "and refresh per-game summaries."
        )
    )
    parser.add_argument(
        "--project-dir",
        default=str(Path(__file__).resolve().parent),
        help="Project root containing the runs directory.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete matching run directories. Defaults to dry-run.",
    )
    return parser


def iter_prunable_run_dirs(project_dir: str | Path) -> dict[str, list[RemovalCandidate]]:
    root = Path(project_dir).resolve() / "runs"
    removals: dict[str, list[RemovalCandidate]] = {}
    if not root.exists():
        return removals

    for run_dir in sorted(root.glob("*/*/*")):
        if not run_dir.is_dir():
            continue
        game = run_dir.parents[1].name
        model_dir = run_dir.parent
        if game.startswith("_") or model_dir.name.startswith("_"):
            continue

        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            removals.setdefault(game, []).append(
                RemovalCandidate(game=game, run_dir=run_dir, reason="missing_summary")
            )
            continue

        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            removals.setdefault(game, []).append(
                RemovalCandidate(game=game, run_dir=run_dir, reason="invalid_summary")
            )
            continue

        duration_seconds = payload.get("duration_seconds")
        if duration_seconds is None:
            removals.setdefault(game, []).append(
                RemovalCandidate(game=game, run_dir=run_dir, reason="missing_duration")
            )
            continue
        if int(duration_seconds) == 30:
            continue

        removals.setdefault(game, []).append(
            RemovalCandidate(game=game, run_dir=run_dir, reason=f"duration_seconds={duration_seconds}")
        )

    return removals


def cleanup_invalid_runs(project_dir: str | Path, apply: bool) -> dict[str, list[Path]]:
    removals = iter_prunable_run_dirs(project_dir)
    for game, candidates in removals.items():
        for candidate in candidates:
            print(
                f"{'DELETE' if apply else 'DRY-RUN'} {candidate.run_dir} "
                f"[reason={candidate.reason}]"
            )
            if apply:
                shutil.rmtree(candidate.run_dir)
        update_game_model_summary(project_dir, game)
    return removals


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    removals = cleanup_invalid_runs(project_dir=args.project_dir, apply=args.apply)

    removed_count = sum(len(run_dirs) for run_dirs in removals.values())
    affected_games = len(removals)
    if args.apply:
        print(f"Removed {removed_count} run directories across {affected_games} games.")
    else:
        print(
            f"Would remove {removed_count} run directories across {affected_games} games. "
            "Re-run with --apply to delete them."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
