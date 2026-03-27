from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT / "AtariBench"
candidate = str(PROJECT_DIR)
if candidate not in sys.path:
    sys.path.insert(0, candidate)

from cleanup_invalid_runs import cleanup_invalid_runs, iter_prunable_run_dirs


class CleanupInvalidRunsTests(unittest.TestCase):
    def test_iter_prunable_run_dirs_finds_only_non_30_second_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            keep_dir = Path(tmpdir) / "runs" / "breakout" / "gemini-2.5-flash" / "20260324_100000"
            remove_dir = Path(tmpdir) / "runs" / "breakout" / "gemini-2.5-flash" / "20260324_110000"
            keep_dir.mkdir(parents=True, exist_ok=True)
            remove_dir.mkdir(parents=True, exist_ok=True)
            (keep_dir / "summary.json").write_text(
                json.dumps({"duration_seconds": 30, "model_name": "gemini-2.5-flash"}),
                encoding="utf-8",
            )
            (remove_dir / "summary.json").write_text(
                json.dumps({"duration_seconds": 10, "model_name": "gemini-2.5-flash"}),
                encoding="utf-8",
            )

            removals = iter_prunable_run_dirs(tmpdir)

        self.assertEqual(
            removals,
            {
                "breakout": [
                    type(removals["breakout"][0])(
                        game="breakout",
                        run_dir=remove_dir.resolve(),
                        reason="duration_seconds=10",
                    )
                ]
            },
        )

    def test_iter_prunable_run_dirs_flags_missing_summary_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            incomplete_dir = Path(tmpdir) / "runs" / "demon_attack" / "gemini-2.5-flash" / "20260324_110000"
            incomplete_dir.mkdir(parents=True, exist_ok=True)

            removals = iter_prunable_run_dirs(tmpdir)

        self.assertEqual(removals["demon_attack"][0].run_dir, incomplete_dir.resolve())
        self.assertEqual(removals["demon_attack"][0].reason, "missing_summary")

    def test_cleanup_invalid_runs_removes_runs_and_refreshes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            keep_dir = Path(tmpdir) / "runs" / "assault" / "gemini-2.5-flash" / "20260324_100000"
            remove_dir = Path(tmpdir) / "runs" / "assault" / "gemini-2.5-flash" / "20260324_110000"
            keep_dir.mkdir(parents=True, exist_ok=True)
            remove_dir.mkdir(parents=True, exist_ok=True)
            (keep_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "duration_seconds": 30,
                        "model_name": "gemini-2.5-flash",
                        "total_reward": 5.0,
                        "total_lost_lives": 1,
                        "turn_count": 10,
                        "frame_count": 901,
                        "stop_reason": "frame_budget",
                    }
                ),
                encoding="utf-8",
            )
            (remove_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "duration_seconds": 60,
                        "model_name": "gemini-2.5-flash",
                        "total_reward": 99.0,
                        "total_lost_lives": 0,
                        "turn_count": 20,
                        "frame_count": 1800,
                        "stop_reason": "frame_budget",
                    }
                ),
                encoding="utf-8",
            )

            cleanup_invalid_runs(tmpdir, apply=True)

            self.assertTrue(keep_dir.exists())
            self.assertFalse(remove_dir.exists())
            summary_path = Path(tmpdir) / "runs" / "assault" / "model_summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))

        model_summaries = payload["models"]["gemini-2.5-flash"]
        self.assertEqual(len(model_summaries), 1)
        self.assertEqual(model_summaries[0]["run_count"], 1)
        self.assertEqual(model_summaries[0]["latest_timestamp"], "20260324_100000")


if __name__ == "__main__":
    unittest.main()
