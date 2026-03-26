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

from run_storage import game_model_dir, game_root, resolve_output_layout, update_game_model_summary, uses_canonical_game_storage


class RunStorageTests(unittest.TestCase):
    def test_canonical_storage_enabled_for_all_prompt_games(self) -> None:
        self.assertTrue(uses_canonical_game_storage("breakout"))
        self.assertTrue(uses_canonical_game_storage("assault"))
        self.assertTrue(uses_canonical_game_storage("seaquest"))
        self.assertTrue(uses_canonical_game_storage("time_pilot"))
        self.assertFalse(uses_canonical_game_storage("termination"))

    def test_resolve_output_layout_uses_canonical_model_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir, nest_output_by_game = resolve_output_layout(
                project_dir=tmpdir,
                game="assault",
                model_name="gpt-5.4-mini",
                requested_output_dir="ignored",
            )

        self.assertEqual(output_dir, game_model_dir(tmpdir, "assault", "gpt-5.4-mini"))
        self.assertFalse(nest_output_by_game)

    def test_update_game_model_summary_aggregates_per_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = game_root(tmpdir, "assault")
            run_a = root / "gpt-5.4-mini" / "20260324_100000"
            run_b = root / "gpt-5.4-mini" / "20260324_110000"
            run_c = root / "gpt-5.4-mini" / "20260324_120000"
            run_a.mkdir(parents=True, exist_ok=True)
            run_b.mkdir(parents=True, exist_ok=True)
            run_c.mkdir(parents=True, exist_ok=True)
            (run_a / "summary.json").write_text(
                json.dumps(
                    {
                        "duration_seconds": 30,
                        "model_name": "gpt-5.4-mini",
                        "total_reward": 2.0,
                        "total_lost_lives": 1,
                        "turn_count": 10,
                        "frame_count": 100,
                        "stop_reason": "frame_budget",
                    }
                ),
                encoding="utf-8",
            )
            (run_b / "summary.json").write_text(
                json.dumps(
                    {
                        "duration_seconds": 30,
                        "model_name": "gpt-5.4-mini",
                        "total_reward": 6.0,
                        "total_lost_lives": 3,
                        "turn_count": 14,
                        "frame_count": 120,
                        "stop_reason": "frame_budget",
                    }
                ),
                encoding="utf-8",
            )
            (run_c / "summary.json").write_text(
                json.dumps(
                    {
                        "duration_seconds": 1,
                        "model_name": "gpt-5.4-mini",
                        "total_reward": 99.0,
                        "total_lost_lives": 0,
                        "turn_count": 4,
                        "frame_count": 31,
                        "stop_reason": "frame_budget",
                    }
                ),
                encoding="utf-8",
            )

            summary_path = update_game_model_summary(tmpdir, "assault")
            payload = json.loads(summary_path.read_text(encoding="utf-8"))

        model_summary = payload["models"]["gpt-5.4-mini"]
        self.assertEqual(payload["game"], "assault")
        self.assertEqual(model_summary["run_count"], 2)
        self.assertEqual(model_summary["avg_total_reward"], 4.0)
        self.assertEqual(model_summary["avg_total_lost_lives"], 2.0)
        self.assertEqual(model_summary["avg_turn_count"], 12.0)
        self.assertEqual(model_summary["avg_frame_count"], 110.0)
        self.assertEqual(model_summary["latest_timestamp"], "20260324_110000")

    def test_update_game_model_summary_ignores_non_full_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = game_root(tmpdir, "assault")
            run_dir = root / "gpt-5.4-mini" / "20260324_100000"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "duration_seconds": 30,
                        "model_name": "gpt-5.4-mini",
                        "total_reward": 2.0,
                        "total_lost_lives": 1,
                        "turn_count": 10,
                        "frame_count": 100,
                        "stop_reason": "lost_lives",
                    }
                ),
                encoding="utf-8",
            )

            summary_path = update_game_model_summary(tmpdir, "assault")
            payload = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["models"], {})


if __name__ == "__main__":
    unittest.main()
