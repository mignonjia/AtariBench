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

from run_storage import (
    game_model_dir,
    game_root,
    resolve_output_layout,
    update_game_model_summary,
    update_runs_model_summary,
    uses_canonical_game_storage,
)


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

    def test_update_game_model_summary_aggregates_per_setting(self) -> None:
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
                        "prompt_mode": "append_only",
                        "frames_per_action": 3,
                        "thinking_mode": "low",
                        "thinking_level": "low",
                        "thinking_budget": None,
                        "history_clips": 3,
                        "non_zero_reward_clips": 2,
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
                        "prompt_mode": "append_only",
                        "frames_per_action": 3,
                        "thinking_mode": "low",
                        "thinking_level": "low",
                        "thinking_budget": None,
                        "history_clips": 3,
                        "non_zero_reward_clips": 2,
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
                        "duration_seconds": 30,
                        "model_name": "gpt-5.4-mini",
                        "prompt_mode": "structured_history",
                        "frames_per_action": 3,
                        "thinking_mode": "off",
                        "thinking_level": None,
                        "thinking_budget": 0,
                        "history_clips": 1,
                        "non_zero_reward_clips": 1,
                        "total_reward": 9.0,
                        "total_lost_lives": 0,
                        "turn_count": 4,
                        "frame_count": 901,
                        "stop_reason": "frame_budget",
                    }
                ),
                encoding="utf-8",
            )

            summary_path = update_game_model_summary(tmpdir, "assault")
            payload = json.loads(summary_path.read_text(encoding="utf-8"))

        model_summaries = payload["models"]["gpt-5.4-mini"]
        append_only_summary = next(
            summary for summary in model_summaries if summary["thinking_mode"] == "low"
        )
        structured_history_summary = next(
            summary for summary in model_summaries if summary["thinking_mode"] == "off"
        )
        self.assertEqual(payload["game"], "assault")
        self.assertEqual(len(model_summaries), 2)
        self.assertEqual(append_only_summary["run_count"], 2)
        self.assertEqual(append_only_summary["avg_total_reward"], 4.0)
        self.assertEqual(append_only_summary["avg_total_lost_lives"], 2.0)
        self.assertEqual(append_only_summary["avg_turn_count"], 12.0)
        self.assertEqual(append_only_summary["avg_frame_count"], 110.0)
        self.assertEqual(append_only_summary["latest_timestamp"], "20260324_110000")
        self.assertEqual(append_only_summary["thinking_mode"], "low")
        self.assertEqual(append_only_summary["prompt_mode"], "append_only")
        self.assertEqual(append_only_summary["frames_per_action"], 3)
        self.assertEqual(append_only_summary["thinking_level"], "low")
        self.assertIsNone(append_only_summary["thinking_budget"])
        self.assertEqual(append_only_summary["history_clips"], -1)
        self.assertEqual(append_only_summary["non_zero_reward_clips"], -1)
        self.assertEqual(structured_history_summary["run_count"], 1)
        self.assertEqual(structured_history_summary["avg_total_reward"], 9.0)
        self.assertEqual(structured_history_summary["prompt_mode"], "structured_history")
        self.assertEqual(structured_history_summary["frames_per_action"], 3)
        self.assertEqual(structured_history_summary["thinking_mode"], "off")
        self.assertEqual(structured_history_summary["thinking_budget"], 0)
        self.assertEqual(structured_history_summary["history_clips"], 1)
        self.assertEqual(structured_history_summary["non_zero_reward_clips"], 1)

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
                        "prompt_mode": "structured_history",
                        "frames_per_action": 3,
                        "thinking_mode": "off",
                        "thinking_level": None,
                        "thinking_budget": 0,
                        "history_clips": 3,
                        "non_zero_reward_clips": 3,
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

    def test_update_runs_model_summary_aggregates_flat_entries_per_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            assault_root = game_root(tmpdir, "assault")
            breakout_root = game_root(tmpdir, "breakout")
            assault_run = assault_root / "gemini-2.5-flash" / "20260324_100000"
            breakout_run_a = breakout_root / "gpt-5.4-mini" / "20260324_110000"
            breakout_run_b = breakout_root / "gpt-5.4-mini" / "20260324_120000"
            assault_run.mkdir(parents=True, exist_ok=True)
            breakout_run_a.mkdir(parents=True, exist_ok=True)
            breakout_run_b.mkdir(parents=True, exist_ok=True)
            (assault_run / "summary.json").write_text(
                json.dumps(
                    {
                        "duration_seconds": 30,
                        "model_name": "gemini-2.5-flash",
                        "prompt_mode": "structured_history",
                        "frames_per_action": 3,
                        "thinking_mode": "off",
                        "thinking_level": None,
                        "thinking_budget": 0,
                        "history_clips": 3,
                        "non_zero_reward_clips": 3,
                        "total_reward": 7.0,
                        "total_lost_lives": 1,
                        "turn_count": 11,
                        "frame_count": 901,
                        "stop_reason": "frame_budget",
                    }
                ),
                encoding="utf-8",
            )
            (breakout_run_a / "summary.json").write_text(
                json.dumps(
                    {
                        "duration_seconds": 30,
                        "model_name": "gpt-5.4-mini",
                        "prompt_mode": "append_only",
                        "frames_per_action": 3,
                        "thinking_mode": "low",
                        "thinking_level": "low",
                        "thinking_budget": None,
                        "history_clips": 4,
                        "non_zero_reward_clips": 2,
                        "total_reward": 3.0,
                        "total_lost_lives": 2,
                        "turn_count": 17,
                        "frame_count": 901,
                        "stop_reason": "frame_budget",
                    }
                ),
                encoding="utf-8",
            )
            (breakout_run_b / "summary.json").write_text(
                json.dumps(
                    {
                        "duration_seconds": 30,
                        "model_name": "gpt-5.4-mini",
                        "prompt_mode": "structured_history",
                        "frames_per_action": 3,
                        "thinking_mode": "off",
                        "thinking_level": None,
                        "thinking_budget": 0,
                        "history_clips": 2,
                        "non_zero_reward_clips": 1,
                        "total_reward": 5.0,
                        "total_lost_lives": 0,
                        "turn_count": 12,
                        "frame_count": 901,
                        "stop_reason": "frame_budget",
                    }
                ),
                encoding="utf-8",
            )

            update_game_model_summary(tmpdir, "assault")
            update_game_model_summary(tmpdir, "breakout")
            summary_path = update_runs_model_summary(tmpdir)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(len(payload["entries"]), 3)
        self.assertEqual(payload["entries"][0]["game"], "assault")
        self.assertEqual(payload["entries"][0]["model_name"], "gemini-2.5-flash")
        self.assertEqual(payload["entries"][0]["prompt_mode"], "structured_history")
        self.assertEqual(payload["entries"][0]["frames_per_action"], 3)
        self.assertEqual(payload["entries"][0]["thinking_mode"], "off")
        self.assertEqual(payload["entries"][0]["history_clips"], 3)
        self.assertEqual(payload["entries"][1]["game"], "breakout")
        self.assertEqual(payload["entries"][1]["model_name"], "gpt-5.4-mini")
        self.assertEqual(payload["entries"][1]["prompt_mode"], "append_only")
        self.assertEqual(payload["entries"][1]["frames_per_action"], 3)
        self.assertEqual(payload["entries"][1]["thinking_mode"], "low")
        self.assertEqual(payload["entries"][1]["history_clips"], -1)
        self.assertEqual(payload["entries"][1]["non_zero_reward_clips"], -1)
        self.assertEqual(payload["entries"][2]["game"], "breakout")
        self.assertEqual(payload["entries"][2]["model_name"], "gpt-5.4-mini")
        self.assertEqual(payload["entries"][2]["prompt_mode"], "structured_history")
        self.assertEqual(payload["entries"][2]["frames_per_action"], 3)
        self.assertEqual(payload["entries"][2]["thinking_mode"], "off")
        self.assertEqual(payload["entries"][2]["history_clips"], 2)
        self.assertEqual(payload["entries"][2]["non_zero_reward_clips"], 1)


if __name__ == "__main__":
    unittest.main()
