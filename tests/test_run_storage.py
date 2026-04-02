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
                        "duration_seconds": 10,
                        "model_name": "gpt-5.4-mini",
                        "prompt_mode": "append_only",
                        "context_cache": True,
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
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "total_tokens": 120,
                        "thinking_tokens": 15,
                        "cached_input_tokens": 60,
                        "token_usage_reported_turns": 10,
                        "token_usage_missing_turns": 0,
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
                        "context_cache": True,
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
                        "input_tokens": 120,
                        "output_tokens": 30,
                        "total_tokens": 150,
                        "thinking_tokens": 18,
                        "cached_input_tokens": 90,
                        "token_usage_reported_turns": 14,
                        "token_usage_missing_turns": 0,
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
                        "context_cache": False,
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
                        "input_tokens": 80,
                        "output_tokens": 40,
                        "total_tokens": 120,
                        "thinking_tokens": 8,
                        "cached_input_tokens": 15,
                        "token_usage_reported_turns": 4,
                        "token_usage_missing_turns": 0,
                        "stop_reason": "frame_budget",
                    }
                ),
                encoding="utf-8",
            )

            summary_path = update_game_model_summary(tmpdir, "assault")
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload_30s = json.loads((root / "model_summary_30s.json").read_text(encoding="utf-8"))

        model_summaries = payload["models"]["gpt-5.4-mini"]
        append_only_summary = next(
            summary for summary in model_summaries if summary["thinking_mode"] == "low"
        )
        structured_history_summary = next(
            summary for summary in model_summaries if summary["thinking_mode"] == "off"
        )
        self.assertEqual(payload["game"], "assault")
        self.assertEqual(payload["run_filter"], "all_successful")
        self.assertEqual(len(model_summaries), 2)
        self.assertEqual(append_only_summary["run_count"], 2)
        self.assertEqual(append_only_summary["avg_total_reward"], 4.0)
        self.assertAlmostEqual(append_only_summary["stderr_total_reward"], 2.0)
        self.assertEqual(append_only_summary["avg_total_lost_lives"], 2.0)
        self.assertAlmostEqual(append_only_summary["stderr_total_lost_lives"], 1.0)
        self.assertEqual(append_only_summary["avg_turn_count"], 12.0)
        self.assertEqual(append_only_summary["avg_frame_count"], 110.0)
        self.assertEqual(append_only_summary["avg_input_tokens"], 110.0)
        self.assertEqual(append_only_summary["avg_output_tokens"], 25.0)
        self.assertEqual(append_only_summary["avg_total_tokens"], 135.0)
        self.assertEqual(append_only_summary["avg_thinking_tokens"], 16.5)
        self.assertEqual(append_only_summary["avg_cached_input_tokens"], 75.0)
        self.assertEqual(append_only_summary["latest_timestamp"], "20260324_110000")
        self.assertEqual(append_only_summary["latest_total_tokens"], 150.0)
        self.assertEqual(append_only_summary["latest_thinking_tokens"], 18.0)
        self.assertEqual(append_only_summary["latest_cached_input_tokens"], 90.0)
        self.assertEqual(append_only_summary["thinking_mode"], "low")
        self.assertEqual(append_only_summary["prompt_mode"], "append_only")
        self.assertTrue(append_only_summary["context_cache"])
        self.assertEqual(append_only_summary["frames_per_action"], 3)
        self.assertEqual(append_only_summary["thinking_level"], "low")
        self.assertIsNone(append_only_summary["thinking_budget"])
        self.assertEqual(append_only_summary["history_clips"], -1)
        self.assertEqual(append_only_summary["non_zero_reward_clips"], -1)
        self.assertEqual(structured_history_summary["run_count"], 1)
        self.assertEqual(structured_history_summary["avg_total_reward"], 9.0)
        self.assertEqual(structured_history_summary["stderr_total_reward"], 0.0)
        self.assertEqual(structured_history_summary["stderr_total_lost_lives"], 0.0)
        self.assertEqual(structured_history_summary["prompt_mode"], "structured_history")
        self.assertFalse(structured_history_summary["context_cache"])
        self.assertEqual(structured_history_summary["frames_per_action"], 3)
        self.assertEqual(structured_history_summary["thinking_mode"], "off")
        self.assertEqual(structured_history_summary["thinking_budget"], 0)
        self.assertEqual(structured_history_summary["history_clips"], 1)
        self.assertEqual(structured_history_summary["non_zero_reward_clips"], 1)
        self.assertEqual(structured_history_summary["avg_total_tokens"], 120.0)
        self.assertEqual(structured_history_summary["avg_thinking_tokens"], 8.0)
        self.assertEqual(structured_history_summary["avg_cached_input_tokens"], 15.0)

        payload_30s_models = payload_30s["models"]["gpt-5.4-mini"]
        append_only_summary_30s = next(
            summary for summary in payload_30s_models if summary["thinking_mode"] == "low"
        )
        self.assertEqual(payload_30s["run_filter"], "full_30s")
        self.assertEqual(len(payload_30s_models), 2)
        self.assertEqual(append_only_summary_30s["run_count"], 1)
        self.assertEqual(append_only_summary_30s["avg_total_reward"], 6.0)
        self.assertEqual(append_only_summary_30s["stderr_total_reward"], 0.0)
        self.assertEqual(append_only_summary_30s["stderr_total_lost_lives"], 0.0)
        self.assertEqual(append_only_summary_30s["latest_timestamp"], "20260324_110000")

    def test_update_game_model_summary_ignores_failed_runs(self) -> None:
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
                        "context_cache": False,
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
                        "input_tokens": 50,
                        "output_tokens": 10,
                        "total_tokens": 60,
                        "thinking_tokens": 6,
                        "cached_input_tokens": 8,
                        "token_usage_reported_turns": 10,
                        "token_usage_missing_turns": 0,
                        "stop_reason": "lost_lives",
                    }
                ),
                encoding="utf-8",
            )

            summary_path = update_game_model_summary(tmpdir, "assault")
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload_30s = json.loads((root / "model_summary_30s.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["models"], {})
        self.assertEqual(payload_30s["models"], {})

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
                        "context_cache": False,
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
                        "input_tokens": 90,
                        "output_tokens": 30,
                        "total_tokens": 120,
                        "thinking_tokens": 10,
                        "cached_input_tokens": 20,
                        "token_usage_reported_turns": 11,
                        "token_usage_missing_turns": 0,
                        "stop_reason": "frame_budget",
                    }
                ),
                encoding="utf-8",
            )
            (breakout_run_a / "summary.json").write_text(
                json.dumps(
                    {
                        "duration_seconds": 10,
                        "model_name": "gpt-5.4-mini",
                        "prompt_mode": "append_only",
                        "context_cache": True,
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
                        "input_tokens": 140,
                        "output_tokens": 50,
                        "total_tokens": 190,
                        "thinking_tokens": 25,
                        "cached_input_tokens": 100,
                        "token_usage_reported_turns": 17,
                        "token_usage_missing_turns": 0,
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
                        "context_cache": False,
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
                        "input_tokens": 110,
                        "output_tokens": 45,
                        "total_tokens": 155,
                        "thinking_tokens": 9,
                        "cached_input_tokens": 11,
                        "token_usage_reported_turns": 12,
                        "token_usage_missing_turns": 0,
                        "stop_reason": "frame_budget",
                    }
                ),
                encoding="utf-8",
            )

            update_game_model_summary(tmpdir, "assault")
            update_game_model_summary(tmpdir, "breakout")
            summary_path = update_runs_model_summary(tmpdir)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload_30s = json.loads(
                ((Path(tmpdir) / "runs") / "model_summary_30s.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(payload["entries"]), 3)
        self.assertEqual(payload["run_filter"], "all_successful")
        self.assertEqual(payload["entries"][0]["game"], "assault")
        self.assertEqual(payload["entries"][0]["model_name"], "gemini-2.5-flash")
        self.assertEqual(payload["entries"][0]["prompt_mode"], "structured_history")
        self.assertFalse(payload["entries"][0]["context_cache"])
        self.assertEqual(payload["entries"][0]["frames_per_action"], 3)
        self.assertEqual(payload["entries"][0]["thinking_mode"], "off")
        self.assertEqual(payload["entries"][0]["history_clips"], 3)
        self.assertEqual(payload["entries"][0]["avg_total_tokens"], 120.0)
        self.assertEqual(payload["entries"][0]["stderr_total_reward"], 0.0)
        self.assertEqual(payload["entries"][0]["stderr_total_lost_lives"], 0.0)
        self.assertEqual(payload["entries"][0]["avg_thinking_tokens"], 10.0)
        self.assertEqual(payload["entries"][0]["avg_cached_input_tokens"], 20.0)
        self.assertEqual(payload["entries"][1]["game"], "breakout")
        self.assertEqual(payload["entries"][1]["model_name"], "gpt-5.4-mini")
        self.assertEqual(payload["entries"][1]["prompt_mode"], "append_only")
        self.assertTrue(payload["entries"][1]["context_cache"])
        self.assertEqual(payload["entries"][1]["frames_per_action"], 3)
        self.assertEqual(payload["entries"][1]["thinking_mode"], "low")
        self.assertEqual(payload["entries"][1]["history_clips"], -1)
        self.assertEqual(payload["entries"][1]["non_zero_reward_clips"], -1)
        self.assertEqual(payload["entries"][1]["stderr_total_reward"], 0.0)
        self.assertEqual(payload["entries"][1]["stderr_total_lost_lives"], 0.0)
        self.assertEqual(payload["entries"][1]["latest_total_tokens"], 190.0)
        self.assertEqual(payload["entries"][1]["latest_thinking_tokens"], 25.0)
        self.assertEqual(payload["entries"][1]["latest_cached_input_tokens"], 100.0)
        self.assertEqual(payload["entries"][2]["game"], "breakout")
        self.assertEqual(payload["entries"][2]["model_name"], "gpt-5.4-mini")
        self.assertEqual(payload["entries"][2]["prompt_mode"], "structured_history")
        self.assertFalse(payload["entries"][2]["context_cache"])
        self.assertEqual(payload["entries"][2]["frames_per_action"], 3)
        self.assertEqual(payload["entries"][2]["thinking_mode"], "off")
        self.assertEqual(payload["entries"][2]["history_clips"], 2)
        self.assertEqual(payload["entries"][2]["non_zero_reward_clips"], 1)
        self.assertEqual(payload["entries"][2]["stderr_total_reward"], 0.0)
        self.assertEqual(payload["entries"][2]["stderr_total_lost_lives"], 0.0)
        self.assertEqual(payload["entries"][2]["avg_total_tokens"], 155.0)
        self.assertEqual(payload["entries"][2]["avg_thinking_tokens"], 9.0)
        self.assertEqual(payload["entries"][2]["avg_cached_input_tokens"], 11.0)
        self.assertEqual(len(payload_30s["entries"]), 2)
        self.assertEqual(payload_30s["run_filter"], "full_30s")
        self.assertEqual(payload_30s["entries"][0]["game"], "assault")
        self.assertEqual(payload_30s["entries"][1]["game"], "breakout")
        self.assertEqual(payload_30s["entries"][1]["prompt_mode"], "structured_history")


if __name__ == "__main__":
    unittest.main()
