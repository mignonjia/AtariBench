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

from core.clip import ParsedClipResponse
from core.trajectory import ActionRecord, Trajectory, apply_minimal_logging_policy


def fake_frame_writer(frame, path: Path) -> None:
    path.write_bytes(b"fake-png-data")


class TrajectoryTests(unittest.TestCase):
    def test_run_dir_uses_label_and_collision_suffix_without_microseconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Trajectory(
                base_output_dir=tmpdir,
                game_key="breakout",
                frame_writer=fake_frame_writer,
                run_label="0328_104742_cfg_005_run_001",
            )
            second = Trajectory(
                base_output_dir=tmpdir,
                game_key="breakout",
                frame_writer=fake_frame_writer,
                run_label="0328_104742_cfg_005_run_001",
            )

            self.assertEqual(first.run_dir.name, "0328_104742_cfg_005_run_001")
            self.assertEqual(second.run_dir.name, "0328_104742_cfg_005_run_001_2")
            self.assertNotEqual(first.run_dir, second.run_dir)

    def test_persists_frames_turns_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory = Trajectory(
                base_output_dir=tmpdir,
                game_key="breakout",
                frame_writer=fake_frame_writer,
            )
            first_frame = trajectory.record_frame(
                frame="frame0",
                reward=0.0,
                info={
                    "lives": 5,
                    "episode_frame_number": 0,
                    "frame_number": 0,
                },
                local_frame_index=0,
            )
            second_frame = trajectory.record_frame(
                frame="frame1",
                reward=1.0,
                info={
                    "lives": 5,
                    "episode_frame_number": 3,
                    "frame_number": 3,
                },
                local_frame_index=1,
            )
            parsed = ParsedClipResponse(
                raw_text="thought: go\nmove: [right]",
                thought="go",
                action_strings=["right"],
                action_ids=[2],
                errors=[],
            )
            trajectory.record_turn(
                prompt_text="before\nIMG_HOLDER\nafter\nIMG_HOLDER\nend",
                raw_response=parsed.raw_text,
                parsed_response=parsed,
                referenced_image_paths=[first_frame.frame_path, second_frame.frame_path],
                input_tokens=14,
                output_tokens=6,
                total_tokens=20,
                thinking_tokens=4,
                cached_input_tokens=12,
                start_frame_index=0,
                start_frame_path=first_frame.frame_path,
                executed_frame_end=1,
                reward_delta=1.0,
                action_records=[
                    ActionRecord(
                        action_name="right",
                        action_id=2,
                        start_frame_index=0,
                        end_frame_index=1,
                        reward_delta=1.0,
                        lost_life=False,
                        end_frame_path=second_frame.frame_path,
                        end_info={"frame_number": 3},
                    )
                ],
            )
            summary = trajectory.finalize(
                stop_reason="frame_budget",
                total_reward=1.0,
                total_lost_lives=0,
                duration_seconds=30,
                model_name="gemini-2.5-flash",
                thinking_mode="off",
                thinking_budget=0,
                thinking_level=None,
                history_clips=3,
                non_zero_reward_clips=3,
                prompt_mode="append_only",
                context_cache=True,
                input_tokens=14,
                output_tokens=6,
                total_tokens=20,
                thinking_tokens=4,
                cached_input_tokens=12,
                token_usage_reported_turns=1,
                token_usage_missing_turns=0,
            )

            self.assertTrue(Path(first_frame.frame_path).exists())
            self.assertTrue(trajectory.turns_path.exists())
            self.assertTrue(trajectory.summary_path.exists())
            self.assertEqual(summary["frame_count"], 2)
            self.assertEqual(summary["stop_reason"], "frame_budget")
            self.assertEqual(summary["duration_seconds"], 30)
            self.assertEqual(summary["model_name"], "gemini-2.5-flash")
            self.assertEqual(summary["thinking_budget"], 0)
            self.assertEqual(summary["history_clips"], -1)
            self.assertEqual(summary["non_zero_reward_clips"], -1)
            self.assertEqual(summary["prompt_mode"], "append_only")
            self.assertTrue(summary["context_cache"])
            self.assertEqual(summary["input_tokens"], 14)
            self.assertEqual(summary["output_tokens"], 6)
            self.assertEqual(summary["total_tokens"], 20)
            self.assertEqual(summary["thinking_tokens"], 4)
            self.assertEqual(summary["cached_input_tokens"], 12)
            self.assertFalse(summary["minimal_logging"])
            prompt_html_path = Path(trajectory.turn_records[0].prompt_html_path)
            self.assertTrue(prompt_html_path.exists())
            prompt_html = prompt_html_path.read_text(encoding="utf-8")
            self.assertIn("IMG_HOLDER #1", prompt_html)
            self.assertIn("frame_000000.png", prompt_html)
            self.assertIn("frame_000001.png", prompt_html)

            saved_summary = json.loads(trajectory.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["total_reward"], 1.0)
            self.assertEqual(saved_summary["last_frame"]["frame_number"], 3)
            self.assertEqual(saved_summary["thinking_mode"], "off")
            self.assertEqual(saved_summary["history_clips"], -1)
            self.assertEqual(saved_summary["non_zero_reward_clips"], -1)
            self.assertEqual(saved_summary["prompt_mode"], "append_only")
            self.assertEqual(saved_summary["token_usage_reported_turns"], 1)
            self.assertEqual(saved_summary["token_usage_missing_turns"], 0)
            saved_turn = json.loads(trajectory.turns_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertFalse(saved_turn["new_game_started"])
            self.assertEqual(saved_turn["input_tokens"], 14)
            self.assertEqual(saved_turn["output_tokens"], 6)
            self.assertEqual(saved_turn["total_tokens"], 20)
            self.assertEqual(saved_turn["thinking_tokens"], 4)
            self.assertEqual(saved_turn["cached_input_tokens"], 12)

    def test_apply_minimal_logging_policy_keeps_only_compact_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "breakout" / "0328_104742"
            (run_dir / "frames").mkdir(parents=True)
            (run_dir / "prompts").mkdir()
            (run_dir / "responses").mkdir()
            (run_dir / "visualization_frames").mkdir()
            (run_dir / "frames" / "frame_000000.png").write_bytes(b"frame")
            (run_dir / "prompts" / "turn_0001.txt").write_text("prompt", encoding="utf-8")
            (run_dir / "responses" / "turn_0001.txt").write_text("response", encoding="utf-8")
            (run_dir / "visualization_frames" / "viz_000000.png").write_bytes(b"viz")
            (run_dir / "summary.json").write_text("{}", encoding="utf-8")
            (run_dir / "turns.jsonl").write_text("{}", encoding="utf-8")
            (run_dir / "visualization.mp4").write_bytes(b"video")
            (run_dir / "extra.txt").write_text("extra", encoding="utf-8")

            apply_minimal_logging_policy(run_dir)

            self.assertEqual(
                sorted(path.name for path in run_dir.iterdir()),
                ["summary.json", "turns.jsonl", "visualization.mp4"],
            )

    def test_prompt_html_renders_chat_transcript_without_raw_role_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory = Trajectory(
                base_output_dir=tmpdir,
                game_key="breakout",
                frame_writer=fake_frame_writer,
            )
            first_frame = trajectory.record_frame(
                frame="frame0",
                reward=0.0,
                info={"lives": 5, "episode_frame_number": 0, "frame_number": 0},
                local_frame_index=0,
            )
            second_frame = trajectory.record_frame(
                frame="frame1",
                reward=1.0,
                info={"lives": 5, "episode_frame_number": 1, "frame_number": 1},
                local_frame_index=1,
            )
            parsed = ParsedClipResponse(
                raw_text="thought: go\nmove: [right]",
                thought="go",
                action_strings=["right"],
                action_ids=[2],
                errors=[],
            )
            trajectory.record_turn(
                prompt_text=(
                    "<user>\nstate\nIMG_HOLDER\n</user>\n\n"
                    "<assistant>\nthought: go\nmove: [right]\n</assistant>\n\n"
                    "<user>\noutcome\nIMG_HOLDER\n</user>"
                ),
                raw_response=parsed.raw_text,
                parsed_response=parsed,
                referenced_image_paths=[first_frame.frame_path, second_frame.frame_path],
                input_tokens=5,
                output_tokens=2,
                total_tokens=7,
                thinking_tokens=1,
                cached_input_tokens=3,
                start_frame_index=0,
                start_frame_path=first_frame.frame_path,
                executed_frame_end=1,
                reward_delta=1.0,
                action_records=[],
            )

            prompt_html = Path(trajectory.turn_records[0].prompt_html_path).read_text(encoding="utf-8")
            self.assertIn('class="row user"', prompt_html)
            self.assertIn('class="row assistant"', prompt_html)
            self.assertNotIn("&lt;user&gt;", prompt_html)
            self.assertNotIn("&lt;assistant&gt;", prompt_html)
            self.assertIn("frame_000000.png", prompt_html)
            self.assertIn("frame_000001.png", prompt_html)


if __name__ == "__main__":
    unittest.main()
