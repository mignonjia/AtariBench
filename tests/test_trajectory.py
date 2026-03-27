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
from core.trajectory import ActionRecord, Trajectory


def fake_frame_writer(frame, path: Path) -> None:
    path.write_bytes(b"fake-png-data")


class TrajectoryTests(unittest.TestCase):
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
            )

            self.assertTrue(Path(first_frame.frame_path).exists())
            self.assertTrue(trajectory.turns_path.exists())
            self.assertTrue(trajectory.summary_path.exists())
            self.assertEqual(summary["frame_count"], 2)
            self.assertEqual(summary["stop_reason"], "frame_budget")
            self.assertEqual(summary["duration_seconds"], 30)
            self.assertEqual(summary["model_name"], "gemini-2.5-flash")
            self.assertEqual(summary["thinking_budget"], 0)
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
            saved_turn = json.loads(trajectory.turns_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertFalse(saved_turn["new_game_started"])


if __name__ == "__main__":
    unittest.main()
