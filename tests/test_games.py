from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
candidate = str(SRC_DIR)
if candidate not in sys.path:
    sys.path.insert(0, candidate)

from ataribench.games import get_game_spec, list_game_keys, list_game_selection_keys, resolve_game_selection
from ataribench.core.trajectory import ActionRecord, FrameRecord, TurnRecord
from ataribench.games.prompt_builder import build_prompt


class GameSpecTests(unittest.TestCase):
    def test_text_feedback_ablation_preserves_images_and_action_history(self) -> None:
        spec = get_game_spec("breakout")
        action = ActionRecord(
            action_name="right",
            action_id=2,
            start_frame_index=0,
            end_frame_index=3,
            reward_delta=1.0,
            lost_life=True,
            end_frame_path="frame_000003.png",
            end_info={},
        )
        turn = TurnRecord(
            turn_index=1,
            prompt_path="",
            prompt_html_path="",
            response_path="",
            prompt_text="",
            raw_response="thought: align\nmove: [right]",
            parsed_thought="align",
            planned_action_strings=["right"],
            planned_action_ids=[2],
            parse_errors=[],
            referenced_image_paths=["frame_000000.png", "frame_000003.png"],
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            thinking_tokens=None,
            cached_input_tokens=None,
            start_frame_index=0,
            start_frame_path="frame_000000.png",
            executed_frame_end=3,
            reward_delta=1.0,
            action_records=[action],
            new_game_started=True,
        )
        current_frame = FrameRecord(
            local_frame_index=4,
            reward=0.0,
            lives=5,
            episode_frame_number=0,
            frame_number=4,
            frame_path="frame_000004.png",
        )
        trajectory = SimpleNamespace(
            turn_records=[turn],
            latest_frame=lambda: current_frame,
        )

        full = build_prompt(spec, trajectory, 1, 1, 30, text_feedback_mode="full")
        no_text = build_prompt(spec, trajectory, 1, 1, 30, text_feedback_mode="none")

        self.assertEqual(full.image_paths, no_text.image_paths)
        self.assertEqual(full.text.count("IMG_HOLDER"), no_text.text.count("IMG_HOLDER"))
        self.assertIn("<actions>['right']</actions>", no_text.text)
        self.assertIn("Feedback: You received a positive score!", full.text)
        self.assertIn("Feedback: A life was lost", full.text)
        self.assertIn("current frame is the start of a new game", full.text)
        self.assertNotIn("Feedback: You received a positive score!", no_text.text)
        self.assertNotIn("Feedback: A life was lost", no_text.text)
        self.assertNotIn("current frame is the start of a new game", no_text.text)
        self.assertNotIn("<non_zero_reward_history>", no_text.text)
        self.assertIn("<earlier_selected_history>", no_text.text)

    def test_all_prompt_games_are_registered(self) -> None:
        game_keys = list_game_keys()
        self.assertIn("assault", game_keys)
        self.assertIn("breakout", game_keys)
        self.assertIn("seaquest", game_keys)
        self.assertIn("time_pilot", game_keys)
        self.assertIn("air_raid", game_keys)

    def test_qbert_env_id_is_normalized_correctly(self) -> None:
        spec = get_game_spec("qbert")
        self.assertEqual(spec.env_id, "ALE/Qbert-v5")

    def test_full_game_selection_expands_to_all_registered_games(self) -> None:
        self.assertEqual(list_game_selection_keys(), ["full"])
        self.assertEqual(resolve_game_selection("full"), list_game_keys())
        self.assertEqual(len(list_game_keys()), 20)

    def test_action_chunk_games_render_bounded_action_prompts(self) -> None:
        for game in ("breakout", "assault", "fishing_derby", "pacman", "air_raid"):
            spec = get_game_spec(game)
            one_action = spec.render_game_prompt(max_actions_per_turn=1)
            self.assertIn("must contain exactly 1 action", one_action)
            self.assertNotIn("shorter plans", one_action.lower())
            self.assertNotIn("fewer actions", one_action.lower())
            example_line = next(
                line for line in one_action.splitlines() if line.startswith("move: [")
            )
            self.assertNotIn(",", example_line)

            three_actions = spec.render_game_prompt(max_actions_per_turn=3)
            self.assertIn("1 to 3 actions", three_actions)
            self.assertIn("0.1 to 0.3 seconds", three_actions)

            five_actions = spec.render_game_prompt(max_actions_per_turn=5)
            self.assertIn("1 to 5 actions", five_actions)
            self.assertIn("0.1 to 0.5 seconds", five_actions)


if __name__ == "__main__":
    unittest.main()
