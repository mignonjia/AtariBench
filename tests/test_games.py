from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT / "AtariBench"
candidate = str(PROJECT_DIR)
if candidate not in sys.path:
    sys.path.insert(0, candidate)

from games import get_game_spec, list_game_keys, list_game_selection_keys, resolve_game_selection


class GameSpecTests(unittest.TestCase):
    def test_all_prompt_games_are_registered(self) -> None:
        game_keys = list_game_keys()
        self.assertIn("assault", game_keys)
        self.assertIn("breakout", game_keys)
        self.assertIn("seaquest", game_keys)
        self.assertIn("time_pilot", game_keys)
        self.assertIn("air_raid", game_keys)

    def test_assault_is_registered(self) -> None:
        self.assertIn("assault", list_game_keys())

    def test_assault_spec_fields(self) -> None:
        spec = get_game_spec("assault")
        self.assertEqual(spec.env_id, "ALE/Assault-v5")
        self.assertEqual(spec.action_map["noop"], 0)
        self.assertEqual(spec.action_map["fire up"], 2)
        self.assertEqual(spec.action_map["move right"], 3)
        self.assertEqual(spec.action_map["move left"], 4)
        self.assertEqual(spec.action_map["fire right"], 5)
        self.assertEqual(spec.action_map["fire left"], 6)
        self.assertEqual(spec.fps, 30)
        self.assertEqual(spec.frames_per_action, 3)

    def test_breakout_is_registered(self) -> None:
        self.assertIn("breakout", list_game_keys())

    def test_breakout_spec_fields(self) -> None:
        spec = get_game_spec("breakout")
        self.assertEqual(spec.env_id, "ALE/Breakout-v5")
        self.assertEqual(spec.action_map["noop"], 0)
        self.assertEqual(spec.action_map["start"], 1)
        self.assertEqual(spec.action_map["right"], 2)
        self.assertEqual(spec.action_map["left"], 3)
        self.assertEqual(spec.fps, 30)
        self.assertEqual(spec.frames_per_action, 3)

    def test_qbert_env_id_is_normalized_correctly(self) -> None:
        spec = get_game_spec("qbert")
        self.assertEqual(spec.env_id, "ALE/Qbert-v5")

    def test_game_selection_presets_are_registered(self) -> None:
        self.assertIn("selected", list_game_selection_keys())
        self.assertIn("full", list_game_selection_keys())

    def test_selected_game_selection_contains_breakout_and_assault(self) -> None:
        self.assertEqual(resolve_game_selection("selected"), ["breakout", "assault"])

    def test_full_game_selection_expands_to_all_registered_games(self) -> None:
        self.assertEqual(resolve_game_selection("full"), list_game_keys())


if __name__ == "__main__":
    unittest.main()
