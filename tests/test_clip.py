from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT / "AtariBench"
candidate = str(PROJECT_DIR)
if candidate not in sys.path:
    sys.path.insert(0, candidate)

from core.clip import parse_model_response
from games import get_game_spec


class ParseModelResponseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = get_game_spec("breakout")

    def test_parses_valid_response(self) -> None:
        parsed = parse_model_response(
            "thought: track the ball\nmove: [right, right, noop]",
            game_spec=self.spec,
            max_actions=10,
        )
        self.assertTrue(parsed.is_valid)
        self.assertEqual(parsed.action_strings, ["right", "right", "noop"])
        self.assertEqual(parsed.action_ids, [2, 2, 0])

    def test_strips_code_fences(self) -> None:
        parsed = parse_model_response(
            "```text\nthought: launch first\nmove: [start]\n```",
            game_spec=self.spec,
            max_actions=10,
        )
        self.assertTrue(parsed.is_valid)
        self.assertEqual(parsed.action_strings, ["start"])

    def test_rejects_missing_move_line(self) -> None:
        parsed = parse_model_response(
            "thought: track the ball",
            game_spec=self.spec,
            max_actions=10,
        )
        self.assertIn("Missing required 'move:' section.", parsed.errors)

    def test_rejects_invalid_action(self) -> None:
        parsed = parse_model_response(
            "thought: move\nmove: [jump]",
            game_spec=self.spec,
            max_actions=10,
        )
        self.assertIn("Unknown action: jump", parsed.errors)

    def test_rejects_empty_action_list(self) -> None:
        parsed = parse_model_response(
            "thought: move\nmove: []",
            game_spec=self.spec,
            max_actions=10,
        )
        self.assertIn("Move list must contain at least 1 action.", parsed.errors)

    def test_rejects_too_many_actions(self) -> None:
        parsed = parse_model_response(
            "thought: move\nmove: [noop, noop, noop]",
            game_spec=self.spec,
            max_actions=2,
        )
        self.assertIn(
            "Move list must contain at most 2 actions; got 3.",
            parsed.errors,
        )


if __name__ == "__main__":
    unittest.main()
