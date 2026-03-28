from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT / "AtariBench"
candidate = str(PROJECT_DIR)
if candidate not in sys.path:
    sys.path.insert(0, candidate)

from games import get_game_spec
from core.pipeline import PipelineConfig, PipelineRunner


def fake_frame_writer(frame, path: Path) -> None:
    path.write_bytes(b"fake-png-data")


class FakeGeminiClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate_turn(
        self,
        prompt_text: str,
        image_paths: list[str],
        model_name: str,
        thinking_mode: str = "default",
        prompt_messages=None,
    ) -> str:
        self.calls.append(
            {
                "prompt_text": prompt_text,
                "image_paths": list(image_paths),
                "model_name": model_name,
                "thinking_mode": thinking_mode,
                "prompt_messages": prompt_messages,
            }
        )
        if not self.responses:
            raise AssertionError("No more fake Gemini responses configured.")
        return self.responses.pop(0)


class FakeEnv:
    def __init__(self, lives_schedule: dict[int, int] | None = None):
        self.step_count = 0
        self.total_step_count = 0
        self.closed = False
        self.lives_schedule = lives_schedule or {}
        self.reset_count = 0

    def reset(self, seed=None):
        self.reset_count += 1
        self.step_count = 0
        return "obs0", {
            "lives": 5,
            "episode_frame_number": 0,
            "frame_number": self.total_step_count,
        }

    def step(self, action_id: int):
        del action_id
        self.step_count += 1
        self.total_step_count += 1
        lives = self.lives_schedule.get(self.step_count, 5)
        reward = 1.0 if self.step_count % 3 == 0 else 0.0
        info = {
            "lives": lives,
            "episode_frame_number": self.step_count,
            "frame_number": self.total_step_count,
        }
        terminated = False
        truncated = False
        return f"obs{self.step_count}", reward, terminated, truncated, info

    def render(self):
        return f"frame{self.step_count}"

    def close(self):
        self.closed = True


class PipelineRunnerTests(unittest.TestCase):
    def test_actions_expand_to_three_frames_and_stop_at_budget(self) -> None:
        spec = get_game_spec("breakout")
        client = FakeGeminiClient(
            responses=[
                "thought: keep moving\nmove: [right, left, right, left, right, left, right, left, right, left]",
            ]
        )
        env = FakeEnv()
        runner = PipelineRunner(
            game_spec=spec,
            model_client=client,
            config=PipelineConfig(
                duration_seconds=1,
                max_actions_per_turn=10,
                history_clips=2,
                output_dir=tempfile.mkdtemp(),
            ),
            env_factory=lambda: env,
            frame_writer=fake_frame_writer,
        )

        summary = runner.run()

        self.assertEqual(summary["stop_reason"], "frame_budget")
        self.assertEqual(summary["frame_count"], 31)
        self.assertEqual(summary["duration_seconds"], 1)
        self.assertEqual(env.step_count, 30)
        self.assertEqual(summary["model_name"], "gemini-2.5-flash")
        self.assertIsNone(summary["thinking_budget"])
        self.assertEqual(client.calls[0]["thinking_mode"], "default")
        self.assertEqual(summary["history_clips"], 2)
        self.assertEqual(summary["non_zero_reward_clips"], 3)
        self.assertEqual(summary["prompt_mode"], "structured_history")

    def test_life_loss_does_not_stop_the_run(self) -> None:
        spec = get_game_spec("breakout")
        client = FakeGeminiClient(
            responses=["thought: start\nmove: [start, noop, noop, noop, noop, noop, noop, noop, noop, noop]"]
        )
        env = FakeEnv(lives_schedule={4: 4, 5: 4, 6: 4, 7: 3, 8: 3, 9: 3, 10: 2, 11: 2, 12: 2, 13: 1, 14: 1, 15: 1, 16: 0})
        runner = PipelineRunner(
            game_spec=spec,
            model_client=client,
            config=PipelineConfig(
                duration_seconds=1,
                max_actions_per_turn=10,
                history_clips=1,
                output_dir=tempfile.mkdtemp(),
            ),
            env_factory=lambda: env,
            frame_writer=fake_frame_writer,
        )

        summary = runner.run()

        self.assertEqual(summary["stop_reason"], "frame_budget")
        self.assertEqual(summary["total_lost_lives"], 5)

    def test_terminated_episode_resets_and_continues_to_frame_budget(self) -> None:
        class TerminatingEnv(FakeEnv):
            def __init__(self):
                super().__init__(lives_schedule={4: 4, 5: 0})

            def step(self, action_id: int):
                obs, reward, _, truncated, info = super().step(action_id)
                terminated = self.step_count >= 5
                return obs, reward, terminated, truncated, info

        spec = get_game_spec("breakout")
        client = FakeGeminiClient(
            responses=[
                "thought: recover\nmove: [start, noop, noop, noop, noop, noop, noop, noop, noop, noop]",
            ]
            * 10
        )
        env = TerminatingEnv()
        output_dir = tempfile.mkdtemp()
        runner = PipelineRunner(
            game_spec=spec,
            model_client=client,
            config=PipelineConfig(
                duration_seconds=1,
                max_actions_per_turn=10,
                history_clips=2,
                output_dir=output_dir,
            ),
            env_factory=lambda: env,
            frame_writer=fake_frame_writer,
        )

        summary = runner.run()

        self.assertEqual(summary["stop_reason"], "frame_budget")
        self.assertEqual(summary["frame_count"], 31)
        self.assertGreaterEqual(env.reset_count, 2)
        self.assertEqual(env.total_step_count, 25)
        self.assertIn("current frame is the start of a new game", str(client.calls[1]["prompt_text"]))

        run_root = next(Path(output_dir).glob("breakout/*"))
        turns = (run_root / "turns.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertTrue(turns)
        self.assertIn('"new_game_started": true', turns[0])

    def test_prompt_describes_time_budget_and_life_loss_tradeoff(self) -> None:
        spec = get_game_spec("breakout")
        client = FakeGeminiClient(
            responses=[
                "thought: drift\nmove: [noop, noop, noop, noop, noop, noop, noop, noop, noop, noop]",
            ]
            * 30
        )
        env = FakeEnv()
        runner = PipelineRunner(
            game_spec=spec,
            model_client=client,
            config=PipelineConfig(
                duration_seconds=30,
                max_actions_per_turn=10,
                history_clips=1,
                output_dir=tempfile.mkdtemp(),
            ),
            env_factory=lambda: env,
            frame_writer=fake_frame_writer,
        )

        runner.run()

        prompt_text = str(client.calls[0]["prompt_text"])
        self.assertIn("fixed budget of 30 seconds", prompt_text)
        self.assertIn("does not directly reduce your score", prompt_text)

    def test_life_loss_clip_is_included_in_non_zero_reward_history(self) -> None:
        class LifeLossThenContinueEnv(FakeEnv):
            def __init__(self):
                super().__init__(lives_schedule={1: 4, 2: 4, 3: 4})

            def step(self, action_id: int):
                obs, reward, terminated, truncated, info = super().step(action_id)
                if self.total_step_count in {1, 2, 3}:
                    reward = 0.0
                return obs, reward, terminated, truncated, info

        spec = get_game_spec("breakout")
        client = FakeGeminiClient(
            responses=[
                "thought: serve\nmove: [left]",
                "thought: continue\nmove: [noop]",
            ]
            + ["thought: continue\nmove: [noop]"] * 20
        )
        env = LifeLossThenContinueEnv()
        runner = PipelineRunner(
            game_spec=spec,
            model_client=client,
            config=PipelineConfig(
                duration_seconds=1,
                max_actions_per_turn=10,
                history_clips=2,
                output_dir=tempfile.mkdtemp(),
            ),
            env_factory=lambda: env,
            frame_writer=fake_frame_writer,
        )

        runner.run()

        prompt_text = str(client.calls[1]["prompt_text"])
        self.assertIn("<non_zero_reward_history>", prompt_text)
        self.assertIn("A life was lost and a new life has begun", prompt_text)
        self.assertIn("<clip start=\"0.00s\" to end=\"0.10s\">", prompt_text)

    def test_append_only_prompt_accumulates_previous_response_and_outcome(self) -> None:
        spec = get_game_spec("breakout")
        client = FakeGeminiClient(
            responses=[
                "thought: serve now\nmove: [start]",
                "thought: drift right\nmove: [right]",
            ]
            + ["thought: continue\nmove: [noop]"] * 20
        )
        env = FakeEnv()
        runner = PipelineRunner(
            game_spec=spec,
            model_client=client,
            config=PipelineConfig(
                duration_seconds=1,
                max_actions_per_turn=10,
                history_clips=1,
                non_zero_reward_clips=1,
                prompt_mode="append_only",
                output_dir=tempfile.mkdtemp(),
            ),
            env_factory=lambda: env,
            frame_writer=fake_frame_writer,
        )

        summary = runner.run()

        prompt_text = str(client.calls[1]["prompt_text"])
        self.assertEqual(summary["prompt_mode"], "append_only")
        self.assertEqual(summary["history_clips"], -1)
        self.assertEqual(summary["non_zero_reward_clips"], -1)
        self.assertIn("<assistant>", prompt_text)
        self.assertIn("thought: serve now", prompt_text)
        self.assertIn("move: [start]", prompt_text)
        self.assertIn("<user>", prompt_text)
        self.assertIn("Updated states, rewards, and instruction after your actions:", prompt_text)
        self.assertIn("<clip start=\"0.00s\" to end=\"0.10s\">", prompt_text)
        self.assertIn("<states_and_rewards_after_actions>", prompt_text)
        self.assertNotIn("<recent_history>", prompt_text)
        self.assertNotIn("<non_zero_reward_history>", prompt_text)
        prompt_messages = client.calls[1]["prompt_messages"]
        self.assertIsNotNone(prompt_messages)
        self.assertEqual(prompt_messages[0].role, "user")
        self.assertEqual(prompt_messages[1].role, "assistant")
        self.assertEqual(prompt_messages[2].role, "user")

    def test_parse_error_defaults_to_noop_and_persists_raw_response(self) -> None:
        spec = get_game_spec("breakout")
        client = FakeGeminiClient(
            responses=[
                "move: [right]",
                "thought: drift\nmove: [noop, noop, noop, noop, noop, noop, noop, noop, noop, noop]",
            ]
        )
        env = FakeEnv()
        output_dir = tempfile.mkdtemp()
        runner = PipelineRunner(
            game_spec=spec,
            model_client=client,
            config=PipelineConfig(
                duration_seconds=1,
                max_actions_per_turn=10,
                history_clips=1,
                output_dir=output_dir,
            ),
            env_factory=lambda: env,
            frame_writer=fake_frame_writer,
        )

        summary = runner.run()

        self.assertEqual(summary["stop_reason"], "frame_budget")
        self.assertEqual(summary["thinking_mode"], "default")
        run_root = next(Path(output_dir).glob("breakout/*"))
        response_file = run_root / "responses" / "turn_0001.txt"
        self.assertTrue(response_file.exists())
        self.assertEqual(response_file.read_text(encoding="utf-8"), "move: [right]")

        turns_path = run_root / "turns.jsonl"
        turns = turns_path.read_text(encoding="utf-8").splitlines()
        self.assertTrue(turns)
        self.assertIn('"planned_action_strings": ["noop"]', turns[0])
        self.assertIn("Missing required 'thought:' section.", turns[0])

    def test_summary_records_thinking_off_budget(self) -> None:
        spec = get_game_spec("breakout")
        client = FakeGeminiClient(
            responses=[
                "thought: drift\nmove: [noop, noop, noop, noop, noop, noop, noop, noop, noop, noop]",
            ]
        )
        env = FakeEnv()
        runner = PipelineRunner(
            game_spec=spec,
            model_client=client,
            config=PipelineConfig(
                duration_seconds=1,
                max_actions_per_turn=10,
                history_clips=1,
                output_dir=tempfile.mkdtemp(),
                thinking_mode="off",
            ),
            env_factory=lambda: env,
            frame_writer=fake_frame_writer,
        )

        summary = runner.run()

        self.assertEqual(summary["thinking_mode"], "off")
        self.assertEqual(summary["thinking_budget"], 0)
        self.assertIsNone(summary["thinking_level"])
        self.assertEqual(client.calls[0]["thinking_mode"], "off")
        self.assertEqual(summary["history_clips"], 1)
        self.assertEqual(summary["non_zero_reward_clips"], 3)
        self.assertEqual(summary["prompt_mode"], "structured_history")

    def test_summary_records_thinking_low_level(self) -> None:
        spec = get_game_spec("breakout")
        client = FakeGeminiClient(
            responses=[
                "thought: drift\nmove: [noop, noop, noop, noop, noop, noop, noop, noop, noop, noop]",
            ]
        )
        env = FakeEnv()
        runner = PipelineRunner(
            game_spec=spec,
            model_client=client,
            config=PipelineConfig(
                duration_seconds=1,
                max_actions_per_turn=10,
                history_clips=1,
                output_dir=tempfile.mkdtemp(),
                thinking_mode="low",
            ),
            env_factory=lambda: env,
            frame_writer=fake_frame_writer,
        )

        summary = runner.run()

        self.assertEqual(summary["thinking_mode"], "low")
        self.assertEqual(summary["thinking_level"], "low")
        self.assertIsNone(summary["thinking_budget"])
        self.assertEqual(client.calls[0]["thinking_mode"], "low")
        self.assertEqual(summary["history_clips"], 1)
        self.assertEqual(summary["non_zero_reward_clips"], 3)
        self.assertEqual(summary["prompt_mode"], "structured_history")

    def test_summary_preserves_minimal_mode_with_low_level(self) -> None:
        spec = get_game_spec("breakout")
        client = FakeGeminiClient(
            responses=[
                "thought: drift\nmove: [noop, noop, noop, noop, noop, noop, noop, noop, noop, noop]",
            ]
        )
        env = FakeEnv()
        runner = PipelineRunner(
            game_spec=spec,
            model_client=client,
            config=PipelineConfig(
                duration_seconds=1,
                max_actions_per_turn=10,
                history_clips=1,
                output_dir=tempfile.mkdtemp(),
                thinking_mode="minimal",
            ),
            env_factory=lambda: env,
            frame_writer=fake_frame_writer,
        )

        summary = runner.run()

        self.assertEqual(summary["thinking_mode"], "minimal")
        self.assertEqual(summary["thinking_level"], "minimal")
        self.assertIsNone(summary["thinking_budget"])
        self.assertEqual(client.calls[0]["thinking_mode"], "minimal")
        self.assertEqual(summary["history_clips"], 1)
        self.assertEqual(summary["non_zero_reward_clips"], 3)
        self.assertEqual(summary["prompt_mode"], "structured_history")

    def test_non_zero_reward_clips_can_differ_from_history_clips(self) -> None:
        spec = get_game_spec("breakout")
        client = FakeGeminiClient(
            responses=[
                "thought: serve\nmove: [left]",
                "thought: continue\nmove: [noop]",
            ]
            + ["thought: continue\nmove: [noop]"] * 20
        )
        env = FakeEnv()
        runner = PipelineRunner(
            game_spec=spec,
            model_client=client,
            config=PipelineConfig(
                duration_seconds=1,
                max_actions_per_turn=10,
                history_clips=1,
                non_zero_reward_clips=2,
                output_dir=tempfile.mkdtemp(),
            ),
            env_factory=lambda: env,
            frame_writer=fake_frame_writer,
        )

        summary = runner.run()

        self.assertEqual(summary["history_clips"], 1)
        self.assertEqual(summary["non_zero_reward_clips"], 2)
        self.assertEqual(summary["prompt_mode"], "structured_history")


if __name__ == "__main__":
    unittest.main()
