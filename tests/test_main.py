from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import json
import os

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
candidate = str(SRC_DIR)
if candidate not in sys.path:
    sys.path.insert(0, candidate)

from ataribench import main


class MainTests(unittest.TestCase):
    def test_main_passes_minimal_logging_and_prunes_after_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "breakout" / "0328_104742"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text("{}", encoding="utf-8")
            runner_instance = mock.Mock()
            runner_instance.run.return_value = {
                "run_dir": str(run_dir),
                "stop_reason": "frame_budget",
            }
            captured_config = {}

            def build_runner(*, game_spec, model_client, config):
                del game_spec, model_client
                captured_config["config"] = config
                return runner_instance

            with mock.patch("ataribench.main.validate_model_thinking_mode"):
                with mock.patch("ataribench.main.get_game_spec", return_value=mock.Mock(fps=30)):
                    with mock.patch("ataribench.main.resolve_output_layout", return_value=(Path(tmpdir) / "runs", True)):
                        with mock.patch("ataribench.main.build_model_client", return_value=object()):
                            with mock.patch("ataribench.main.PipelineRunner", side_effect=build_runner):
                                with mock.patch(
                                    "ataribench.main.render_run_video",
                                    return_value=run_dir / "visualization.mp4",
                                ):
                                    with mock.patch("ataribench.main.apply_minimal_logging_policy") as prune_mock:
                                        with mock.patch("ataribench.main.uses_canonical_game_storage", return_value=False):
                                            exit_code = main.main(["--game", "breakout", "--minimal-logging"])

            self.assertEqual(exit_code, 0)
            self.assertTrue(captured_config["config"].minimal_logging)
            prune_mock.assert_called_once_with(str(run_dir))

    def test_main_only_enables_context_cache_for_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "breakout" / "0328_104742"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text("{}", encoding="utf-8")
            runner_instance = mock.Mock()
            runner_instance.run.return_value = {
                "run_dir": str(run_dir),
                "stop_reason": "frame_budget",
            }
            captured_configs: list[object] = []

            def build_runner(*, game_spec, model_client, config):
                del game_spec, model_client
                captured_configs.append(config)
                return runner_instance

            with mock.patch("ataribench.main.validate_model_thinking_mode"):
                with mock.patch("ataribench.main.get_game_spec", return_value=mock.Mock(fps=30)):
                    with mock.patch("ataribench.main.resolve_output_layout", return_value=(Path(tmpdir) / "runs", True)):
                        with mock.patch("ataribench.main.build_model_client", return_value=object()):
                            with mock.patch("ataribench.main.PipelineRunner", side_effect=build_runner):
                                with mock.patch("ataribench.main.render_run_video", return_value=run_dir / "visualization.mp4"):
                                    with mock.patch("ataribench.main.uses_canonical_game_storage", return_value=False):
                                        main.main(["--game", "breakout", "--context-cache"])
                                        main.main(
                                            [
                                                "--game",
                                                "breakout",
                                                "--prompt-mode",
                                                "append_only",
                                                "--context-cache",
                                            ]
                                        )

        self.assertFalse(captured_configs[0].context_cache)
        self.assertTrue(captured_configs[1].context_cache)

    def test_internal_batch_request_skips_rendering_and_summary_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "breakout" / "0328_104742"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text("{}", encoding="utf-8")
            runner_instance = mock.Mock()
            runner_instance.run.return_value = {
                "run_dir": str(run_dir),
                "stop_reason": "frame_budget",
            }
            payload = {
                "game": "breakout",
                "model": "gemini-2.5-flash",
                "thinking": "off",
                "output_dir": str(Path(tmpdir) / "runs"),
                "minimal_logging": True,
            }

            with mock.patch.dict(os.environ, {main._INTERNAL_REQUEST_ENV: json.dumps(payload)}):
                with mock.patch("ataribench.main.validate_model_thinking_mode"):
                    with mock.patch("ataribench.main.get_game_spec", return_value=mock.Mock(fps=30)):
                        with mock.patch("ataribench.main.resolve_output_layout", return_value=(Path(tmpdir) / "runs", True)):
                            with mock.patch("ataribench.main.build_model_client", return_value=object()):
                                with mock.patch("ataribench.main.PipelineRunner", return_value=runner_instance):
                                    with mock.patch("ataribench.main.render_run_video") as render_mock:
                                        with mock.patch("ataribench.main.apply_minimal_logging_policy") as prune_mock:
                                            with mock.patch("ataribench.main.update_game_model_summary") as summary_mock:
                                                exit_code = main.main()

        self.assertEqual(exit_code, 0)
        render_mock.assert_not_called()
        prune_mock.assert_not_called()
        summary_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
