from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT / "AtariBench"
candidate = str(PROJECT_DIR)
if candidate not in sys.path:
    sys.path.insert(0, candidate)

import main


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

            with mock.patch("main.validate_model_thinking_mode"):
                with mock.patch("main.get_game_spec", return_value=mock.Mock(fps=30)):
                    with mock.patch("main.resolve_output_layout", return_value=(Path(tmpdir) / "runs", True)):
                        with mock.patch("main.build_model_client", return_value=object()):
                            with mock.patch("main.PipelineRunner", side_effect=build_runner):
                                with mock.patch(
                                    "main.render_run_video",
                                    return_value=run_dir / "visualization.mp4",
                                ):
                                    with mock.patch("main.apply_minimal_logging_policy") as prune_mock:
                                        with mock.patch("main.uses_canonical_game_storage", return_value=False):
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

            with mock.patch("main.validate_model_thinking_mode"):
                with mock.patch("main.get_game_spec", return_value=mock.Mock(fps=30)):
                    with mock.patch("main.resolve_output_layout", return_value=(Path(tmpdir) / "runs", True)):
                        with mock.patch("main.build_model_client", return_value=object()):
                            with mock.patch("main.PipelineRunner", side_effect=build_runner):
                                with mock.patch("main.render_run_video", return_value=run_dir / "visualization.mp4"):
                                    with mock.patch("main.uses_canonical_game_storage", return_value=False):
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


if __name__ == "__main__":
    unittest.main()
