from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
candidate = str(SRC_DIR)
if candidate not in sys.path:
    sys.path.insert(0, candidate)

from ataribench.batch_run import (
    BatchJobSpec,
    build_jobs_from_config,
    classify_error_output,
    execute_run,
    expand_run_requests,
    load_yaml_config,
    _run_subprocess,
)


class BatchRunTests(unittest.TestCase):
    def test_build_jobs_from_config_merges_common_and_setting_specific_values(self) -> None:
        batch_options, jobs = build_jobs_from_config(
            common_settings={
                "max_concurrency_by_company": {
                    "gemini": 1,
                    "openai": 2,
                    "anthropic": 1,
                    "together": 3,
                },
                "games": {
                    "subset": ["assault"],
                    "full": ["all"],
                },
                "max_retries": 8,
                "render_video_fps": 24,
                "retry_backoff_seconds": 12,
                "duration_seconds": 30,
                "max_actions_per_turn": 10,
                "frames_per_action": 3,
                "minimal_logging": "true",
                "context_cache": "true",
            },
            setting_entries=[
                {
                    "model_name": "gemini-2.5-flash",
                    "thinking_mode": "off",
                    "prompt_mode": "structured_history",
                    "text_feedback_mode": "none",
                    "history_clips": 3,
                    "non_zero_reward_clips": 3,
                    "games": "subset",
                    "seed_start": 0,
                    "num_runs": 2,
                },
                {
                    "model_name": "gpt-5.4-mini",
                    "thinking_mode": "none",
                    "games": ["assault", "breakout"],
                    "num_runs": 1,
                    "history_clips": 10,
                    "non_zero_reward_clips": 2,
                    "prompt_mode": "structured_history",
                    "max_actions_per_turn": 3,
                },
                {
                    "model_name": "gemini-2.5-flash",
                    "thinking_mode": "off",
                    "games": "subset",
                    "num_runs": 1,
                    "prompt_mode": "append_only",
                },
            ],
        )

        self.assertEqual(
            batch_options["max_concurrency_by_company"],
            {
                "gemini": 1,
                "openai": 2,
                "anthropic": 1,
                "together": 3,
            },
        )
        self.assertEqual(batch_options["max_retries"], 8)
        self.assertEqual(batch_options["render_video_fps"], 24)
        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0].games, ["assault"])
        self.assertEqual(jobs[0].run_count, 2)
        self.assertEqual(jobs[0].seed_start, 0)
        self.assertEqual(jobs[0].frames_per_action, 3)
        self.assertTrue(jobs[0].minimal_logging)
        self.assertEqual(jobs[0].history_clips, 3)
        self.assertEqual(jobs[0].text_feedback_mode, "none")
        self.assertEqual(jobs[1].games, ["assault", "breakout"])
        self.assertEqual(jobs[1].history_clips, 10)
        self.assertEqual(jobs[1].max_actions_per_turn, 3)
        self.assertEqual(jobs[2].history_clips, -1)
        self.assertEqual(jobs[2].non_zero_reward_clips, -1)
        self.assertTrue(jobs[2].context_cache)

    def test_load_yaml_config_keeps_on_off_as_strings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "thinking_mode: off\nother_thinking_mode: on\n",
                encoding="utf-8",
            )

            payload = load_yaml_config(config_path)

        self.assertEqual(
            payload,
            {
                "thinking_mode": "off",
                "other_thinking_mode": "on",
            },
        )

    def test_expand_run_requests_uses_seed_start_for_each_run(self) -> None:
        jobs = [
            BatchJobSpec(
                model_name="gemini-2.5-flash",
                run_count=3,
                thinking_mode="off",
                label="gemini-2.5-flash",
                games=["breakout"],
                seed_start=0,
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            requests = expand_run_requests(
                jobs=jobs,
                project_dir=PROJECT_DIR,
                base_output_dir=Path(tmpdir) / "runs",
                log_dir=Path(tmpdir) / "logs",
                batch_timestamp="0328_104742",
            )

        self.assertEqual([request.seed for request in requests], [0, 1, 2])
        self.assertEqual(
            [request.run_label for request in requests],
            [
                "0328_104742_run_001",
                "0328_104742_run_002",
                "0328_104742_run_003",
            ],
        )

    def test_expand_run_requests_can_target_canonical_model_root(self) -> None:
        jobs = [
            BatchJobSpec(
                model_name="gpt-5.4-mini",
                run_count=1,
                thinking_mode="off",
                label="gpt-5.4-mini_cfg_005",
                games=["breakout"],
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            requests = expand_run_requests(
                jobs=jobs,
                project_dir=PROJECT_DIR,
                base_output_dir=Path(tmpdir) / "ignored",
                log_dir=Path(tmpdir) / "logs",
                batch_timestamp="0328_104742",
            )

        self.assertEqual(requests[0].output_dir, str(PROJECT_DIR / "runs" / "breakout" / "gpt-5.4-mini"))
        self.assertEqual(requests[0].run_label, "0328_104742_cfg_005_run_001")

    def test_classify_error_output(self) -> None:
        self.assertEqual(
            classify_error_output("google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED"),
            "transient",
        )
        self.assertEqual(
            classify_error_output("OpenAI API error: Rate limit reached for requests"),
            "transient",
        )
        self.assertEqual(
            classify_error_output("httpx.ConnectError: [Errno 65] No route to host"),
            "transient",
        )
        self.assertEqual(
            classify_error_output("httpx.ConnectError: [Errno 8] nodename nor servname provided, or not known"),
            "transient",
        )
        self.assertEqual(
            classify_error_output("httpx.ReadTimeout: timed out"),
            "transient",
        )
        self.assertEqual(
            classify_error_output("Budget 0 is invalid. This model only works in thinking mode."),
            "thinking_required",
        )
        self.assertIsNone(classify_error_output("some unrelated traceback"))

    def test_run_subprocess_targets_main_entrypoint(self) -> None:
        request = BatchJobSpec(
            model_name="gpt-5.4",
            run_count=1,
            thinking_mode="low",
            label="gpt-5.4",
            games=["breakout"],
            prompt_mode="append_only",
            context_cache=True,
            text_feedback_mode="none",
            minimal_logging=True,
        )
        run_request = expand_run_requests(
            jobs=[request],
            project_dir=PROJECT_DIR,
            base_output_dir=Path("/tmp") / "runs",
            log_dir=Path("/tmp") / "logs",
            batch_timestamp="0328_104742",
        )[0]

        with mock.patch("ataribench.batch_run.subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="")
            _run_subprocess(
                request=run_request,
                thinking_mode="low",
            )

        command = run_mock.call_args.kwargs["args"] if "args" in run_mock.call_args.kwargs else run_mock.call_args.args[0]
        cwd = run_mock.call_args.kwargs["cwd"]
        env = run_mock.call_args.kwargs["env"]
        self.assertEqual(command[:3], [sys.executable, "-m", "ataribench.main"])
        self.assertEqual(len(command), 3)
        payload = env["ATARIBENCH_INTERNAL_RUN_REQUEST"]
        self.assertIn('"model": "gpt-5.4"', payload)
        self.assertIn('"frames_per_action": 3', payload)
        self.assertIn('"history_clips": 3', payload)
        self.assertIn('"non_zero_reward_clips": 3', payload)
        self.assertIn('"prompt_mode": "append_only"', payload)
        self.assertIn('"text_feedback_mode": "none"', payload)
        self.assertIn('"context_cache": true', payload)
        self.assertIn('"minimal_logging": true', payload)
        self.assertIn('"run_label": "0328_104742_run_001"', payload)
        self.assertEqual(Path(cwd).resolve(), PROJECT_DIR.resolve())

    def test_execute_run_retries_incomplete_clean_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            request = expand_run_requests(
                jobs=[
                    BatchJobSpec(
                        model_name="gemini-2.5-flash",
                        run_count=1,
                        thinking_mode="off",
                        label="gemini-2.5-flash",
                        games=["breakout"],
                    )
                ],
                project_dir=PROJECT_DIR,
                base_output_dir=Path(tmpdir) / "runs",
                log_dir=Path(tmpdir) / "logs",
            )[0]

            responses = [
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="runs/example/breakout/20260316_000000\nterminated\n",
                ),
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="runs/example/breakout/20260316_000001\nframe_budget\n",
                ),
            ]

            with mock.patch("ataribench.batch_run._run_subprocess", side_effect=responses) as run_mock:
                with mock.patch(
                    "ataribench.batch_run.load_run_summary",
                    side_effect=[
                        {"stop_reason": "terminated", "duration_seconds": 30, "frame_count": 779},
                        {"stop_reason": "frame_budget", "duration_seconds": 30, "frame_count": 901},
                    ],
                ):
                    with mock.patch(
                        "ataribench.batch_run.render_run_video",
                        return_value=Path("runs/example/breakout/20260316_000001/visualization.mp4"),
                    ) as render_mock:
                        result = execute_run(
                            request=request,
                            max_retries=1,
                            retry_backoff_seconds=0.0,
                            render_video_fps=30,
                        )

        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(result.stop_reason, "frame_budget")
        render_mock.assert_called_once()
        self.assertEqual(render_mock.call_args.kwargs["fps"], 30)
        self.assertEqual(
            Path(render_mock.call_args.kwargs["run_dir"]).parts[-4:],
            ("runs", "example", "breakout", "20260316_000001"),
        )
        self.assertEqual(
            result.video_path,
            "runs/example/breakout/20260316_000001/visualization.mp4",
        )
        self.assertEqual(
            result.summary["video_path"],
            "runs/example/breakout/20260316_000001/visualization.mp4",
        )


if __name__ == "__main__":
    unittest.main()
