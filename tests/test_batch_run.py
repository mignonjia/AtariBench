from __future__ import annotations

import subprocess
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

from batch_run import (
    BatchJobSpec,
    classify_error_output,
    compute_retry_sleep_seconds,
    execute_run,
    expand_run_requests,
    normalize_run_dir,
    parse_job_spec,
    _run_subprocess,
)


class BatchRunTests(unittest.TestCase):
    def test_parse_job_spec_with_explicit_thinking(self) -> None:
        job = parse_job_spec("gemini-2.5-flash:5:off")
        self.assertEqual(job.model_name, "gemini-2.5-flash")
        self.assertEqual(job.run_count, 5)
        self.assertEqual(job.thinking_mode, "off")
        self.assertEqual(job.label, "gemini-2.5-flash")

    def test_parse_job_spec_defaults_thinking(self) -> None:
        job = parse_job_spec("gemini-2.5-pro:3")
        self.assertEqual(job.thinking_mode, "default")

    def test_expand_run_requests_creates_unique_output_roots(self) -> None:
        jobs = [
            BatchJobSpec(
                model_name="gemini-2.5-flash",
                run_count=2,
                thinking_mode="off",
                label="gemini-2.5-flash",
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            requests = expand_run_requests(
                jobs=jobs,
                base_output_dir=Path(tmpdir) / "runs",
                log_dir=Path(tmpdir) / "logs",
            )

        self.assertEqual(len(requests), 2)
        self.assertTrue(requests[0].output_dir.endswith("run_001"))
        self.assertTrue(requests[1].output_dir.endswith("run_002"))
        self.assertNotEqual(requests[0].log_path, requests[1].log_path)

    def test_expand_run_requests_can_target_canonical_model_root(self) -> None:
        jobs = [
            BatchJobSpec(
                model_name="gpt-5.4-mini",
                run_count=1,
                thinking_mode="off",
                label="gpt-5.4-mini",
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            requests = expand_run_requests(
                jobs=jobs,
                base_output_dir=Path(tmpdir) / "ignored",
                log_dir=Path(tmpdir) / "logs",
                canonical_output_root=Path(tmpdir) / "breakout",
            )

        self.assertEqual(requests[0].output_dir, str(Path(tmpdir) / "breakout" / "gpt-5.4-mini"))

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

    def test_compute_retry_sleep_seconds_uses_exponential_growth(self) -> None:
        with mock.patch("batch_run.random.uniform", return_value=0.0):
            self.assertEqual(
                compute_retry_sleep_seconds(attempt=1, base_backoff_seconds=5.0),
                5.0,
            )
            self.assertEqual(
                compute_retry_sleep_seconds(attempt=2, base_backoff_seconds=5.0),
                10.0,
            )
            self.assertEqual(
                compute_retry_sleep_seconds(attempt=3, base_backoff_seconds=5.0),
                20.0,
            )

    def test_run_subprocess_targets_main_entrypoint(self) -> None:
        request = BatchJobSpec(
            model_name="gpt-5.4",
            run_count=1,
            thinking_mode="low",
            label="gpt-5.4",
        )
        run_request = expand_run_requests(
            jobs=[request],
            base_output_dir=Path("/tmp") / "runs",
            log_dir=Path("/tmp") / "logs",
        )[0]

        with mock.patch("batch_run.subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="")
            _run_subprocess(
                request=run_request,
                game="breakout",
                duration_seconds=30,
                max_actions_per_turn=10,
                history_clips=3,
                non_zero_reward_clips=3,
                prompt_mode="append_only",
                seed=None,
                thinking_mode="low",
            )

        command = run_mock.call_args.kwargs["args"] if "args" in run_mock.call_args.kwargs else run_mock.call_args.args[0]
        cwd = run_mock.call_args.kwargs["cwd"]
        self.assertTrue(command[1].endswith("main.py"))
        self.assertIn("--model", command)
        self.assertEqual(command[command.index("--model") + 1], "gpt-5.4")
        self.assertEqual(command[command.index("--history-clips") + 1], "3")
        self.assertEqual(command[command.index("--non-zero-reward-clips") + 1], "3")
        self.assertEqual(command[command.index("--prompt-mode") + 1], "append_only")
        self.assertEqual(Path(cwd).resolve(), PROJECT_DIR.resolve())

    def test_normalize_run_dir_resolves_subprocess_relative_paths(self) -> None:
        normalized = normalize_run_dir("runs/example/breakout/20260324_000000")
        self.assertTrue(normalized.endswith("/runs/example/breakout/20260324_000000"))
        self.assertTrue(normalized.startswith("/"))

    def test_execute_run_falls_back_from_off_to_minimal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            request = expand_run_requests(
                jobs=[
                    BatchJobSpec(
                        model_name="gemini-2.5-pro",
                        run_count=1,
                        thinking_mode="off",
                        label="gemini-2.5-pro",
                    )
                ],
                base_output_dir=Path(tmpdir) / "runs",
                log_dir=Path(tmpdir) / "logs",
            )[0]

            responses = [
                subprocess.CompletedProcess(
                    args=[],
                    returncode=1,
                    stdout="Budget 0 is invalid. This model only works in thinking mode.",
                ),
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="runs/example/breakout/20260316_000000\nframe_budget\n",
                ),
            ]

            with mock.patch(
                "batch_run._run_subprocess",
                side_effect=responses,
            ) as run_mock:
                with mock.patch(
                    "batch_run.load_run_summary",
                    return_value={"stop_reason": "frame_budget"},
                ):
                    with mock.patch(
                        "batch_run.render_run_video",
                        return_value=Path("runs/example/breakout/20260316_000000/visualization.mp4"),
                    ):
                        result = execute_run(
                            request=request,
                            game="breakout",
                            duration_seconds=30,
                            max_actions_per_turn=10,
                            history_clips=3,
                            non_zero_reward_clips=3,
                            prompt_mode="structured_history",
                            seed=None,
                            fallback_thinking="minimal",
                            max_retries=1,
                            retry_backoff_seconds=0.0,
                            render_video_fps=30,
                        )

        self.assertTrue(result.success)
        self.assertEqual(result.final_thinking_mode, "minimal")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(run_mock.call_count, 2)
        self.assertTrue(str(result.video_path).endswith("visualization.mp4"))

    def test_execute_run_retries_incomplete_clean_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            request = expand_run_requests(
                jobs=[
                    BatchJobSpec(
                        model_name="gemini-2.5-flash",
                        run_count=1,
                        thinking_mode="off",
                        label="gemini-2.5-flash",
                    )
                ],
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

            with mock.patch("batch_run._run_subprocess", side_effect=responses) as run_mock:
                with mock.patch(
                    "batch_run.load_run_summary",
                    side_effect=[
                        {"stop_reason": "terminated", "duration_seconds": 30, "frame_count": 779},
                        {"stop_reason": "frame_budget", "duration_seconds": 30, "frame_count": 901},
                    ],
                ):
                    with mock.patch(
                        "batch_run.render_run_video",
                        return_value=Path("runs/example/breakout/20260316_000001/visualization.mp4"),
                    ):
                        result = execute_run(
                            request=request,
                            game="breakout",
                            duration_seconds=30,
                            max_actions_per_turn=10,
                            history_clips=3,
                            non_zero_reward_clips=3,
                            prompt_mode="structured_history",
                            seed=None,
                            fallback_thinking="minimal",
                            max_retries=1,
                            retry_backoff_seconds=0.0,
                            render_video_fps=30,
                        )

        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(result.stop_reason, "frame_budget")

    def test_execute_run_retries_non_transient_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            request = expand_run_requests(
                jobs=[
                    BatchJobSpec(
                        model_name="gemini-2.5-flash",
                        run_count=1,
                        thinking_mode="off",
                        label="gemini-2.5-flash",
                    )
                ],
                base_output_dir=Path(tmpdir) / "runs",
                log_dir=Path(tmpdir) / "logs",
            )[0]

            responses = [
                subprocess.CompletedProcess(
                    args=[],
                    returncode=1,
                    stdout="some unrelated traceback",
                ),
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="runs/example/breakout/20260316_000002\nframe_budget\n",
                ),
            ]

            with mock.patch("batch_run._run_subprocess", side_effect=responses) as run_mock:
                with mock.patch(
                    "batch_run.load_run_summary",
                    return_value={"stop_reason": "frame_budget", "duration_seconds": 30, "frame_count": 901},
                ):
                    with mock.patch(
                        "batch_run.render_run_video",
                        return_value=Path("runs/example/breakout/20260316_000002/visualization.mp4"),
                    ):
                        result = execute_run(
                            request=request,
                            game="breakout",
                            duration_seconds=30,
                            max_actions_per_turn=10,
                            history_clips=3,
                            non_zero_reward_clips=3,
                            prompt_mode="structured_history",
                            seed=None,
                            fallback_thinking="minimal",
                            max_retries=1,
                            retry_backoff_seconds=0.0,
                            render_video_fps=30,
                        )

        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(run_mock.call_count, 2)

    def test_execute_run_marks_incomplete_run_failed_after_retry_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            request = expand_run_requests(
                jobs=[
                    BatchJobSpec(
                        model_name="gemini-2.5-flash",
                        run_count=1,
                        thinking_mode="off",
                        label="gemini-2.5-flash",
                    )
                ],
                base_output_dir=Path(tmpdir) / "runs",
                log_dir=Path(tmpdir) / "logs",
            )[0]

            with mock.patch(
                "batch_run._run_subprocess",
                return_value=subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="runs/example/breakout/20260316_000003\nterminated\n",
                ),
            ) as run_mock:
                with mock.patch(
                    "batch_run.load_run_summary",
                    return_value={"stop_reason": "terminated", "duration_seconds": 30, "frame_count": 728},
                ):
                    with mock.patch(
                        "batch_run.render_run_video",
                        return_value=Path("runs/example/breakout/20260316_000003/visualization.mp4"),
                    ):
                        result = execute_run(
                            request=request,
                            game="breakout",
                            duration_seconds=30,
                            max_actions_per_turn=10,
                            history_clips=3,
                            non_zero_reward_clips=3,
                            prompt_mode="structured_history",
                            seed=None,
                            fallback_thinking="minimal",
                            max_retries=1,
                            retry_backoff_seconds=0.0,
                            render_video_fps=30,
                        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "incomplete_run")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(run_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
