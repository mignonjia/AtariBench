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
    RunRequest,
    _find_next_schedulable_request_index,
    _format_run_start_line,
    _resolve_executor_worker_count,
    build_jobs_from_config,
    classify_error_output,
    compute_retry_sleep_seconds,
    execute_run,
    expand_run_requests,
    load_yaml_config,
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

    def test_build_jobs_from_config_merges_common_and_setting_specific_values(self) -> None:
        batch_options, jobs = build_jobs_from_config(
            common_settings={
                "max_concurrency_by_company": {
                    "gemini": 1,
                    "openai": 2,
                    "anthropic": 1,
                },
                "games": {
                    "selected": ["assault"],
                    "full": ["all"],
                },
                "max_retries": 8,
                "render_video_fps": 24,
                "retry_backoff_seconds": 12,
                "duration_seconds": 30,
                "max_actions_per_turn": 10,
            },
            setting_entries=[
                {
                    "model_name": "gemini-2.5-flash",
                    "thinking_mode": "off",
                    "prompt_mode": "structured_history",
                    "history_clips": 3,
                    "non_zero_reward_clips": 3,
                    "games": "selected",
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
                },
                {
                    "model_name": "gemini-2.5-flash",
                    "thinking_mode": "off",
                    "games": "selected",
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
            },
        )
        self.assertEqual(batch_options["max_retries"], 8)
        self.assertEqual(batch_options["render_video_fps"], 24)
        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0].games, ["assault"])
        self.assertEqual(jobs[0].run_count, 2)
        self.assertEqual(jobs[0].seed_start, 0)
        self.assertEqual(jobs[0].history_clips, 3)
        self.assertEqual(jobs[1].games, ["assault", "breakout"])
        self.assertEqual(jobs[1].history_clips, 10)
        self.assertEqual(jobs[2].history_clips, -1)
        self.assertEqual(jobs[2].non_zero_reward_clips, -1)

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

    def test_resolve_executor_worker_count_uses_company_limits(self) -> None:
        self.assertEqual(
            _resolve_executor_worker_count(
                max_concurrency=2,
                max_concurrency_by_company={
                    "gemini": 2,
                    "openai": 1,
                    "anthropic": 1,
                },
            ),
            4,
        )

    def test_find_next_schedulable_request_index_skips_blocked_company(self) -> None:
        pending_requests = [
            RunRequest(
                game="assault",
                job_label="gemini-2.5-flash_cfg_001",
                run_index=3,
                total_num_runs=3,
                model_name="gemini-2.5-flash",
                company="gemini",
                thinking_mode="off",
                games_label="selected",
                duration_seconds=30,
                max_actions_per_turn=10,
                history_clips=3,
                non_zero_reward_clips=3,
                prompt_mode="structured_history",
                seed=2,
                output_dir="runs/assault/gemini-2.5-flash_cfg_001",
                log_path="logs/assault_gemini.log",
            ),
            RunRequest(
                game="assault",
                job_label="gpt-5.4-mini_cfg_002",
                run_index=1,
                total_num_runs=3,
                model_name="gpt-5.4-mini",
                company="openai",
                thinking_mode="none",
                games_label="selected",
                duration_seconds=30,
                max_actions_per_turn=10,
                history_clips=3,
                non_zero_reward_clips=3,
                prompt_mode="structured_history",
                seed=0,
                output_dir="runs/assault/gpt-5.4-mini_cfg_002",
                log_path="logs/assault_gpt.log",
            ),
        ]
        self.assertEqual(
            _find_next_schedulable_request_index(
                pending_requests=pending_requests,
                active_counts={"gemini": 2, "openai": 0, "anthropic": 0},
                company_limits={"gemini": 2, "openai": 2, "anthropic": 1},
            ),
            1,
        )

    def test_format_run_start_line_is_flat_and_includes_run_indices(self) -> None:
        line = _format_run_start_line(
            RunRequest(
                game="breakout",
                job_label="gemini-2.5-flash_cfg_001",
                run_index=2,
                total_num_runs=3,
                model_name="gemini-2.5-flash",
                company="gemini",
                thinking_mode="off",
                games_label="selected",
                duration_seconds=30,
                max_actions_per_turn=10,
                history_clips=3,
                non_zero_reward_clips=3,
                prompt_mode="structured_history",
                seed=1,
                output_dir="runs/breakout/gemini-2.5-flash_cfg_001",
                log_path="logs/breakout_gemini-2.5-flash_cfg_001_run_002.log",
            )
        )

        self.assertNotIn("\n", line)
        self.assertIn("model_name=gemini-2.5-flash", line)
        self.assertIn("games=selected", line)
        self.assertIn("selected_game=breakout", line)
        self.assertIn("seed=1", line)
        self.assertIn("current_num_run=2", line)
        self.assertIn("total_num_runs=3", line)
        self.assertIn("output_dir=runs/breakout/gemini-2.5-flash_cfg_001", line)

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

    def test_expand_run_requests_creates_unique_output_roots(self) -> None:
        jobs = [
            BatchJobSpec(
                model_name="gemini-2.5-flash",
                run_count=2,
                thinking_mode="off",
                label="gemini-2.5-flash",
                games=["custom_game"],
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

    def test_expand_run_requests_supports_multiple_games(self) -> None:
        jobs = [
            BatchJobSpec(
                model_name="gemini-2.5-flash",
                run_count=1,
                thinking_mode="off",
                label="gemini-2.5-flash",
                games=["breakout", "assault"],
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

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].game, "breakout")
        self.assertEqual(requests[1].game, "assault")
        self.assertEqual(requests[0].company, "gemini")
        self.assertEqual(requests[1].company, "gemini")
        self.assertIn("breakout_gemini-2.5-flash_run_001.log", requests[0].log_path)
        self.assertIn("assault_gemini-2.5-flash_run_001.log", requests[1].log_path)

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
            games=["breakout"],
            prompt_mode="append_only",
        )
        run_request = expand_run_requests(
            jobs=[request],
            project_dir=PROJECT_DIR,
            base_output_dir=Path("/tmp") / "runs",
            log_dir=Path("/tmp") / "logs",
            batch_timestamp="0328_104742",
        )[0]

        with mock.patch("batch_run.subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="")
            _run_subprocess(
                request=run_request,
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
        self.assertEqual(command[command.index("--run-label") + 1], "0328_104742_run_001")
        self.assertEqual(Path(cwd).resolve(), PROJECT_DIR.resolve())

    def test_normalize_run_dir_resolves_subprocess_relative_paths(self) -> None:
        normalized = normalize_run_dir("runs/example/breakout/20260324_000000")
        self.assertTrue(normalized.endswith("/runs/example/breakout/20260324_000000"))
        self.assertTrue(normalized.startswith("/"))

    def test_execute_run_reports_thinking_required_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            request = expand_run_requests(
                jobs=[
                    BatchJobSpec(
                        model_name="gemini-2.5-pro",
                        run_count=1,
                        thinking_mode="off",
                        label="gemini-2.5-pro",
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
                    returncode=1,
                    stdout="Budget 0 is invalid. This model only works in thinking mode.",
                ),
            ]

            with mock.patch(
                "batch_run._run_subprocess",
                side_effect=responses,
            ) as run_mock:
                result = execute_run(
                    request=request,
                    max_retries=1,
                    retry_backoff_seconds=0.0,
                    render_video_fps=30,
                )

        self.assertFalse(result.success)
        self.assertEqual(result.game, "breakout")
        self.assertEqual(result.final_thinking_mode, "off")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.error_type, "thinking_required")
        self.assertEqual(run_mock.call_count, 1)

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
                        games=["breakout"],
                    )
                ],
                project_dir=PROJECT_DIR,
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
