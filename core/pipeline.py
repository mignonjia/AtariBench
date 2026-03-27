"""Main model-driven Atari loop."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable

try:
    from ..games.env import capture_frame, create_env, detect_life_loss, extract_env_info
    from ..games.prompt_builder import build_prompt
    from ..games.registry import GameSpec
    from ..llm import describe_effective_thinking_mode
    from .clip import ParsedClipResponse, parse_model_response
    from .trajectory import ActionRecord, Trajectory
except ImportError:  # Running from inside the AtariBench folder.
    from games.env import capture_frame, create_env, detect_life_loss, extract_env_info
    from games.prompt_builder import build_prompt
    from games.registry import GameSpec
    from llm import describe_effective_thinking_mode
    from core.clip import ParsedClipResponse, parse_model_response
    from core.trajectory import ActionRecord, Trajectory

EnvFactory = Callable[[], Any]


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for one pipeline run."""

    duration_seconds: int = 30
    max_actions_per_turn: int = 10
    history_clips: int = 3
    non_zero_reward_clips: int = 3
    prompt_mode: str = "structured_history"
    model_name: str = "gemini-2.5-flash"
    thinking_mode: str = "default"
    seed: int | None = None
    output_dir: str | Path = "runs"
    nest_output_by_game: bool = True


class PipelineRunner:
    """Run a model policy against one Atari game."""

    def __init__(
        self,
        game_spec: GameSpec,
        model_client: Any,
        config: PipelineConfig,
        env_factory: EnvFactory | None = None,
        frame_writer=None,
    ):
        self.game_spec = game_spec
        self.model_client = model_client
        self.config = config
        self.env_factory = env_factory or (
            lambda: create_env(self.game_spec.env_id, seed=self.config.seed)
        )
        self.frame_writer = frame_writer

    @property
    def frame_budget(self) -> int:
        return self.config.duration_seconds * self.game_spec.fps

    def run(self) -> dict[str, Any]:
        """Execute the pipeline and return the summary."""

        env = self.env_factory()
        thinking_metadata = describe_effective_thinking_mode(
            model_name=self.config.model_name,
            thinking_mode=self.config.thinking_mode,
        )
        trajectory = Trajectory(
            base_output_dir=self.config.output_dir,
            game_key=self.game_spec.game_key,
            frame_writer=self.frame_writer,
            include_game_key=self.config.nest_output_by_game,
        )
        total_reward = 0.0
        total_lost_lives = 0
        stop_reason = "unknown"

        try:
            _, current_info = self._reset_and_record_initial_frame(
                env,
                trajectory,
                local_frame_index=0,
            )
            initial_env_frame = current_info.frame_number or 0
            previous_lives = current_info.lives

            while True:
                latest_frame = trajectory.latest_frame()
                if latest_frame.local_frame_index >= self.frame_budget:
                    stop_reason = "frame_budget"
                    break

                prompt_package = build_prompt(
                    game_spec=self.game_spec,
                    trajectory=trajectory,
                    history_clips=self.config.history_clips,
                    non_zero_reward_clips=self.config.non_zero_reward_clips,
                    duration_seconds=self.config.duration_seconds,
                    prompt_mode=self.config.prompt_mode,
                )
                raw_response = self.model_client.generate_turn(
                    prompt_text=prompt_package.text,
                    image_paths=prompt_package.image_paths,
                    model_name=self.config.model_name,
                    thinking_mode=self.config.thinking_mode,
                    prompt_messages=prompt_package.messages,
                )
                parsed_response = self._parse_response_or_fallback(raw_response)

                action_records: list[ActionRecord] = []
                turn_reward = 0.0
                turn_start_frame = latest_frame.local_frame_index
                turn_start_path = latest_frame.frame_path

                should_stop = False
                should_reset_episode = False
                for action_name, action_id in zip(
                    parsed_response.action_strings,
                    parsed_response.action_ids,
                ):
                    start_frame_index = trajectory.latest_frame().local_frame_index
                    action_reward = 0.0
                    action_lost_life = False
                    end_info_payload: dict[str, Any] = {}

                    for _ in range(self.game_spec.frames_per_action):
                        observation, reward, terminated, truncated, info = env.step(
                            action_id
                        )
                        total_reward += float(reward)
                        turn_reward += float(reward)
                        action_reward += float(reward)

                        normalized_info = extract_env_info(info)
                        life_loss = detect_life_loss(previous_lives, normalized_info.lives)
                        if life_loss:
                            total_lost_lives += life_loss
                            action_lost_life = True
                        previous_lives = normalized_info.lives

                        next_frame_index = trajectory.latest_frame().local_frame_index + 1
                        frame = capture_frame(env, observation)
                        trajectory.record_frame(
                            frame=frame,
                            reward=float(reward),
                            info=info,
                            local_frame_index=next_frame_index,
                        )
                        end_info_payload = dict(info)

                        env_frames_elapsed = (
                            (normalized_info.frame_number or initial_env_frame)
                            - initial_env_frame
                        )
                        local_frames_elapsed = trajectory.latest_frame().local_frame_index
                        if terminated:
                            stop_reason = "frame_budget"
                            should_stop = True
                            should_reset_episode = True
                        elif truncated:
                            stop_reason = "frame_budget"
                            should_stop = True
                            should_reset_episode = True
                        elif env_frames_elapsed >= self.frame_budget:
                            stop_reason = "frame_budget"
                            should_stop = True
                        elif local_frames_elapsed >= self.frame_budget:
                            stop_reason = "frame_budget"
                            should_stop = True
                        if should_stop:
                            break

                    action_records.append(
                        ActionRecord(
                            action_name=action_name,
                            action_id=action_id,
                            start_frame_index=start_frame_index,
                            end_frame_index=trajectory.latest_frame().local_frame_index,
                            reward_delta=action_reward,
                            lost_life=action_lost_life,
                            end_frame_path=trajectory.latest_frame().frame_path,
                            end_info=end_info_payload,
                        )
                    )
                    if should_stop:
                        break

                trajectory.record_turn(
                    prompt_text=prompt_package.text,
                    raw_response=raw_response,
                    parsed_response=parsed_response,
                    referenced_image_paths=prompt_package.image_paths,
                    start_frame_index=turn_start_frame,
                    start_frame_path=turn_start_path,
                    executed_frame_end=trajectory.latest_frame().local_frame_index,
                    reward_delta=turn_reward,
                    action_records=action_records,
                    new_game_started=False,
                )

                if should_stop and should_reset_episode:
                    next_local_frame_index = trajectory.latest_frame().local_frame_index + 1
                    if next_local_frame_index > self.frame_budget:
                        stop_reason = "frame_budget"
                        break

                    _, current_info = self._reset_and_record_initial_frame(
                        env,
                        trajectory,
                        local_frame_index=next_local_frame_index,
                    )
                    previous_lives = current_info.lives
                    initial_env_frame = current_info.frame_number or 0
                    trajectory.replace_last_turn(
                        dataclasses.replace(
                            trajectory.turn_records[-1],
                            new_game_started=True,
                        )
                    )
                    continue

                if should_stop:
                    break

                if trajectory.latest_frame().local_frame_index >= self.frame_budget:
                    stop_reason = "frame_budget"
                    break

            return trajectory.finalize(
                stop_reason=stop_reason,
                total_reward=total_reward,
                total_lost_lives=total_lost_lives,
                duration_seconds=self.config.duration_seconds,
                model_name=self.config.model_name,
                thinking_mode=thinking_metadata["thinking_mode"],
                thinking_budget=thinking_metadata["thinking_budget"],
                thinking_level=thinking_metadata["thinking_level"],
                history_clips=self.config.history_clips,
                non_zero_reward_clips=self.config.non_zero_reward_clips,
                prompt_mode=self.config.prompt_mode,
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    def _reset_and_record_initial_frame(
        self,
        env: Any,
        trajectory: Trajectory,
        local_frame_index: int,
    ):
        try:
            observation, info = env.reset(seed=self.config.seed)
        except TypeError:
            if self.config.seed is None:
                observation, info = env.reset()
            else:
                observation, info = env.reset(seed=self.config.seed)
        frame = capture_frame(env, observation)
        frame_record = trajectory.record_frame(
            frame=frame,
            reward=0.0,
            info=info,
            local_frame_index=local_frame_index,
        )
        return frame_record.local_frame_index, extract_env_info(info)

    def _parse_response(self, raw_response: str) -> ParsedClipResponse:
        return parse_model_response(
            raw_text=raw_response,
            game_spec=self.game_spec,
            max_actions=self.config.max_actions_per_turn,
        )

    def _parse_response_or_fallback(self, raw_response: str) -> ParsedClipResponse:
        parsed_response = self._parse_response(raw_response)
        if not parsed_response.errors:
            return parsed_response

        noop_action_id = self.game_spec.action_map["noop"]
        fallback_thought = (
            "The model response could not be parsed cleanly, so this turn "
            "defaults to a single noop action."
        )
        return ParsedClipResponse(
            raw_text=raw_response,
            thought=fallback_thought,
            action_strings=["noop"],
            action_ids=[noop_action_id],
            errors=list(parsed_response.errors),
        )
