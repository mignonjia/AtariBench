"""Prompt assembly for Atari Gemini turns."""

from __future__ import annotations

import dataclasses

from .prompts import common_prompt, termination
from .registry import GameSpec

try:
    from ..core.trajectory import ActionRecord, Trajectory, TurnRecord
except ImportError:  # Running from inside the AtariBench folder.
    from core.trajectory import ActionRecord, Trajectory, TurnRecord


@dataclasses.dataclass(frozen=True)
class PromptPackage:
    """Prompt text plus the ordered images referenced by IMG_HOLDER."""

    text: str
    image_paths: list[str]


def format_time(frame_index: int, fps: int) -> str:
    """Convert a frame index into a stable prompt timestamp."""

    return f"{frame_index / float(fps):.2f}s"


def build_prompt(
    game_spec: GameSpec,
    trajectory: Trajectory,
    history_clips: int,
    duration_seconds: int,
) -> PromptPackage:
    """Render the single-turn prompt from the current trajectory."""

    current_frame = trajectory.latest_frame()
    recent_turns = trajectory.turn_records[-history_clips:]
    reward_turns = [
        turn for turn in trajectory.turn_records if _belongs_in_reward_history(turn)
    ]
    reward_turns = reward_turns[-history_clips:]

    recent_clip_texts: list[str] = []
    reward_clip_texts: list[str] = []
    image_paths: list[str] = []

    for turn in reward_turns:
        clip_text, clip_images = build_clip_prompt(turn, game_spec)
        reward_clip_texts.append(clip_text)
        image_paths.extend(clip_images)

    for turn in recent_turns:
        clip_text, clip_images = build_clip_prompt(turn, game_spec)
        recent_clip_texts.append(clip_text)
        image_paths.extend(clip_images)

    image_paths.append(current_frame.frame_path)

    reward_history_section = ""
    if reward_clip_texts:
        reward_history_section = common_prompt.REWARD_CLIPS_TEMPLATE.format(
            LIST_OF_CLIPS_TEMPLATE="".join(reward_clip_texts)
        )

    game_prompt = game_spec.game_prompt.format(FPS_SPECIFIC_PROMPT=game_spec.fps_prompt)
    game_over_prompt = termination.MAX_TIME_PROMPT.format(MAX_TIME=duration_seconds)

    prompt_text = common_prompt.SINGLE_TURN_PROMPT_TEMPLATE.format(
        GAME_PROMPT=game_prompt,
        GAME_OVER_PROMPT=game_over_prompt,
        LIST_OF_REWARD_CLIPS_TEMPLATE=reward_history_section,
        LIST_OF_CLIPS_TEMPLATE="".join(recent_clip_texts),
        CURRENT_TIME=format_time(current_frame.local_frame_index, game_spec.fps),
    )
    if recent_turns and recent_turns[-1].new_game_started:
        prompt_text += (
            "\nContext update: The previous episode ended early, and the current "
            "frame is the start of a new game. Keep using the recent clips and "
            "reward history as context, but treat the current ball, paddle, and "
            "lives as reset.\n"
        )
    return PromptPackage(text=prompt_text, image_paths=image_paths)


def _belongs_in_reward_history(turn: TurnRecord) -> bool:
    if turn.reward_delta != 0.0:
        return True
    return any(action.lost_life for action in turn.action_records)


def build_clip_prompt(turn: TurnRecord, game_spec: GameSpec) -> tuple[str, list[str]]:
    """Convert one turn into the common clip template."""

    state_strings: list[str] = []
    image_paths = [turn.start_frame_path]

    for action in turn.action_records:
        state_strings.append(_build_state_reward_prompt(action, game_spec))
        image_paths.append(action.end_frame_path)

    clip_text = common_prompt.SINGLE_TURN_CLIP_TEMPLATE.format(
        START_TIME=format_time(turn.start_frame_index, game_spec.fps),
        END_TIME=format_time(turn.executed_frame_end, game_spec.fps),
        ACTIONS_STR=str(turn.planned_action_strings),
        LIST_OF_STATE_REWARD_TEMPLATE="".join(state_strings),
    )
    if turn.new_game_started:
        clip_text += (
            "Context: This clip ended with the episode terminating, and the run "
            "continued from a freshly reset game afterward.\n"
        )
    return clip_text, image_paths


def _build_state_reward_prompt(action: ActionRecord, game_spec: GameSpec) -> str:
    state_text = common_prompt.STATE_REWARD_TEMPLATE.format(
        TIME=format_time(action.end_frame_index, game_spec.fps)
    )
    if action.reward_delta > 0:
        state_text += "Feedback: You received a positive score!\n"
    if action.reward_delta < 0:
        state_text += "Feedback: You received a negative score!\n"
    if action.lost_life:
        state_text += (
            "Feedback: A life was lost and a new life has begun. This does not "
            "directly reduce your score, but the restart consumes time.\n"
        )
    return state_text
