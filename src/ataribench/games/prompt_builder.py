"""Prompt assembly for Atari Gemini turns."""

from __future__ import annotations

import dataclasses

from .prompts import common_prompt, termination
from .registry import GameSpec

from ..core.trajectory import ActionRecord, Trajectory, TurnRecord


@dataclasses.dataclass(frozen=True)
class PromptMessage:
    """One structured prompt message with ordered images."""

    role: str
    text: str
    image_paths: list[str]


@dataclasses.dataclass(frozen=True)
class PromptPackage:
    """Prompt text plus the ordered images referenced by IMG_HOLDER."""

    text: str
    image_paths: list[str]
    messages: list[PromptMessage] | None = None


def format_time(frame_index: int, fps: int) -> str:
    """Convert a frame index into a stable prompt timestamp."""

    return f"{frame_index / float(fps):.2f}s"


def build_prompt(
    game_spec: GameSpec,
    trajectory: Trajectory,
    history_clips: int,
    non_zero_reward_clips: int,
    duration_seconds: int,
    prompt_mode: str = "structured_history",
    max_actions_per_turn: int = 10,
    text_feedback_mode: str = "full",
) -> PromptPackage:
    """Render the single-turn prompt from the current trajectory."""

    _validate_text_feedback_mode(text_feedback_mode)

    if prompt_mode == "append_only":
        return _build_append_only_prompt(
            game_spec=game_spec,
            trajectory=trajectory,
            duration_seconds=duration_seconds,
            max_actions_per_turn=max_actions_per_turn,
            text_feedback_mode=text_feedback_mode,
        )
    if prompt_mode != "structured_history":
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")

    return _build_structured_history_prompt(
        game_spec=game_spec,
        trajectory=trajectory,
        history_clips=history_clips,
        non_zero_reward_clips=non_zero_reward_clips,
        duration_seconds=duration_seconds,
        max_actions_per_turn=max_actions_per_turn,
        text_feedback_mode=text_feedback_mode,
    )


def _build_structured_history_prompt(
    game_spec: GameSpec,
    trajectory: Trajectory,
    history_clips: int,
    non_zero_reward_clips: int,
    duration_seconds: int,
    max_actions_per_turn: int,
    text_feedback_mode: str,
) -> PromptPackage:
    """Render the curated-history prompt."""

    current_frame = trajectory.latest_frame()
    recent_turns = trajectory.turn_records[-history_clips:]
    reward_turns = [
        turn for turn in trajectory.turn_records if _belongs_in_reward_history(turn)
    ]
    reward_turns = reward_turns[-non_zero_reward_clips:]

    recent_clip_texts: list[str] = []
    reward_clip_texts: list[str] = []
    image_paths: list[str] = []

    for turn in reward_turns:
        clip_text, clip_images = build_clip_prompt(
            turn,
            game_spec,
            text_feedback_mode=text_feedback_mode,
        )
        reward_clip_texts.append(clip_text)
        image_paths.extend(clip_images)

    for turn in recent_turns:
        clip_text, clip_images = build_clip_prompt(
            turn,
            game_spec,
            text_feedback_mode=text_feedback_mode,
        )
        recent_clip_texts.append(clip_text)
        image_paths.extend(clip_images)

    image_paths.append(current_frame.frame_path)

    reward_history_section = ""
    if reward_clip_texts:
        reward_clips_template = (
            common_prompt.REWARD_CLIPS_TEMPLATE
            if text_feedback_mode == "full"
            else common_prompt.SELECTED_HISTORY_CLIPS_TEMPLATE
        )
        reward_history_section = reward_clips_template.format(
            LIST_OF_CLIPS_TEMPLATE="".join(reward_clip_texts)
        )

    game_prompt = game_spec.render_game_prompt(
        max_actions_per_turn=max_actions_per_turn,
    )
    game_over_prompt = termination.MAX_TIME_PROMPT.format(MAX_TIME=duration_seconds)

    prompt_template = (
        common_prompt.SINGLE_TURN_PROMPT_TEMPLATE
        if text_feedback_mode == "full"
        else common_prompt.SINGLE_TURN_NO_TEXT_FEEDBACK_PROMPT_TEMPLATE
    )
    prompt_text = prompt_template.format(
        GAME_PROMPT=game_prompt,
        GAME_OVER_PROMPT=game_over_prompt,
        LIST_OF_REWARD_CLIPS_TEMPLATE=reward_history_section,
        LIST_OF_CLIPS_TEMPLATE="".join(recent_clip_texts),
        CURRENT_TIME=format_time(current_frame.local_frame_index, game_spec.fps),
        ACTION_OUTPUT_INSTRUCTION=common_prompt.action_output_instruction(
            max_actions_per_turn,
        ),
    )
    if (
        text_feedback_mode == "full"
        and recent_turns
        and recent_turns[-1].new_game_started
    ):
        prompt_text += (
            "\nContext update: The previous episode ended early, and the current "
            "frame is the start of a new game. Keep using the recent clips and "
            "reward history as context, but treat the current ball, paddle, and "
            "lives as reset.\n"
        )
    return PromptPackage(text=prompt_text, image_paths=image_paths)


def _build_append_only_prompt(
    game_spec: GameSpec,
    trajectory: Trajectory,
    duration_seconds: int,
    max_actions_per_turn: int,
    text_feedback_mode: str,
) -> PromptPackage:
    """Render the append-only chronological transcript prompt."""

    current_frame = trajectory.latest_frame()
    messages: list[PromptMessage] = []
    game_prompt = game_spec.render_game_prompt(
        max_actions_per_turn=max_actions_per_turn,
    )
    game_over_prompt = termination.MAX_TIME_PROMPT.format(MAX_TIME=duration_seconds)
    action_output_instruction = common_prompt.action_output_instruction(
        max_actions_per_turn,
    )
    initial_frame = trajectory.frame_records[0]
    messages.append(
        PromptMessage(
            role="user",
            text=(
                common_prompt.APPEND_ONLY_HEADER_TEMPLATE
                if text_feedback_mode == "full"
                else common_prompt.APPEND_ONLY_NO_TEXT_FEEDBACK_HEADER_TEMPLATE
            ).format(
                GAME_PROMPT=game_prompt,
                GAME_OVER_PROMPT=game_over_prompt,
            )
            + common_prompt.APPEND_ONLY_INITIAL_USER_TEMPLATE.format(
                CURRENT_TIME=format_time(initial_frame.local_frame_index, game_spec.fps),
                ACTION_OUTPUT_INSTRUCTION=action_output_instruction,
            ),
            image_paths=[initial_frame.frame_path],
        )
    )

    for turn in trajectory.turn_records:
        messages.append(
            PromptMessage(
                role="assistant",
                text=turn.raw_response,
                image_paths=[],
            )
        )
        clip_text, clip_images = build_clip_prompt(
            turn,
            game_spec,
            text_feedback_mode=text_feedback_mode,
        )
        updated_user_template = (
            common_prompt.APPEND_ONLY_UPDATED_USER_TEMPLATE
            if text_feedback_mode == "full"
            else common_prompt.APPEND_ONLY_NO_TEXT_FEEDBACK_UPDATED_USER_TEMPLATE
        )
        messages.append(
            PromptMessage(
                role="user",
                text=updated_user_template.format(
                    CLIP_TEMPLATE=clip_text,
                    ACTION_OUTPUT_INSTRUCTION=action_output_instruction,
                ),
                image_paths=clip_images,
            )
        )

    prompt_text, image_paths = serialize_prompt_messages(messages)
    return PromptPackage(text=prompt_text, image_paths=image_paths, messages=messages)


def serialize_prompt_messages(messages: list[PromptMessage]) -> tuple[str, list[str]]:
    """Serialize structured messages into a text transcript for saved artifacts."""

    blocks: list[str] = []
    image_paths: list[str] = []
    for message in messages:
        block = [f"<{message.role}>", message.text, f"</{message.role}>"]
        blocks.append("\n".join(block))
        image_paths.extend(message.image_paths)
    return "\n\n".join(blocks), image_paths


def _belongs_in_reward_history(turn: TurnRecord) -> bool:
    if turn.reward_delta != 0.0:
        return True
    return any(action.lost_life for action in turn.action_records)


def build_clip_prompt(
    turn: TurnRecord,
    game_spec: GameSpec,
    text_feedback_mode: str = "full",
) -> tuple[str, list[str]]:
    """Convert one turn into the common clip template."""

    _validate_text_feedback_mode(text_feedback_mode)

    state_strings: list[str] = []
    image_paths = [turn.start_frame_path]

    for action in turn.action_records:
        state_strings.append(
            _build_state_reward_prompt(
                action,
                game_spec,
                text_feedback_mode=text_feedback_mode,
            )
        )
        image_paths.append(action.end_frame_path)

    clip_template = (
        common_prompt.SINGLE_TURN_CLIP_TEMPLATE
        if text_feedback_mode == "full"
        else common_prompt.SINGLE_TURN_NO_TEXT_FEEDBACK_CLIP_TEMPLATE
    )
    clip_text = clip_template.format(
        START_TIME=format_time(turn.start_frame_index, game_spec.fps),
        END_TIME=format_time(turn.executed_frame_end, game_spec.fps),
        ACTIONS_STR=str(turn.planned_action_strings),
        LIST_OF_STATE_REWARD_TEMPLATE="".join(state_strings),
    )
    if text_feedback_mode == "full" and turn.new_game_started:
        clip_text += (
            "Context: This clip ended with the episode terminating, and the run "
            "continued from a freshly reset game afterward.\n"
        )
    return clip_text, image_paths


def _build_state_reward_prompt(
    action: ActionRecord,
    game_spec: GameSpec,
    text_feedback_mode: str = "full",
) -> str:
    state_text = common_prompt.STATE_REWARD_TEMPLATE.format(
        TIME=format_time(action.end_frame_index, game_spec.fps)
    )
    if text_feedback_mode == "none":
        return state_text
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


def _validate_text_feedback_mode(text_feedback_mode: str) -> None:
    if text_feedback_mode not in {"full", "none"}:
        raise ValueError(
            "text_feedback_mode must be 'full' or 'none'."
        )
