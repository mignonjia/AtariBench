"""Environment helpers for the Atari runner."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class EnvInfo:
    """Normalized environment metadata used by the runner."""

    lives: int | None
    episode_frame_number: int | None
    frame_number: int | None


def extract_env_info(info: dict[str, Any] | None) -> EnvInfo:
    """Extract the stable fields used by the pipeline."""

    payload = info or {}
    return EnvInfo(
        lives=_as_int(payload.get("lives")),
        episode_frame_number=_as_int(payload.get("episode_frame_number")),
        frame_number=_as_int(payload.get("frame_number")),
    )


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def detect_life_loss(previous_lives: int | None, current_lives: int | None) -> int:
    """Return how many lives were lost between two observations."""

    if previous_lives is None or current_lives is None:
        return 0
    if current_lives >= previous_lives:
        return 0
    return previous_lives - current_lives


def create_env(env_id: str, seed: int | None = None):
    """Create a headless Atari environment with exact frame accounting."""

    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
    env = gym.make(
        env_id,
        render_mode="rgb_array",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    return env


def capture_frame(env: Any, observation: Any):
    """Prefer rendered RGB frames, with observation as fallback."""

    render = getattr(env, "render", None)
    if callable(render):
        frame = render()
        if frame is not None:
            return frame
    return observation
