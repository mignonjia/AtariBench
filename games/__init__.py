"""Game registry and environment helpers."""

from .env import EnvInfo, capture_frame, create_env, detect_life_loss, extract_env_info
from .registry import GameSpec, get_game_spec, list_game_keys

__all__ = [
    "EnvInfo",
    "GameSpec",
    "capture_frame",
    "create_env",
    "detect_life_loss",
    "extract_env_info",
    "get_game_spec",
    "list_game_keys",
]
