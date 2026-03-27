"""Game registry for the AtariBench runner."""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class GameSpec:
    """Static configuration for one game."""

    game_key: str
    env_id: str
    prompt_module: str
    action_map: dict[str, int]
    fps: int
    frames_per_action: int
    game_prompt: str
    fps_prompt: str


def _load_prompt_module(module_name: str):
    if module_name.startswith("."):
        return importlib.import_module(module_name, package=__package__)
    if module_name.startswith("prompt."):
        module_name = module_name.replace("prompt.", ".prompts.", 1)
        return importlib.import_module(module_name, package=__package__)
    if module_name.startswith("prompts."):
        return importlib.import_module(f".{module_name}", package=__package__)
    return importlib.import_module(module_name)


def _build_game_spec(
    game_key: str,
    env_id: str,
    prompt_module_name: str,
    fps: int = 30,
    frames_per_action: int = 3,
) -> GameSpec:
    prompt_module = _load_prompt_module(prompt_module_name)
    raw_action_map = getattr(prompt_module, "ACTION_MAP")
    normalized_action_map = {
        " ".join(key.strip().lower().split()): value
        for key, value in raw_action_map.items()
    }
    return GameSpec(
        game_key=game_key,
        env_id=env_id,
        prompt_module=prompt_module_name,
        action_map=normalized_action_map,
        fps=fps,
        frames_per_action=frames_per_action,
        game_prompt=getattr(prompt_module, "GAME_PROMPT"),
        fps_prompt=getattr(prompt_module, "FPS_10_PROMPT"),
    )


_HELPER_PROMPT_MODULES = frozenset({"__init__", "common_prompt", "game_clip", "termination"})
_ENV_NAME_OVERRIDES = {
    "qbert": "Qbert",
}


def _game_key_to_env_id(game_key: str) -> str:
    env_name = _ENV_NAME_OVERRIDES.get(
        game_key,
        "".join(part.title() for part in game_key.split("_")),
    )
    return f"ALE/{env_name}-v5"


def _discover_game_specs() -> dict[str, GameSpec]:
    prompts_dir = Path(__file__).resolve().parent / "prompts"
    specs: dict[str, GameSpec] = {}
    for path in sorted(prompts_dir.glob("*.py")):
        game_key = path.stem
        if game_key in _HELPER_PROMPT_MODULES:
            continue
        specs[game_key] = _build_game_spec(
            game_key=game_key,
            env_id=_game_key_to_env_id(game_key),
            prompt_module_name=f".prompts.{game_key}",
        )
    return specs


_GAME_SPECS = _discover_game_specs()


def get_game_spec(game_key: str) -> GameSpec:
    """Retrieve a game spec or fail with a clear error."""

    normalized_key = game_key.strip().lower()
    try:
        return _GAME_SPECS[normalized_key]
    except KeyError as exc:
        valid_games = ", ".join(sorted(_GAME_SPECS))
        raise KeyError(f"Unknown game '{game_key}'. Available: {valid_games}") from exc


def list_game_keys() -> list[str]:
    """List supported game keys."""

    return sorted(_GAME_SPECS)
