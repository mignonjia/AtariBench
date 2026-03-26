"""LLM response parsing for Atari clips."""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..games.registry import GameSpec


class ResponseParseError(ValueError):
    """Raised when the model response cannot be converted into actions."""


@dataclasses.dataclass(frozen=True)
class ParsedClipResponse:
    """Normalized model response for a single turn."""

    raw_text: str
    thought: str
    action_strings: list[str]
    action_ids: list[int]
    errors: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.errors


_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)


def normalize_action_name(action: str) -> str:
    """Lowercase and normalize model action strings."""

    normalized = " ".join(action.strip().lower().replace("_", " ").split())
    return normalized


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _CODE_FENCE_RE.sub("", stripped).strip()
    return stripped


def parse_model_response(
    raw_text: str,
    game_spec: "GameSpec",
    max_actions: int,
) -> ParsedClipResponse:
    """Parse a Gemini response into a normalized action list."""

    cleaned = _strip_code_fences(raw_text)
    errors: list[str] = []
    allowed_max_actions = min(max_actions, 10)

    thought_match = re.search(
        r"thought:\s*(.*?)\n\s*move:",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    move_match = re.search(
        r"move:\s*\[(.*?)\]",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    thought = ""
    if thought_match:
        thought = " ".join(thought_match.group(1).strip().split())
    else:
        errors.append("Missing required 'thought:' section.")

    action_strings: list[str] = []
    action_ids: list[int] = []

    if move_match:
        raw_actions = move_match.group(1).strip()
        if not raw_actions:
            errors.append("Move list must contain at least 1 action.")
        else:
            action_strings = [
                normalize_action_name(part)
                for part in raw_actions.split(",")
                if part.strip()
            ]
            if not action_strings:
                errors.append("Move list must contain at least 1 action.")
            if len(action_strings) > allowed_max_actions:
                errors.append(
                    "Move list must contain at most "
                    f"{allowed_max_actions} actions; got {len(action_strings)}."
                )
    else:
        errors.append("Missing required 'move:' section.")

    for action_name in action_strings:
        if action_name not in game_spec.action_map:
            errors.append(f"Unknown action: {action_name}")
        else:
            action_ids.append(game_spec.action_map[action_name])

    return ParsedClipResponse(
        raw_text=raw_text,
        thought=thought,
        action_strings=action_strings,
        action_ids=action_ids,
        errors=errors,
    )


def require_valid_response(parsed: ParsedClipResponse) -> ParsedClipResponse:
    """Raise on parse failure so the caller can fail clearly."""

    if parsed.errors:
        raise ResponseParseError("; ".join(parsed.errors))
    return parsed
