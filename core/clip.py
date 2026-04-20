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

# Direction component tokens in canonical order (vertical before horizontal).
_VERTICALS = ("up", "down")
_HORIZONTALS = ("right", "left")
_SUFFIXES = ("fire", "punch", "shoot")


def _try_reorder_compound_action(name: str) -> str | None:
    """Reorder compound direction words to match game action conventions.

    Models sometimes output leftup/rightdown/up-right; games expect upleft/downright/upright.
    Returns the reordered name, or None if the name can't be decomposed this way.
    """
    remaining = name
    vertical: str | None = None
    horizontal: str | None = None
    suffix: str | None = None

    for s in _SUFFIXES:
        if remaining.endswith(s) and remaining != s:
            suffix = s
            remaining = remaining[: -len(s)]
            break

    for v in _VERTICALS:
        if v in remaining:
            vertical = v
            remaining = remaining.replace(v, "", 1)
            break

    for h in _HORIZONTALS:
        if h in remaining:
            horizontal = h
            remaining = remaining.replace(h, "", 1)
            break

    if remaining:
        return None

    parts = [p for p in (vertical, horizontal, suffix) if p]
    return "".join(parts) if len(parts) >= 2 else None


def normalize_action_name(action: str) -> str:
    """Lowercase and normalize model action strings."""

    stripped = action.strip().strip("\"'")
    normalized = " ".join(stripped.lower().replace("_", " ").split())
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

    # Primary: thought: ... \n [optional number+dot] move:
    thought_match = re.search(
        r"thought:\s*(.*?)\n[^\S\n]*(?:\d+\.\s*)?move:",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Fallback: "thought" without colon (model omitted it), same newline tolerance
    if not thought_match:
        thought_match = re.search(
            r"thought[^:\w]\s*(.*?)\n[^\S\n]*(?:\d+\.\s*)?move[:\s]",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
    # Fallback: thought and move on the same line (no newline between them)
    if not thought_match:
        thought_match = re.search(
            r"thought:\s*(.*?)(?=\s*move:\s*\[)",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
    # Bracketed: move: [a, b, c]
    move_match = re.search(
        r"move:\s*\[(.*?)\]",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Fallback: bracket-free "move: a, b, c" (single line, no brackets)
    if not move_match:
        move_match = re.search(
            r"move:\s*([^\[\]\n][^\n]*)",
            cleaned,
            flags=re.IGNORECASE,
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
        if action_name in game_spec.action_map:
            action_ids.append(game_spec.action_map[action_name])
            continue

        # Try removing hyphens: "up-right" → "upright"
        dehyphenated = action_name.replace("-", "")
        if dehyphenated in game_spec.action_map:
            action_ids.append(game_spec.action_map[dehyphenated])
            continue

        # Try reordering compound direction words: "leftup" → "upleft"
        reordered = _try_reorder_compound_action(action_name)
        if reordered and reordered in game_spec.action_map:
            action_ids.append(game_spec.action_map[reordered])
            continue

        # Try reorder then dehyphenate: "left-up" → "leftup" → "upleft"
        if reordered:
            reordered_dh = reordered.replace("-", "")
            if reordered_dh in game_spec.action_map:
                action_ids.append(game_spec.action_map[reordered_dh])
                continue

        errors.append(f"Unknown action: {action_name}")

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
