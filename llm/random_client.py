"""Random-action baseline client — picks actions uniformly at random."""
from __future__ import annotations

import random
import re

from .common import LlmTurnResponse, TokenUsage


def _parse_actions_from_prompt(prompt_text: str) -> list[str]:
    """Extract action names from the 'Available actions:' block in the prompt."""
    actions: list[str] = []
    in_section = False
    for line in prompt_text.splitlines():
        if "available actions" in line.lower():
            in_section = True
            continue
        if not in_section:
            continue
        m = re.match(r"^\s*-\s+([^:]+):", line)
        if m:
            actions.append(m.group(1).strip())
        elif re.match(r"^[A-Z][^:]+:$", line.strip()) or line.strip().startswith("Your Task"):
            # New section header — end of actions block
            break
    return actions if actions else ["noop"]


class RandomClient:
    """Baseline agent: samples actions uniformly at random each turn."""

    def generate_turn(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        model_name: str = "random",
        thinking_mode: str = "off",
        prompt_messages: list | None = None,
        context_cache: bool = False,
        html_log_path=None,
    ) -> LlmTurnResponse:
        actions = _parse_actions_from_prompt(prompt_text)
        n = random.randint(1, min(3, len(actions)))
        chosen = random.choices(actions, k=n)
        text = f"thought: random action\nmove: [{', '.join(chosen)}]"
        return LlmTurnResponse(text=text, token_usage=TokenUsage())
