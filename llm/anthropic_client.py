"""Anthropic client adapter used by the Atari runner."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from .common import describe_thinking_mode
from .retry import call_with_retries

DEFAULT_ANTHROPIC_MAX_TOKENS = 20_000
THINKING_BUDGET_TOKENS = {
    "on": 16_000,
    "low": 4_000,
    "medium": 8_000,
    "high": 12_000,
    "max": 16_000,
}


class AnthropicClient:
    """Thin wrapper around the official Anthropic SDK."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required to call Anthropic.")

    def generate_turn(
        self,
        prompt_text: str,
        image_paths: list[str],
        model_name: str,
        thinking_mode: str = "default",
    ) -> str:
        """Send one multimodal request and return the raw model text."""

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic is required for live Anthropic calls.") from exc

        client = Anthropic(api_key=self.api_key)
        response = call_with_retries(
            lambda: client.messages.create(
                model=model_name,
                max_tokens=DEFAULT_ANTHROPIC_MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": _build_input_content(prompt_text, image_paths),
                    }
                ],
                **_build_request_kwargs(model_name=model_name, thinking_mode=thinking_mode),
            )
        )
        text = _extract_response_text(response)
        if text:
            return text
        return _empty_response_fallback(response)


def _build_input_content(prompt_text: str, image_paths: list[str]) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [{"type": "text", "text": prompt_text}]
    for image_path in image_paths:
        image_bytes = Path(image_path).read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        mime_type = _guess_mime_type(image_path)
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": encoded,
                },
            }
        )
    return content


def _build_request_kwargs(model_name: str, thinking_mode: str) -> dict[str, object]:
    metadata = describe_thinking_mode(thinking_mode)
    if metadata["thinking_mode"] in {"default", "auto"}:
        return {}
    normalized_model = model_name.strip().lower()
    if normalized_model.startswith("claude-haiku"):
        if metadata["thinking_mode"] in {"off", "none"}:
            return {"thinking": {"type": "disabled"}}
        budget_tokens = THINKING_BUDGET_TOKENS[metadata["thinking_mode"]]
        return {"thinking": {"type": "enabled", "budget_tokens": budget_tokens}}
    if metadata["thinking_mode"] in {"off", "none"}:
        return {}
    effort = metadata["thinking_level"]
    if metadata["thinking_mode"] == "on":
        effort = "medium"
    return {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort},
    }


def _empty_response_fallback(response) -> str:
    response_id = getattr(response, "id", None)
    stop_reason = getattr(response, "stop_reason", None)
    details = []
    if response_id:
        details.append(f"id={response_id}")
    if stop_reason:
        details.append(f"stop_reason={stop_reason}")
    summary = "; ".join(details) if details else "empty response"
    return (
        "thought: Anthropic returned no text output; defaulting to noop. "
        f"Metadata: {summary}\n"
        "move: [noop]"
    )


def _guess_mime_type(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/octet-stream"


def _extract_response_text(response) -> str | None:
    content_items = getattr(response, "content", None) or []
    text_parts = []
    for item in content_items:
        if getattr(item, "type", None) == "text":
            text_value = getattr(item, "text", None)
            if text_value:
                text_parts.append(text_value)
    if text_parts:
        return "\n".join(text_parts)
    return None
