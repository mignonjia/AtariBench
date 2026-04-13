"""Anthropic client adapter used by the Atari runner."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from .common import LlmTurnResponse, build_token_usage, describe_effective_thinking_mode, read_usage_value
from .retry import call_with_retries

try:
    from ..games.prompt_builder import PromptMessage
except ImportError:  # Running from inside the AtariBench folder.
    from games.prompt_builder import PromptMessage

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
        prompt_messages: list[PromptMessage] | None = None,
        context_cache: bool = False,
    ) -> LlmTurnResponse:
        """Send one multimodal request and return the raw model text plus usage."""

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic is required for live Anthropic calls.") from exc

        client = Anthropic(api_key=self.api_key)
        response = call_with_retries(
            lambda: client.messages.create(
                model=model_name,
                max_tokens=DEFAULT_ANTHROPIC_MAX_TOKENS,
                messages=_build_input_messages(
                    prompt_text=prompt_text,
                    image_paths=image_paths,
                    prompt_messages=prompt_messages,
                ),
                **_build_request_kwargs(
                    model_name=model_name,
                    thinking_mode=thinking_mode,
                    context_cache=context_cache,
                ),
            )
        )
        token_usage = _extract_token_usage(response)
        text = _extract_response_text(response)
        if text:
            return LlmTurnResponse(text=text, token_usage=token_usage)
        return LlmTurnResponse(
            text=_empty_response_fallback(response),
            token_usage=token_usage,
        )


def _build_input_content(prompt_text: str, image_paths: list[str]) -> list[dict[str, object]]:
    segments = prompt_text.split("IMG_HOLDER")
    num_placeholders = len(segments) - 1
    if num_placeholders != len(image_paths):
        raise ValueError(
            f"IMG_HOLDER count ({num_placeholders}) does not match "
            f"number of image paths ({len(image_paths)})."
        )
    content: list[dict[str, object]] = []
    for i, text_seg in enumerate(segments):
        if text_seg:
            content.append({"type": "text", "text": text_seg})
        if i < len(image_paths):
            image_bytes = Path(image_paths[i]).read_bytes()
            encoded = base64.b64encode(image_bytes).decode("ascii")
            mime_type = _guess_mime_type(image_paths[i])
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


def _build_input_messages(
    prompt_text: str,
    image_paths: list[str],
    prompt_messages: list[PromptMessage] | None,
) -> list[dict[str, object]]:
    if not prompt_messages:
        return [{"role": "user", "content": _build_input_content(prompt_text, image_paths)}]
    payload: list[dict[str, object]] = []
    for message in prompt_messages:
        payload.append(
            {
                "role": message.role,
                "content": _build_input_content(message.text, message.image_paths),
            }
        )
    return payload


def _build_request_kwargs(
    model_name: str,
    thinking_mode: str,
    *,
    context_cache: bool = False,
) -> dict[str, object]:
    metadata = describe_effective_thinking_mode(model_name=model_name, thinking_mode=thinking_mode)
    kwargs: dict[str, object] = {}
    if context_cache:
        kwargs["cache_control"] = {"type": "ephemeral"}
    if metadata["thinking_mode"] in {"default", "auto"}:
        return kwargs
    normalized_model = model_name.strip().lower()
    if normalized_model.startswith("claude-haiku"):
        if metadata["thinking_mode"] in {"off", "none"}:
            kwargs["thinking"] = {"type": "disabled"}
            return kwargs
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": metadata["thinking_budget"]}
        return kwargs
    if metadata["thinking_mode"] in {"off", "none"}:
        return kwargs
    effort = metadata["thinking_level"]
    kwargs["thinking"] = {"type": "adaptive"}
    kwargs["output_config"] = {"effort": effort}
    return kwargs


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


def _extract_token_usage(response) -> object:
    usage = getattr(response, "usage", None)
    if usage is None:
        return build_token_usage()
    return build_token_usage(
        input_tokens=read_usage_value(usage, "input_tokens"),
        output_tokens=read_usage_value(usage, "output_tokens"),
        total_tokens=read_usage_value(usage, "total_tokens"),
        thinking_tokens=read_usage_value(usage, "thinking_tokens", "reasoning_tokens"),
        cached_input_tokens=read_usage_value(usage, "cache_read_input_tokens"),
    )
