"""OpenAI client adapter used by the Atari runner."""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

from .common import LlmTurnResponse, build_token_usage, describe_effective_thinking_mode, read_usage_value
from .retry import call_with_retries

try:
    from ..games.prompt_builder import PromptMessage
except ImportError:  # Running from inside the AtariBench folder.
    from games.prompt_builder import PromptMessage


class OpenAIClient:
    """Thin wrapper around the official OpenAI SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required to call OpenAI.")
        self.organization = organization or os.getenv("OPENAI_ORG_ID")
        self.project = project or os.getenv("OPENAI_PROJECT_ID")

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
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is required for live OpenAI calls.") from exc

        client_kwargs = {"api_key": self.api_key}
        if self.organization:
            client_kwargs["organization"] = self.organization
        if self.project:
            client_kwargs["project"] = self.project
        client = OpenAI(**client_kwargs)

        response = call_with_retries(
            lambda: client.responses.create(
                model=model_name,
                input=_build_input_messages(
                    prompt_text=prompt_text,
                    image_paths=image_paths,
                    prompt_messages=prompt_messages,
                ),
                **_build_request_kwargs(
                    model_name=model_name,
                    thinking_mode=thinking_mode,
                    context_cache=context_cache,
                    prompt_text=prompt_text,
                    prompt_messages=prompt_messages,
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


def _build_input_content(
    prompt_text: str,
    image_paths: list[str],
    *,
    role: str,
) -> list[dict[str, str]]:
    text_type = "output_text" if role == "assistant" else "input_text"
    content = [{"type": text_type, "text": prompt_text}]
    for image_path in image_paths:
        image_bytes = Path(image_path).read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        mime_type = _guess_mime_type(image_path)
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{encoded}",
            }
        )
    return content


def _build_input_messages(
    prompt_text: str,
    image_paths: list[str],
    prompt_messages: list[PromptMessage] | None,
) -> list[dict[str, object]]:
    if not prompt_messages:
        return [
            {
                "role": "user",
                "content": _build_input_content(prompt_text, image_paths, role="user"),
            }
        ]
    payload: list[dict[str, object]] = []
    for message in prompt_messages:
        payload.append(
            {
                "role": message.role,
                "content": _build_input_content(
                    message.text,
                    message.image_paths,
                    role=message.role,
                ),
            }
        )
    return payload


def _build_request_kwargs(
    model_name: str,
    thinking_mode: str,
    *,
    context_cache: bool = False,
    prompt_text: str = "",
    prompt_messages: list[PromptMessage] | None = None,
) -> dict[str, object]:
    metadata = describe_effective_thinking_mode(model_name=model_name, thinking_mode=thinking_mode)
    kwargs: dict[str, object] = {}
    if metadata["thinking_mode"] not in {"default", "auto"}:
        if metadata["thinking_level"] == "none":
            kwargs["reasoning"] = {"effort": "none"}
        else:
            kwargs["reasoning"] = {"effort": metadata["thinking_level"]}
    if context_cache:
        kwargs["prompt_cache_key"] = _build_prompt_cache_key(
            model_name=model_name,
            prompt_text=prompt_text,
            prompt_messages=prompt_messages,
        )
        kwargs["prompt_cache_retention"] = "in-memory"
    return kwargs


def _empty_response_fallback(response) -> str:
    response_id = getattr(response, "id", None)
    status = getattr(response, "status", None)
    details = []
    if response_id:
        details.append(f"id={response_id}")
    if status:
        details.append(f"status={status}")
    summary = "; ".join(details) if details else "empty response"
    return (
        "thought: OpenAI returned no text output; defaulting to noop. "
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
    direct_text = getattr(response, "output_text", None)
    if direct_text:
        return direct_text

    output_items = getattr(response, "output", None) or []
    for item in output_items:
        content_items = getattr(item, "content", None) or []
        text_parts = []
        for content_item in content_items:
            if getattr(content_item, "type", None) == "output_text":
                text_value = getattr(content_item, "text", None)
                if text_value:
                    text_parts.append(text_value)
        if text_parts:
            return "\n".join(text_parts)
    return None


def _extract_token_usage(response) -> object:
    usage = getattr(response, "usage", None)
    if usage is None:
        return build_token_usage()
    input_tokens_details = read_usage_value(usage, "input_tokens_details")
    output_tokens_details = read_usage_value(usage, "output_tokens_details")
    return build_token_usage(
        input_tokens=read_usage_value(usage, "input_tokens", "prompt_tokens"),
        output_tokens=read_usage_value(usage, "output_tokens", "completion_tokens"),
        total_tokens=read_usage_value(usage, "total_tokens"),
        thinking_tokens=read_usage_value(output_tokens_details, "reasoning_tokens"),
        cached_input_tokens=read_usage_value(input_tokens_details, "cached_tokens"),
    )


def _build_prompt_cache_key(
    *,
    model_name: str,
    prompt_text: str,
    prompt_messages: list[PromptMessage] | None,
) -> str:
    if prompt_messages:
        root_text = prompt_messages[0].text
        root_image_count = len(prompt_messages[0].image_paths)
    else:
        root_text = prompt_text
        root_image_count = 0
    digest = hashlib.sha256(
        f"{model_name}\n{root_image_count}\n{root_text}".encode("utf-8")
    ).hexdigest()
    return f"ataribench:{digest[:32]}"
