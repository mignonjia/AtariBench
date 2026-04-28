"""DashScope (Alibaba) client adapter using the OpenAI-compatible interface."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from .common import LlmTurnResponse, build_token_usage, read_usage_value
from .retry import call_with_retries

try:
    from ..games.prompt_builder import PromptMessage
except ImportError:  # Running from inside the AtariBench folder.
    from games.prompt_builder import PromptMessage

_DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class DashScopeClient:
    """Thin wrapper around DashScope's OpenAI-compatible chat completions API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required to call DashScope.")

    def generate_turn(
        self,
        prompt_text: str,
        image_paths: list[str],
        model_name: str,
        thinking_mode: str = "default",
        prompt_messages: list[PromptMessage] | None = None,
        context_cache: bool = False,
        html_log_path=None,
    ) -> LlmTurnResponse:
        """Send one multimodal request and return the raw model text plus usage."""
        del context_cache

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is required for DashScope calls.") from exc

        client = OpenAI(api_key=self.api_key, base_url=_DASHSCOPE_BASE_URL)
        response = call_with_retries(
            lambda: client.chat.completions.create(
                model=model_name,
                messages=_build_input_messages(
                    prompt_text=prompt_text,
                    image_paths=image_paths,
                    prompt_messages=prompt_messages,
                ),
                **_build_request_kwargs(thinking_mode=thinking_mode),
            )
        )
        token_usage = _extract_token_usage(response)
        text = _extract_response_text(response)
        if text:
            return LlmTurnResponse(text=text, token_usage=token_usage)
        return LlmTurnResponse(
            text="thought: DashScope returned no text output; defaulting to noop.\nmove: [noop]",
            token_usage=token_usage,
        )


def _build_request_kwargs(*, thinking_mode: str) -> dict[str, object]:
    normalized_mode = thinking_mode.strip().lower()
    if normalized_mode in {"off", "none"}:
        return {"extra_body": {"enable_thinking": False}}
    # default / on: let the model decide
    return {}


def _build_input_messages(
    prompt_text: str,
    image_paths: list[str],
    prompt_messages: list[PromptMessage] | None,
) -> list[dict[str, object]]:
    if not prompt_messages:
        return [{"role": "user", "content": _build_message_content(prompt_text, image_paths)}]
    payload: list[dict[str, object]] = []
    for message in prompt_messages:
        payload.append(
            {
                "role": message.role,
                "content": _build_message_content(message.text, message.image_paths),
            }
        )
    return payload


def _build_message_content(prompt_text: str, image_paths: list[str]) -> str | list[dict[str, object]]:
    segments = prompt_text.split("IMG_HOLDER")
    num_placeholders = len(segments) - 1
    if num_placeholders != len(image_paths):
        raise ValueError(
            f"IMG_HOLDER count ({num_placeholders}) does not match "
            f"number of image paths ({len(image_paths)})."
        )
    if not image_paths:
        return prompt_text
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
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded}",
                    },
                }
            )
    return content


def _guess_mime_type(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/octet-stream"


def _extract_response_text(response) -> str | None:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    if message is None:
        return None
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return content
    return None


def _extract_token_usage(response) -> object:
    usage = getattr(response, "usage", None)
    if usage is None:
        return build_token_usage()
    return build_token_usage(
        input_tokens=read_usage_value(usage, "prompt_tokens", "input_tokens"),
        output_tokens=read_usage_value(usage, "completion_tokens", "output_tokens"),
        total_tokens=read_usage_value(usage, "total_tokens"),
    )
