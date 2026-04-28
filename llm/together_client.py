"""Together client adapter used by the Atari runner."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from .common import LlmTurnResponse, build_token_usage
from .retry import call_with_retries

try:
    from ..games.prompt_builder import PromptMessage
except ImportError:  # Running from inside the AtariBench folder.
    from games.prompt_builder import PromptMessage


class TogetherClient:
    """Thin wrapper around the official Together SDK."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise RuntimeError("TOGETHER_API_KEY is required to call Together.")

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
            from together import Together
        except ImportError as exc:
            raise RuntimeError("together is required for live Together calls.") from exc

        client = Together(api_key=self.api_key)

        def _call():
            stream = client.chat.completions.create(
                model=model_name,
                messages=_build_input_messages(
                    prompt_text=prompt_text,
                    image_paths=image_paths,
                    prompt_messages=prompt_messages,
                ),
                stream=True,
                **_build_request_kwargs(thinking_mode=thinking_mode),
            )
            chunks = []
            for token in stream:
                if hasattr(token, "choices") and token.choices:
                    delta = token.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        chunks.append(delta.content)
            return "".join(chunks)

        text = call_with_retries(_call)
        if text:
            return LlmTurnResponse(text=text, token_usage=build_token_usage())
        return LlmTurnResponse(
            text="thought: Together returned no text output; defaulting to noop.\nmove: [noop]",
            token_usage=build_token_usage(),
        )


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


def _build_request_kwargs(*, thinking_mode: str) -> dict[str, object]:
    normalized_mode = thinking_mode.strip().lower()
    if normalized_mode in {"default", "auto"}:
        return {}
    if normalized_mode in {"off", "none"}:
        return {"reasoning": {"enabled": False}}
    if normalized_mode == "on":
        return {"reasoning": {"enabled": True}}
    raise ValueError(
        "Together models currently support only thinking_mode='default', 'auto', 'off', 'none', or 'on'."
    )


def _guess_mime_type(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/octet-stream"


