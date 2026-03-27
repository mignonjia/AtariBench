"""Gemini client adapter used by the Atari runner."""

from __future__ import annotations

import os
from pathlib import Path

from .common import describe_effective_thinking_mode
from .retry import call_with_retries

try:
    from ..games.prompt_builder import PromptMessage
except ImportError:  # Running from inside the AtariBench folder.
    from games.prompt_builder import PromptMessage

DEFAULT_GEMINI_TIMEOUT_MS = 60_000


class GeminiClient:
    """Thin wrapper around the official Gemini SDK."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required to call Gemini.")

    def generate_turn(
        self,
        prompt_text: str,
        image_paths: list[str],
        model_name: str,
        thinking_mode: str = "default",
        prompt_messages: list[PromptMessage] | None = None,
    ) -> str:
        """Send one multimodal request and return the raw model text."""

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for live Gemini calls."
            ) from exc

        client = genai.Client(
            api_key=self.api_key,
            http_options=_build_http_options(types),
        )
        response = call_with_retries(
            lambda: client.models.generate_content(
                model=model_name,
                contents=_build_contents(
                    types=types,
                    prompt_text=prompt_text,
                    image_paths=image_paths,
                    prompt_messages=prompt_messages,
                ),
                config=_build_generate_config(types, model_name, thinking_mode),
            )
        )
        text = _extract_response_text(response)
        if text:
            return text
        return _empty_response_fallback(response)


def _empty_response_fallback(response) -> str:
    finish_reasons = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is not None:
            finish_reasons.append(str(finish_reason))

    prompt_feedback = getattr(response, "prompt_feedback", None)
    details = []
    if finish_reasons:
        details.append(f"finish_reason={','.join(finish_reasons)}")
    if prompt_feedback:
        details.append(f"prompt_feedback={prompt_feedback}")
    summary = "; ".join(details) if details else "empty response"
    return (
        "thought: Gemini returned no text output; defaulting to noop. "
        f"Metadata: {summary}\n"
        "move: [noop]"
        )


def _build_contents(types, prompt_text: str, image_paths: list[str], prompt_messages: list[PromptMessage] | None):
    if not prompt_messages:
        return [types.Content(role="user", parts=_build_parts(types, prompt_text, image_paths))]
    contents = []
    for message in prompt_messages:
        role = "model" if message.role == "assistant" else "user"
        contents.append(
            types.Content(
                role=role,
                parts=_build_parts(types, message.text, message.image_paths),
            )
        )
    return contents


def _build_parts(types, prompt_text: str, image_paths: list[str]):
    parts = [types.Part.from_text(text=prompt_text)]
    for image_path in image_paths:
        image_bytes = Path(image_path).read_bytes()
        mime_type = _guess_mime_type(image_path)
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    return parts


def _build_generate_config(types, model_name: str, thinking_mode: str):
    metadata = describe_effective_thinking_mode(model_name=model_name, thinking_mode=thinking_mode)
    if metadata["thinking_mode"] in {"default", "auto", "none"}:
        return None
    if metadata["thinking_budget"] is not None:
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=metadata["thinking_budget"],
                include_thoughts=False,
            )
        )
    if metadata["thinking_level"] == "low":
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW,
                include_thoughts=False,
            )
        )
    if metadata["thinking_level"] == "medium":
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MEDIUM,
                include_thoughts=False,
            )
        )
    if metadata["thinking_level"] == "high":
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH,
                include_thoughts=False,
            )
        )
    if metadata["thinking_level"] == "minimal":
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            )
        )
    raise AssertionError(f"Unhandled thinking mode metadata: {metadata}")


def _build_http_options(types):
    timeout_ms = _resolve_timeout_ms()
    return types.HttpOptions(timeout=timeout_ms)


def _resolve_timeout_ms() -> int:
    raw_value = os.getenv("ATARIBENCH_GEMINI_TIMEOUT_MS", str(DEFAULT_GEMINI_TIMEOUT_MS))
    try:
        timeout_ms = int(raw_value)
    except ValueError:
        return DEFAULT_GEMINI_TIMEOUT_MS
    return max(timeout_ms, 1)


def _guess_mime_type(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/octet-stream"


def _extract_response_text(response) -> str | None:
    direct_text = getattr(response, "text", None)
    if direct_text:
        return direct_text

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        text_parts = [getattr(part, "text", None) for part in parts]
        filtered = [part for part in text_parts if part]
        if filtered:
            return "\n".join(filtered)
    return None
