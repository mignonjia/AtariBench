"""Shared helpers for model client configuration."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_ANTHROPIC_HAIKU_BUDGETS = {
    "on": 16_000,
    "low": 4_000,
    "medium": 8_000,
    "high": 12_000,
    "max": 16_000,
}


def describe_thinking_mode(thinking_mode: str) -> dict[str, str | int | None]:
    """Return summary-safe metadata for the configured thinking mode."""

    normalized_mode = thinking_mode.strip().lower()
    if normalized_mode in {"default", "auto"}:
        return {
            "thinking_mode": normalized_mode,
            "thinking_budget": None,
            "thinking_level": None,
        }
    if normalized_mode in {"off", "none"}:
        return {
            "thinking_mode": normalized_mode,
            "thinking_budget": 0 if normalized_mode == "off" else None,
            "thinking_level": None,
        }
    if normalized_mode == "minimal":
        return {
            "thinking_mode": "minimal",
            "thinking_budget": None,
            "thinking_level": "minimal",
        }
    if normalized_mode == "low":
        return {
            "thinking_mode": "low",
            "thinking_budget": None,
            "thinking_level": "low",
        }
    if normalized_mode == "medium":
        return {
            "thinking_mode": "medium",
            "thinking_budget": None,
            "thinking_level": "medium",
        }
    if normalized_mode == "high":
        return {
            "thinking_mode": "high",
            "thinking_budget": None,
            "thinking_level": "high",
        }
    if normalized_mode == "xhigh":
        return {
            "thinking_mode": "xhigh",
            "thinking_budget": None,
            "thinking_level": "xhigh",
        }
    if normalized_mode == "max":
        return {
            "thinking_mode": "max",
            "thinking_budget": None,
            "thinking_level": "max",
        }
    if normalized_mode == "on":
        return {
            "thinking_mode": "on",
            "thinking_budget": None,
            "thinking_level": "medium",
        }
    raise ValueError(f"Unsupported thinking mode: {thinking_mode}")


def describe_effective_thinking_mode(
    model_name: str,
    thinking_mode: str,
    provider: str = "auto",
) -> dict[str, str | int | None]:
    """Return provider-aware thinking metadata that matches the actual request."""

    metadata = describe_thinking_mode(thinking_mode)
    resolved_provider = resolve_model_provider(model_name=model_name, provider=provider)
    normalized_mode = str(metadata["thinking_mode"])
    result = {
        "thinking_mode": normalized_mode,
        "thinking_budget": None,
        "thinking_level": None,
    }

    if resolved_provider == "gemini":
        if normalized_mode in {"default", "auto", "none"}:
            return result
        if normalized_mode == "off":
            result["thinking_budget"] = 0
            return result
        if normalized_mode == "on":
            if model_name.strip().lower().startswith("gemini-2.5-flash"):
                result["thinking_budget"] = -1
                return result
            result["thinking_level"] = "medium"
            return result
        if normalized_mode in {"minimal", "low", "medium", "high"}:
            result["thinking_level"] = normalized_mode
            return result
        raise ValueError(f"Unsupported Gemini thinking mode: {thinking_mode}")

    if resolved_provider == "openai":
        if normalized_mode in {"default", "auto"}:
            return result
        if normalized_mode in {"off", "none"}:
            result["thinking_level"] = "none"
            return result
        if normalized_mode in {"low", "medium", "high", "xhigh"}:
            result["thinking_level"] = normalized_mode
            return result
        raise ValueError(f"Unsupported OpenAI thinking mode: {thinking_mode}")

    if resolved_provider == "anthropic":
        normalized_model = model_name.strip().lower()
        if normalized_model.startswith("claude-haiku"):
            if normalized_mode in {"default", "auto", "off", "none"}:
                return result
            if normalized_mode in _ANTHROPIC_HAIKU_BUDGETS:
                result["thinking_budget"] = _ANTHROPIC_HAIKU_BUDGETS[normalized_mode]
                return result
            raise ValueError(f"Unsupported Claude Haiku thinking mode: {thinking_mode}")
        if normalized_mode in {"default", "auto", "off", "none"}:
            return result
        if normalized_mode == "on":
            result["thinking_level"] = "medium"
            return result
        if normalized_mode in {"low", "medium", "high", "max"}:
            result["thinking_level"] = normalized_mode
            return result
        raise ValueError(f"Unsupported Anthropic thinking mode: {thinking_mode}")

    raise AssertionError(f"Unhandled provider: {resolved_provider}")


@lru_cache(maxsize=1)
def load_model_thinking_config() -> dict[str, list[str]]:
    """Load the supported thinking modes per model from disk."""

    config_path = Path(__file__).resolve().parent / "model_thinking.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        _normalize_model_thinking_key(str(model_name)): list(options)
        for model_name, options in payload.items()
    }


def validate_model_thinking_mode(model_name: str, thinking_mode: str) -> None:
    """Raise when a model/mode pair is not declared as supported."""

    normalized_mode = thinking_mode.strip().lower()
    if normalized_mode == "default":
        return

    payload = load_model_thinking_config()
    allowed_modes = payload.get(_normalize_model_thinking_key(model_name))
    if allowed_modes is None:
        raise ValueError(
            f"Model '{model_name}' is not declared in llm/model_thinking.json. "
            "Only thinking mode 'default' is allowed until the model is added there."
        )

    normalized_allowed = {option.strip().lower() for option in allowed_modes}
    if normalized_mode not in normalized_allowed:
        supported = ", ".join(["default", *sorted(normalized_allowed)])
        raise ValueError(
            f"Thinking mode '{thinking_mode}' is not supported for model '{model_name}'. "
            f"Supported modes: {supported}."
        )


def _normalize_model_thinking_key(model_name: str) -> str:
    normalized_name = model_name.strip().lower()
    if normalized_name.startswith("models/"):
        return normalized_name.split("/", 1)[1]
    return normalized_name


def infer_model_provider(model_name: str) -> str:
    """Infer the backing provider from a model name."""

    normalized_name = model_name.strip().lower()
    if normalized_name.startswith("models/gemini") or normalized_name.startswith("gemini"):
        return "gemini"
    if normalized_name.startswith("claude-"):
        return "anthropic"
    if normalized_name.startswith(
        (
            "gpt-",
            "o1",
            "o3",
            "o4",
            "gpt-oss-",
            "codex-",
            "chatgpt-",
            "computer-use-preview",
        )
    ):
        return "openai"
    raise ValueError(
        f"Could not infer provider from model '{model_name}'. "
        "Use a Gemini/OpenAI model name or pass --provider explicitly."
    )


def resolve_model_provider(model_name: str, provider: str = "auto") -> str:
    """Resolve the configured provider, inferring it when requested."""

    normalized_provider = provider.strip().lower()
    if normalized_provider == "auto":
        return infer_model_provider(model_name)
    if normalized_provider in {"gemini", "openai", "anthropic"}:
        return normalized_provider
    raise ValueError(f"Unsupported provider: {provider}")
