"""Shared helpers for model client configuration."""

from __future__ import annotations


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
