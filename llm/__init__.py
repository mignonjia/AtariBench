"""LLM client exports."""

from .anthropic_client import AnthropicClient
from .common import (
    LlmTurnResponse,
    TokenUsage,
    build_token_usage,
    describe_effective_thinking_mode,
    describe_thinking_mode,
    infer_model_provider,
    read_usage_value,
    resolve_model_provider,
    validate_model_thinking_mode,
)
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient


def build_model_client(model_name: str, provider: str = "auto"):
    """Build a live model client for the requested provider."""

    resolved_provider = resolve_model_provider(model_name=model_name, provider=provider)
    if resolved_provider == "gemini":
        return GeminiClient()
    if resolved_provider == "openai":
        return OpenAIClient()
    if resolved_provider == "anthropic":
        return AnthropicClient()
    raise AssertionError(f"Unhandled provider: {resolved_provider}")


__all__ = [
    "AnthropicClient",
    "GeminiClient",
    "LlmTurnResponse",
    "OpenAIClient",
    "TokenUsage",
    "build_model_client",
    "build_token_usage",
    "describe_effective_thinking_mode",
    "describe_thinking_mode",
    "infer_model_provider",
    "read_usage_value",
    "resolve_model_provider",
    "validate_model_thinking_mode",
]
