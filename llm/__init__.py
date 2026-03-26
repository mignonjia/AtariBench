"""LLM client exports."""

from .common import describe_thinking_mode, infer_model_provider, resolve_model_provider
from .gemini import GeminiClient
from .openai_client import OpenAIClient


def build_model_client(model_name: str, provider: str = "auto"):
    """Build a live model client for the requested provider."""

    resolved_provider = resolve_model_provider(model_name=model_name, provider=provider)
    if resolved_provider == "gemini":
        return GeminiClient()
    if resolved_provider == "openai":
        return OpenAIClient()
    raise AssertionError(f"Unhandled provider: {resolved_provider}")


__all__ = [
    "GeminiClient",
    "OpenAIClient",
    "build_model_client",
    "describe_thinking_mode",
    "infer_model_provider",
    "resolve_model_provider",
]
