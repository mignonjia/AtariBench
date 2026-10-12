from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
candidate = str(SRC_DIR)
if candidate not in sys.path:
    sys.path.insert(0, candidate)

from ataribench.llm import (
    build_token_usage,
    describe_effective_thinking_mode,
    infer_model_provider,
    resolve_model_provider,
    validate_model_thinking_mode,
)
from ataribench.llm.anthropic_client import AnthropicClient
from ataribench.llm.gemini_client import _build_generate_config
from ataribench.llm.openai_client import OpenAIClient
from ataribench.llm.retry import (
    RetryableResponseError,
    call_with_retries,
    compute_retry_delay_seconds,
    is_retryable_error,
)
from ataribench.llm.together_client import TogetherClient, _build_request_kwargs as _build_together_request_kwargs


class LlmTests(unittest.TestCase):
    def test_provider_resolution(self) -> None:
        self.assertEqual(infer_model_provider("gemini-2.5-flash"), "gemini")
        self.assertEqual(infer_model_provider("gpt-5.4"), "openai")
        self.assertEqual(infer_model_provider("claude-sonnet-4-6"), "anthropic")
        self.assertEqual(infer_model_provider("deepseek-ai/DeepSeek-V3.1"), "together")
        self.assertEqual(resolve_model_provider("custom-model", provider="openai"), "openai")

    def test_effective_thinking_mode_is_provider_aware(self) -> None:
        self.assertEqual(
            describe_effective_thinking_mode("gemini-2.5-flash", "on"),
            {"thinking_mode": "on", "thinking_budget": -1, "thinking_level": None},
        )
        self.assertEqual(
            describe_effective_thinking_mode("gpt-5.4", "none"),
            {"thinking_mode": "none", "thinking_budget": None, "thinking_level": "none"},
        )
        self.assertEqual(
            describe_effective_thinking_mode("gpt-5.6-sol", "max"),
            {"thinking_mode": "max", "thinking_budget": None, "thinking_level": "max"},
        )
        self.assertEqual(
            describe_effective_thinking_mode("deepseek-ai/DeepSeek-V3.1", "off"),
            {"thinking_mode": "off", "thinking_budget": None, "thinking_level": "none"},
        )

    def test_validate_model_thinking_mode_rejects_unsupported_pair(self) -> None:
        with self.assertRaisesRegex(ValueError, "not supported for model 'gemini-2.5-flash'"):
            validate_model_thinking_mode("gemini-2.5-flash", "minimal")

    def test_validate_gpt_5_6_sol_thinking_modes(self) -> None:
        for mode in ("none", "low", "medium", "high", "xhigh", "max"):
            validate_model_thinking_mode("gpt-5.6-sol", mode)

    def test_validate_claude_opus_4_8_thinking_modes(self) -> None:
        for mode in ("low", "medium", "high", "max"):
            validate_model_thinking_mode("claude-opus-4-8", mode)

    def test_openai_client_builds_multimodal_request(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeResponseText:
            def __init__(self, text: str):
                self.type = "output_text"
                self.text = text

        class FakeOutputItem:
            def __init__(self, text: str):
                self.content = [FakeResponseText(text)]

        class FakeResponse:
            output_text = None

            def __init__(self):
                self.output = [FakeOutputItem("thought: aim right\nmove: [right]")]
                self.usage = types.SimpleNamespace(
                    input_tokens=21,
                    output_tokens=9,
                    total_tokens=30,
                    input_tokens_details=types.SimpleNamespace(cached_tokens=6),
                    output_tokens_details=types.SimpleNamespace(reasoning_tokens=4),
                )

        class FakeResponsesApi:
            def create(self, **kwargs):
                calls.append(kwargs)
                return FakeResponse()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                del kwargs
                self.responses = FakeResponsesApi()

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = FakeOpenAI
        previous_module = sys.modules.get("openai")
        sys.modules["openai"] = fake_module
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                image_path = Path(tmpdir) / "frame.png"
                image_path.write_bytes(b"png-bytes")
                response = OpenAIClient(api_key="test-key").generate_turn(
                    prompt_text="state\nIMG_HOLDER",
                    image_paths=[str(image_path)],
                    model_name="gpt-5.4",
                    thinking_mode="low",
                    context_cache=True,
                )
        finally:
            if previous_module is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = previous_module

        self.assertEqual(response.text, "thought: aim right\nmove: [right]")
        self.assertEqual(
            response.token_usage,
            build_token_usage(
                input_tokens=21,
                output_tokens=9,
                total_tokens=30,
                thinking_tokens=4,
                cached_input_tokens=6,
            ),
        )
        self.assertEqual(calls[0]["model"], "gpt-5.4")
        self.assertEqual(calls[0]["reasoning"], {"effort": "low"})
        self.assertEqual(calls[0]["prompt_cache_retention"], "in-memory")
        content = calls[0]["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "state\n"})
        self.assertEqual(content[1]["type"], "input_image")

    def test_anthropic_client_builds_multimodal_request(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeResponse:
            def __init__(self):
                self.content = [types.SimpleNamespace(type="text", text="thought: aim right\nmove: [right]")]
                self.usage = types.SimpleNamespace(
                    input_tokens=13,
                    output_tokens=5,
                    cache_read_input_tokens=4,
                )

        class FakeMessages:
            def create(self, **kwargs):
                calls.append(kwargs)
                return FakeResponse()

        class FakeAnthropic:
            def __init__(self, **kwargs):
                del kwargs
                self.messages = FakeMessages()

        fake_module = types.ModuleType("anthropic")
        fake_module.Anthropic = FakeAnthropic
        previous_module = sys.modules.get("anthropic")
        sys.modules["anthropic"] = fake_module
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                image_path = Path(tmpdir) / "frame.png"
                image_path.write_bytes(b"png-bytes")
                response = AnthropicClient(api_key="test-key").generate_turn(
                    prompt_text="state\nIMG_HOLDER",
                    image_paths=[str(image_path)],
                    model_name="claude-sonnet-4-6",
                    thinking_mode="high",
                    context_cache=True,
                )
        finally:
            if previous_module is None:
                sys.modules.pop("anthropic", None)
            else:
                sys.modules["anthropic"] = previous_module

        self.assertEqual(response.text, "thought: aim right\nmove: [right]")
        self.assertEqual(response.token_usage.total_tokens, 18)
        self.assertEqual(calls[0]["thinking"], {"type": "adaptive"})
        self.assertEqual(calls[0]["output_config"], {"effort": "high"})
        self.assertEqual(calls[0]["cache_control"], {"type": "ephemeral"})
        content = calls[0]["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "state\n"})
        self.assertEqual(content[1]["type"], "image")

    def test_gemini_flash_on_uses_unbounded_budget(self) -> None:
        class FakeThinkingConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeGenerateContentConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_types = types.SimpleNamespace(
            ThinkingConfig=FakeThinkingConfig,
            GenerateContentConfig=FakeGenerateContentConfig,
            ThinkingLevel=types.SimpleNamespace(MEDIUM="medium", LOW="low", MINIMAL="minimal"),
        )

        config = _build_generate_config(fake_types, "gemini-2.5-flash", "on")
        self.assertEqual(config.kwargs["thinking_config"].kwargs["thinking_budget"], -1)
        self.assertEqual(config.kwargs["thinking_config"].kwargs["include_thoughts"], False)

    def test_together_client_builds_streaming_multimodal_request(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeChatCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return [
                    types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                delta=types.SimpleNamespace(content="thought: drift right\n")
                            )
                        ]
                    ),
                    types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(delta=types.SimpleNamespace(content="move: [right]"))
                        ]
                    ),
                ]

        class FakeTogether:
            def __init__(self, **kwargs):
                del kwargs
                self.chat = types.SimpleNamespace(completions=FakeChatCompletions())

        fake_module = types.ModuleType("together")
        fake_module.Together = FakeTogether
        previous_module = sys.modules.get("together")
        sys.modules["together"] = fake_module
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                image_path = Path(tmpdir) / "frame.png"
                image_path.write_bytes(b"png-bytes")
                response = TogetherClient(api_key="test-key").generate_turn(
                    prompt_text="state\nIMG_HOLDER",
                    image_paths=[str(image_path)],
                    model_name="deepseek-ai/DeepSeek-V3.1",
                    thinking_mode="on",
                )
        finally:
            if previous_module is None:
                sys.modules.pop("together", None)
            else:
                sys.modules["together"] = previous_module

        self.assertEqual(response.text, "thought: drift right\nmove: [right]")
        self.assertEqual(calls[0]["model"], "deepseek-ai/DeepSeek-V3.1")
        self.assertEqual(calls[0]["reasoning"], {"enabled": True})
        content = calls[0]["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "state\n"})
        self.assertEqual(content[1]["type"], "image_url")

    def test_together_request_kwargs_map_reasoning_flag(self) -> None:
        self.assertEqual(_build_together_request_kwargs(thinking_mode="default"), {})
        self.assertEqual(_build_together_request_kwargs(thinking_mode="off"), {"reasoning": {"enabled": False}})
        self.assertEqual(_build_together_request_kwargs(thinking_mode="on"), {"reasoning": {"enabled": True}})
        with self.assertRaisesRegex(ValueError, "currently support only"):
            _build_together_request_kwargs(thinking_mode="low")

    def test_retry_helpers_cover_transient_errors_and_provider_hints(self) -> None:
        self.assertTrue(is_retryable_error(RuntimeError("429 RESOURCE_EXHAUSTED")))
        self.assertTrue(is_retryable_error(RetryableResponseError("empty provider response")))
        self.assertEqual(
            compute_retry_delay_seconds(
                RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 22.5s."),
                retry_index=7,
            ),
            22.5,
        )

        calls = 0

        def flaky_operation() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("503 unavailable")
            return "ok"

        with mock.patch("ataribench.llm.retry.compute_retry_delay_seconds", return_value=1.0), mock.patch(
            "ataribench.llm.retry.time.sleep"
        ):
            self.assertEqual(
                call_with_retries(flaky_operation, max_retries=1, max_retry_window_seconds=30.0),
                "ok",
            )
        self.assertEqual(calls, 3)


if __name__ == "__main__":
    unittest.main()
