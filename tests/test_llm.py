from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT / "AtariBench"
candidate = str(PROJECT_DIR)
if candidate not in sys.path:
    sys.path.insert(0, candidate)

from llm import (
    AnthropicClient,
    GeminiClient,
    build_model_client,
    build_token_usage,
    describe_effective_thinking_mode,
    infer_model_provider,
    resolve_model_provider,
    validate_model_thinking_mode,
)
from llm.anthropic_client import _build_request_kwargs as _build_anthropic_request_kwargs
from llm.gemini_client import _build_generate_config
from llm.openai_client import OpenAIClient
from llm.retry import RetryableResponseError, is_retryable_error
from games.prompt_builder import PromptMessage


class LlmTests(unittest.TestCase):
    def test_infer_model_provider(self) -> None:
        self.assertEqual(infer_model_provider("gemini-2.5-flash"), "gemini")
        self.assertEqual(infer_model_provider("gpt-5.4"), "openai")
        self.assertEqual(infer_model_provider("claude-sonnet-4-6"), "anthropic")
        self.assertEqual(infer_model_provider("o3"), "openai")

    def test_resolve_model_provider_honors_explicit_provider(self) -> None:
        self.assertEqual(resolve_model_provider("custom-model", provider="openai"), "openai")

    def test_build_model_client_returns_openai_client(self) -> None:
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            client = build_model_client("gpt-5.4", provider="openai")
        self.assertIsInstance(client, OpenAIClient)

    def test_build_model_client_returns_anthropic_client(self) -> None:
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            client = build_model_client("claude-sonnet-4-6", provider="anthropic")
        self.assertIsInstance(client, AnthropicClient)

    def test_describe_effective_thinking_mode_matches_provider_specific_request_shape(self) -> None:
        self.assertEqual(
            describe_effective_thinking_mode("gemini-2.5-flash", "on"),
            {"thinking_mode": "on", "thinking_budget": -1, "thinking_level": None},
        )
        self.assertEqual(
            describe_effective_thinking_mode("claude-haiku-4-5", "on"),
            {"thinking_mode": "on", "thinking_budget": 16000, "thinking_level": None},
        )
        self.assertEqual(
            describe_effective_thinking_mode("gpt-5.4", "none"),
            {"thinking_mode": "none", "thinking_budget": None, "thinking_level": "none"},
        )

    def test_validate_model_thinking_mode_rejects_unsupported_pair(self) -> None:
        with self.assertRaisesRegex(ValueError, "not supported for model 'gemini-2.5-flash'"):
            validate_model_thinking_mode("gemini-2.5-flash", "minimal")

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
            def __init__(self, text: str):
                self.output_text = None
                self.output = [FakeOutputItem(text)]
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
                return FakeResponse("thought: aim right\nmove: [right]")

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.client_kwargs = kwargs
                self.responses = FakeResponsesApi()

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = FakeOpenAI

        previous_module = sys.modules.get("openai")
        sys.modules["openai"] = fake_module
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                image_path = Path(tmpdir) / "frame.png"
                image_path.write_bytes(b"png-bytes")
                client = OpenAIClient(api_key="test-key")
                response = client.generate_turn(
                    prompt_text="state",
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
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], "gpt-5.4")
        self.assertEqual(calls[0]["reasoning"], {"effort": "low"})
        self.assertEqual(calls[0]["prompt_cache_retention"], "in-memory")
        self.assertTrue(str(calls[0]["prompt_cache_key"]).startswith("ataribench:"))
        content = calls[0]["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "state"})
        self.assertEqual(content[1]["type"], "input_image")
        self.assertTrue(content[1]["image_url"].startswith("data:image/png;base64,"))

    def test_openai_client_sends_structured_prompt_messages(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeResponse:
            def __init__(self):
                self.output_text = "thought: wait\nmove: [noop]"
                self.usage = {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "total_tokens": 14,
                    "input_tokens_details": {"cached_tokens": 3},
                    "output_tokens_details": {"reasoning_tokens": 2},
                }

        class FakeResponsesApi:
            def create(self, **kwargs):
                calls.append(kwargs)
                return FakeResponse()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.responses = FakeResponsesApi()

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = FakeOpenAI

        previous_module = sys.modules.get("openai")
        sys.modules["openai"] = fake_module
        try:
            client = OpenAIClient(api_key="test-key")
            response = client.generate_turn(
                prompt_text="unused",
                image_paths=[],
                model_name="gpt-5.4-mini",
                thinking_mode="off",
                prompt_messages=[
                    PromptMessage(role="user", text="state", image_paths=[]),
                    PromptMessage(role="assistant", text="thought: go\nmove: [right]", image_paths=[]),
                    PromptMessage(role="user", text="new state", image_paths=[]),
                ],
            )
        finally:
            if previous_module is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = previous_module

        self.assertEqual(response.text, "thought: wait\nmove: [noop]")
        self.assertEqual(response.token_usage.total_tokens, 14)
        self.assertEqual(response.token_usage.thinking_tokens, 2)
        self.assertEqual(response.token_usage.cached_input_tokens, 3)
        self.assertEqual(calls[0]["input"][0]["role"], "user")
        self.assertEqual(calls[0]["input"][1]["role"], "assistant")
        self.assertEqual(calls[0]["input"][2]["role"], "user")
        self.assertEqual(calls[0]["input"][0]["content"][0]["type"], "input_text")
        self.assertEqual(calls[0]["input"][1]["content"][0]["type"], "output_text")
        self.assertEqual(calls[0]["input"][2]["content"][0]["type"], "input_text")

    def test_openai_client_maps_off_to_reasoning_none(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeResponse:
            def __init__(self):
                self.output_text = "thought: wait\nmove: [noop]"
                self.usage = types.SimpleNamespace(
                    input_tokens=8,
                    output_tokens=3,
                    total_tokens=11,
                    input_tokens_details=types.SimpleNamespace(cached_tokens=2),
                    output_tokens_details=types.SimpleNamespace(reasoning_tokens=1),
                )

        class FakeResponsesApi:
            def create(self, **kwargs):
                calls.append(kwargs)
                return FakeResponse()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.responses = FakeResponsesApi()

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = FakeOpenAI

        previous_module = sys.modules.get("openai")
        sys.modules["openai"] = fake_module
        try:
            client = OpenAIClient(api_key="test-key")
            response = client.generate_turn(
                prompt_text="state",
                image_paths=[],
                model_name="gpt-5.4-mini",
                thinking_mode="off",
            )
        finally:
            if previous_module is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = previous_module

        self.assertEqual(response.text, "thought: wait\nmove: [noop]")
        self.assertEqual(response.token_usage.total_tokens, 11)
        self.assertEqual(response.token_usage.thinking_tokens, 1)
        self.assertEqual(response.token_usage.cached_input_tokens, 2)
        self.assertEqual(calls[0]["reasoning"], {"effort": "none"})

    def test_openai_client_retries_transient_errors(self) -> None:
        calls = 0

        class FakeResponse:
            def __init__(self):
                self.output_text = "thought: wait\nmove: [noop]"
                self.usage = types.SimpleNamespace(
                    input_tokens=8,
                    output_tokens=3,
                    total_tokens=11,
                    input_tokens_details=types.SimpleNamespace(cached_tokens=2),
                    output_tokens_details=types.SimpleNamespace(reasoning_tokens=1),
                )

        class FakeResponsesApi:
            def create(self, **kwargs):
                nonlocal calls
                del kwargs
                calls += 1
                if calls < 3:
                    raise RuntimeError("429 RESOURCE_EXHAUSTED")
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
            with mock.patch("llm.retry.random.uniform", return_value=1.0), mock.patch(
                "llm.retry.time.sleep"
            ) as sleep_mock:
                client = OpenAIClient(api_key="test-key")
                response = client.generate_turn(
                    prompt_text="state",
                    image_paths=[],
                    model_name="gpt-5.4-mini",
                    thinking_mode="off",
                )
        finally:
            if previous_module is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = previous_module

        self.assertEqual(response.text, "thought: wait\nmove: [noop]")
        self.assertEqual(calls, 3)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_retry_classifier_treats_remote_protocol_disconnect_as_transient(self) -> None:
        exc = RuntimeError("httpx.RemoteProtocolError: Server disconnected without sending a response.")
        self.assertTrue(is_retryable_error(exc))

    def test_retry_classifier_treats_retryable_response_errors_as_transient(self) -> None:
        exc = RetryableResponseError("Gemini returned no text output.")
        self.assertTrue(is_retryable_error(exc))

    def test_anthropic_client_builds_multimodal_request(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeContentItem:
            def __init__(self, text: str):
                self.type = "text"
                self.text = text

        class FakeResponse:
            def __init__(self, text: str):
                self.content = [FakeContentItem(text)]
                self.usage = types.SimpleNamespace(
                    input_tokens=13,
                    output_tokens=5,
                    cache_read_input_tokens=4,
                )

        class FakeMessagesApi:
            def create(self, **kwargs):
                calls.append(kwargs)
                return FakeResponse("thought: aim right\nmove: [right]")

        class FakeAnthropic:
            def __init__(self, **kwargs):
                self.client_kwargs = kwargs
                self.messages = FakeMessagesApi()

        fake_module = types.ModuleType("anthropic")
        fake_module.Anthropic = FakeAnthropic

        previous_module = sys.modules.get("anthropic")
        sys.modules["anthropic"] = fake_module
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                image_path = Path(tmpdir) / "frame.png"
                image_path.write_bytes(b"png-bytes")
                client = AnthropicClient(api_key="test-key")
                response = client.generate_turn(
                    prompt_text="state",
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
        self.assertIsNone(response.token_usage.thinking_tokens)
        self.assertEqual(response.token_usage.cached_input_tokens, 4)
        self.assertEqual(calls[0]["model"], "claude-sonnet-4-6")
        self.assertEqual(calls[0]["max_tokens"], 20000)
        self.assertEqual(calls[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(calls[0]["thinking"], {"type": "adaptive"})
        self.assertEqual(calls[0]["output_config"], {"effort": "high"})
        content = calls[0]["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "state"})
        self.assertEqual(content[1]["type"], "image")
        self.assertEqual(content[1]["source"]["type"], "base64")

    def test_anthropic_request_kwargs_use_adaptive_for_sonnet(self) -> None:
        self.assertEqual(
            _build_anthropic_request_kwargs("claude-sonnet-4-6", "off"),
            {},
        )
        self.assertEqual(
            _build_anthropic_request_kwargs("claude-sonnet-4-6", "on"),
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}},
        )

    def test_anthropic_request_kwargs_use_enabled_for_haiku(self) -> None:
        self.assertEqual(
            _build_anthropic_request_kwargs("claude-haiku-4-5", "off"),
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(
            _build_anthropic_request_kwargs("claude-haiku-4-5", "on"),
            {"thinking": {"type": "enabled", "budget_tokens": 16000}},
        )

    def test_gemini_client_sets_http_timeout(self) -> None:
        calls: list[dict[str, object]] = []
        client_kwargs: list[dict[str, object]] = []

        class FakePart:
            @staticmethod
            def from_text(*, text: str):
                return {"type": "text", "text": text}

            @staticmethod
            def from_bytes(*, data: bytes, mime_type: str):
                return {"type": "bytes", "data": data, "mime_type": mime_type}

        class FakeContent:
            def __init__(self, role: str, parts: list[object]):
                self.role = role
                self.parts = parts

        class FakeThinkingConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeGenerateContentConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeHttpOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeResponse:
            text = "thought: hold fire\nmove: [noop]"
            usage_metadata = types.SimpleNamespace(
                cached_content_token_count=7,
                prompt_token_count=15,
                candidates_token_count=6,
                thoughts_token_count=5,
                total_token_count=21,
            )

        class FakeModels:
            def generate_content(self, **kwargs):
                calls.append(kwargs)
                return FakeResponse()

        class FakeClient:
            def __init__(self, **kwargs):
                client_kwargs.append(kwargs)
                self.kwargs = kwargs
                self.models = FakeModels()

        fake_types = types.SimpleNamespace(
            Part=FakePart,
            Content=FakeContent,
            ThinkingConfig=FakeThinkingConfig,
            GenerateContentConfig=FakeGenerateContentConfig,
            ThinkingLevel=types.SimpleNamespace(MEDIUM="medium", LOW="low", MINIMAL="minimal"),
            HttpOptions=FakeHttpOptions,
        )
        fake_genai_module = types.ModuleType("google.genai")
        fake_genai_module.Client = FakeClient
        fake_genai_module.types = fake_types
        fake_google_module = types.ModuleType("google")
        fake_google_module.genai = fake_genai_module

        previous_google = sys.modules.get("google")
        previous_google_genai = sys.modules.get("google.genai")
        try:
            sys.modules["google"] = fake_google_module
            sys.modules["google.genai"] = fake_genai_module
            with mock.patch.dict(os.environ, {"ATARIBENCH_GEMINI_TIMEOUT_MS": "12345"}, clear=False):
                client = GeminiClient(api_key="test-key")
                response = client.generate_turn(
                    prompt_text="state",
                    image_paths=[],
                    model_name="gemini-2.5-flash",
                    thinking_mode="off",
                )
        finally:
            if previous_google is None:
                sys.modules.pop("google", None)
            else:
                sys.modules["google"] = previous_google
            if previous_google_genai is None:
                sys.modules.pop("google.genai", None)
            else:
                sys.modules["google.genai"] = previous_google_genai

        self.assertEqual(response.text, "thought: hold fire\nmove: [noop]")
        self.assertEqual(response.token_usage.total_tokens, 21)
        self.assertEqual(response.token_usage.thinking_tokens, 5)
        self.assertEqual(response.token_usage.cached_input_tokens, 7)
        self.assertEqual(calls[0]["model"], "gemini-2.5-flash")
        self.assertEqual(calls[0]["config"].kwargs["thinking_config"].kwargs["thinking_budget"], 0)
        self.assertEqual(client.api_key, "test-key")
        self.assertEqual(client_kwargs[0]["api_key"], "test-key")
        self.assertEqual(client_kwargs[0]["http_options"].kwargs["timeout"], 12345)

    def test_gemini_25_flash_on_uses_unbounded_budget(self) -> None:
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

    def test_gemini_client_retries_transient_errors(self) -> None:
        calls = 0

        class FakePart:
            @staticmethod
            def from_text(*, text: str):
                return {"type": "text", "text": text}

            @staticmethod
            def from_bytes(*, data: bytes, mime_type: str):
                return {"type": "bytes", "data": data, "mime_type": mime_type}

        class FakeContent:
            def __init__(self, role: str, parts: list[object]):
                self.role = role
                self.parts = parts

        class FakeThinkingConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeGenerateContentConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeHttpOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeResponse:
            text = "thought: hold fire\nmove: [noop]"
            usage_metadata = types.SimpleNamespace(
                cached_content_token_count=5,
                prompt_token_count=12,
                candidates_token_count=5,
                thoughts_token_count=4,
                total_token_count=17,
            )

        class FakeModels:
            def generate_content(self, **kwargs):
                nonlocal calls
                del kwargs
                calls += 1
                if calls < 4:
                    raise RuntimeError("503 unavailable")
                return FakeResponse()

        class FakeClient:
            def __init__(self, **kwargs):
                del kwargs
                self.models = FakeModels()

        fake_types = types.SimpleNamespace(
            Part=FakePart,
            Content=FakeContent,
            ThinkingConfig=FakeThinkingConfig,
            GenerateContentConfig=FakeGenerateContentConfig,
            ThinkingLevel=types.SimpleNamespace(MEDIUM="medium", LOW="low", MINIMAL="minimal"),
            HttpOptions=FakeHttpOptions,
        )
        fake_genai_module = types.ModuleType("google.genai")
        fake_genai_module.Client = FakeClient
        fake_genai_module.types = fake_types
        fake_google_module = types.ModuleType("google")
        fake_google_module.genai = fake_genai_module

        previous_google = sys.modules.get("google")
        previous_google_genai = sys.modules.get("google.genai")
        try:
            sys.modules["google"] = fake_google_module
            sys.modules["google.genai"] = fake_genai_module
            with mock.patch("llm.retry.random.uniform", return_value=1.0), mock.patch(
                "llm.retry.time.sleep"
            ) as sleep_mock:
                client = GeminiClient(api_key="test-key")
                response = client.generate_turn(
                    prompt_text="state",
                    image_paths=[],
                    model_name="gemini-2.5-flash",
                    thinking_mode="off",
                )
        finally:
            if previous_google is None:
                sys.modules.pop("google", None)
            else:
                sys.modules["google"] = previous_google
            if previous_google_genai is None:
                sys.modules.pop("google.genai", None)
            else:
                sys.modules["google.genai"] = previous_google_genai

        self.assertEqual(response.text, "thought: hold fire\nmove: [noop]")
        self.assertEqual(calls, 4)
        self.assertEqual(sleep_mock.call_count, 3)

    def test_gemini_client_retries_empty_text_responses(self) -> None:
        calls = 0

        class FakePart:
            @staticmethod
            def from_text(*, text: str):
                return {"type": "text", "text": text}

            @staticmethod
            def from_bytes(*, data: bytes, mime_type: str):
                return {"type": "bytes", "data": data, "mime_type": mime_type}

        class FakeContent:
            def __init__(self, role: str, parts: list[object]):
                self.role = role
                self.parts = parts

        class FakeThinkingConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeGenerateContentConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeHttpOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class EmptyResponse:
            text = None
            usage_metadata = types.SimpleNamespace(
                prompt_token_count=10,
                total_token_count=10,
            )
            candidates = [types.SimpleNamespace(finish_reason="STOP", content=types.SimpleNamespace(parts=[]))]

        class SuccessResponse:
            text = "thought: hold fire\nmove: [noop]"
            usage_metadata = types.SimpleNamespace(
                cached_content_token_count=5,
                prompt_token_count=12,
                candidates_token_count=5,
                thoughts_token_count=4,
                total_token_count=17,
            )

        class FakeModels:
            def generate_content(self, **kwargs):
                nonlocal calls
                del kwargs
                calls += 1
                if calls == 1:
                    return EmptyResponse()
                return SuccessResponse()

        class FakeClient:
            def __init__(self, **kwargs):
                del kwargs
                self.models = FakeModels()

        fake_types = types.SimpleNamespace(
            Part=FakePart,
            Content=FakeContent,
            ThinkingConfig=FakeThinkingConfig,
            GenerateContentConfig=FakeGenerateContentConfig,
            ThinkingLevel=types.SimpleNamespace(MEDIUM="medium", LOW="low", MINIMAL="minimal"),
            HttpOptions=FakeHttpOptions,
        )
        fake_genai_module = types.ModuleType("google.genai")
        fake_genai_module.Client = FakeClient
        fake_genai_module.types = fake_types
        fake_google_module = types.ModuleType("google")
        fake_google_module.genai = fake_genai_module

        previous_google = sys.modules.get("google")
        previous_google_genai = sys.modules.get("google.genai")
        try:
            sys.modules["google"] = fake_google_module
            sys.modules["google.genai"] = fake_genai_module
            with mock.patch("llm.retry.random.uniform", return_value=1.0), mock.patch(
                "llm.retry.time.sleep"
            ) as sleep_mock:
                client = GeminiClient(api_key="test-key")
                response = client.generate_turn(
                    prompt_text="state",
                    image_paths=[],
                    model_name="gemini-2.5-flash",
                    thinking_mode="off",
                )
        finally:
            if previous_google is None:
                sys.modules.pop("google", None)
            else:
                sys.modules["google"] = previous_google
            if previous_google_genai is None:
                sys.modules.pop("google.genai", None)
            else:
                sys.modules["google.genai"] = previous_google_genai

        self.assertEqual(response.text, "thought: hold fire\nmove: [noop]")
        self.assertEqual(response.token_usage.total_tokens, 17)
        self.assertEqual(calls, 2)
        self.assertEqual(sleep_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
