from __future__ import annotations

import json
import os
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT / "AtariBench"
candidate = str(PROJECT_DIR)
if candidate not in sys.path:
    sys.path.insert(0, candidate)

from llm import build_model_client
from llm.common import describe_effective_thinking_mode, describe_thinking_mode, infer_model_provider
from llm.gemini_client import _build_generate_config
from llm.anthropic_client import _build_request_kwargs as _build_anthropic_request_kwargs
from llm.openai_client import _build_request_kwargs


class ModelThinkingOptionsTests(unittest.TestCase):
    def test_model_thinking_json_options_are_supported_by_llm_layer(self) -> None:
        config_path = PROJECT_DIR / "llm" / "model_thinking.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))

        fake_types = types.SimpleNamespace(
            ThinkingConfig=lambda **kwargs: types.SimpleNamespace(kwargs=kwargs),
            GenerateContentConfig=lambda **kwargs: types.SimpleNamespace(kwargs=kwargs),
            ThinkingLevel=types.SimpleNamespace(
                MINIMAL="minimal",
                LOW="low",
                MEDIUM="medium",
                HIGH="high",
            ),
        )

        for model_name, options in payload.items():
            provider = infer_model_provider(model_name)
            for option in options:
                with self.subTest(model=model_name, option=option):
                    metadata = describe_thinking_mode(option)
                    self.assertEqual(metadata["thinking_mode"], option.lower())
                    if provider == "gemini":
                        config = _build_generate_config(fake_types, model_name, option)
                        if option == "off":
                            self.assertIsNotNone(config)
                            self.assertEqual(
                                config.kwargs["thinking_config"].kwargs["thinking_budget"],
                                0,
                            )
                        elif option in {"auto", "none"}:
                            self.assertIsNone(config)
                        else:
                            self.assertIsNotNone(config)
                            if model_name == "gemini-2.5-flash" and option == "on":
                                self.assertEqual(
                                    config.kwargs["thinking_config"].kwargs["thinking_budget"],
                                    -1,
                                )
                                self.assertEqual(
                                    config.kwargs["thinking_config"].kwargs["include_thoughts"],
                                    False,
                                )
                                continue
                            expected_level = metadata["thinking_level"]
                            if option == "on":
                                expected_level = "medium"
                            self.assertEqual(
                                config.kwargs["thinking_config"].kwargs["thinking_level"],
                                expected_level,
                            )
                    elif provider == "openai":
                        request_kwargs = _build_request_kwargs(model_name, option)
                        if option in {"auto", "default"}:
                            self.assertEqual(request_kwargs, {})
                        elif option == "none":
                            self.assertEqual(request_kwargs, {"reasoning": {"effort": "none"}})
                        else:
                            self.assertEqual(
                                request_kwargs,
                                {"reasoning": {"effort": metadata["thinking_level"]}},
                            )
                    elif provider == "anthropic":
                        request_kwargs = _build_anthropic_request_kwargs(model_name, option)
                        if option in {"auto", "default"}:
                            self.assertEqual(request_kwargs, {})
                        elif model_name.startswith("claude-haiku"):
                            if option in {"off", "none"}:
                                self.assertEqual(request_kwargs, {"thinking": {"type": "disabled"}})
                            else:
                                self.assertEqual(
                                    request_kwargs,
                                    {
                                        "thinking": {
                                            "type": "enabled",
                                            "budget_tokens": _expected_anthropic_budget(option),
                                        }
                                    },
                                )
                        else:
                            if option in {"off", "none"}:
                                self.assertEqual(request_kwargs, {})
                            else:
                                expected_effort = metadata["thinking_level"]
                                if option == "on":
                                    expected_effort = "medium"
                                self.assertEqual(
                                    request_kwargs,
                                    {
                                        "thinking": {"type": "adaptive"},
                                        "output_config": {"effort": expected_effort},
                                    },
                                )
                    else:  # pragma: no cover
                        self.fail(f"Unexpected provider for {model_name}: {provider}")

                    effective = describe_effective_thinking_mode(model_name=model_name, thinking_mode=option)
                    if model_name == "gemini-2.5-flash" and option == "on":
                        self.assertEqual(effective["thinking_budget"], -1)
                        self.assertIsNone(effective["thinking_level"])
                    if model_name == "claude-haiku-4-5" and option == "on":
                        self.assertEqual(effective["thinking_budget"], 16000)
                        self.assertIsNone(effective["thinking_level"])

    def test_model_thinking_json_options_support_live_requests(self) -> None:
        if os.getenv("ATARIBENCH_RUN_LIVE_MODEL_THINKING_TESTS") != "1":
            self.skipTest("Set ATARIBENCH_RUN_LIVE_MODEL_THINKING_TESTS=1 to run live API checks.")

        _load_api_env(PROJECT_DIR / "llm" / "api.sh")
        _assert_required_live_sdk_modules()

        config_path = PROJECT_DIR / "llm" / "model_thinking.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        output_path = PROJECT_DIR / "tests" / "artifacts" / "model_thinking_live_results.json"
        records: list[dict[str, object]] = []
        prompt_text = "hello"
        _write_live_results(output_path, records)

        for model_name, options in payload.items():
            client = build_model_client(model_name)
            for option in options:
                with self.subTest(model=model_name, option=option):
                    try:
                        response = client.generate_turn(
                            prompt_text=prompt_text,
                            image_paths=[],
                            model_name=model_name,
                            thinking_mode=option,
                        )
                    except Exception as exc:
                        records.append(
                            {
                                "model": model_name,
                                "thinking_mode": option,
                                "prompt": prompt_text,
                                "response": None,
                                "error": str(exc),
                            }
                        )
                        _write_live_results(output_path, records)
                        raise
                    records.append(
                        {
                                "model": model_name,
                                "thinking_mode": option,
                                "prompt": prompt_text,
                                "response": response.text,
                                "error": None,
                            }
                        )
                    _write_live_results(output_path, records)
                    self.assertIsInstance(response.text, str)
                    self.assertTrue(
                        response.text.strip(),
                        f"Expected non-empty response for {model_name} with {option}",
                    )


def _load_api_env(api_path: Path) -> None:
    for raw_line in api_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith("export "):
            continue
        key, _, value = line[len("export ") :].partition("=")
        if not key or not value:
            continue
        os.environ.setdefault(key.strip(), value.strip())


def _assert_required_live_sdk_modules() -> None:
    missing = []
    try:
        import openai  # noqa: F401
    except ImportError:
        missing.append("openai")
    try:
        from google import genai  # noqa: F401
    except ImportError:
        missing.append("google-genai")
    try:
        import anthropic  # noqa: F401
    except ImportError:
        missing.append("anthropic")
    if missing:
        raise RuntimeError(
            "Live model thinking tests require installed SDKs: " + ", ".join(missing)
        )


def _expected_anthropic_budget(option: str) -> int:
    return {
        "on": 16000,
        "low": 4000,
        "medium": 8000,
        "high": 12000,
        "max": 16000,
    }[option]


def _write_live_results(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"records": records}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
