# LLM Agent Notes

Purpose:
- Own provider resolution, thinking-mode validation, transient retry policy, and live model adapters.

Key files:
- `common.py`: thinking-mode metadata and provider inference
- `gemini_client.py`: Gemini client
- `openai_client.py`: OpenAI client
- `anthropic_client.py`: Anthropic client
- `together_client.py`: Together client
- `retry.py`: transient provider retry classification and backoff helpers
- `model_thinking.json`: declared supported thinking modes per model
- `__init__.py`: public client factory

Edit guidance:
- Keep provider inference in `common.py` synchronized with `build_model_client()`.
- Keep provider adapters aligned on the `generate_turn(..., prompt_messages=None, context_cache=False)` interface.
- If request-shape, retry behavior, or reasoning config changes, update `model_thinking.json`, `retry.py`, tests in `tests/test_llm.py` / `tests/test_model_thinking_options.py`, and any batch fallback logic in `batch_run.py`.

Important invariants:
- `thinking_mode="default"` is always accepted, even for models not yet declared in `model_thinking.json`; non-default modes require an explicit declaration there.
- `thinking_mode` is the requested knob, not the full effective provider config.
- Persisted `thinking_level` and `thinking_budget` must come from `describe_effective_thinking_mode()`, which resolves them using both `model_name` and `thinking_mode`.
- Runtime validation of supported model/mode pairs comes from `model_thinking.json`.
- Append-only prompt mode may send structured user/assistant messages instead of one flat text prompt.
- Explicit `context_cache` hints are only meaningful for providers that support them; Gemini ignores the flag.
- OpenAI requests inline images as base64 data URLs.
- Anthropic requests inline images as base64 source blocks.
- Together requests inline images as OpenAI-style `image_url` content blocks and maps `off`/`on` to `reasoning.enabled=false/true`.
- Gemini `on` for `gemini-2.5-flash` resolves to budget `-1`, while OpenAI and Anthropic expose provider-specific effort/budget mappings.
- Transient provider failures and malformed empty responses are retried inside the active turn via `llm/retry.py` instead of being pushed directly to batch-level reruns.

Usual follow-up tests:
- `python -m unittest tests.test_llm tests.test_batch_run tests.test_pipeline`
