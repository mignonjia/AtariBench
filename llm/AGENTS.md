# LLM Agent Notes

Purpose:
- Own provider resolution and live model adapters.

Key files:
- `common.py`: thinking-mode metadata and provider inference
- `gemini_client.py`: Gemini client
- `openai_client.py`: OpenAI client
- `anthropic_client.py`: Anthropic client
- `model_thinking.json`: declared supported thinking modes per model
- `__init__.py`: public client factory

Edit guidance:
- Keep provider inference in `common.py` synchronized with `build_model_client()`.
- Keep provider adapters aligned on the `generate_turn(..., prompt_messages=None)` interface.
- If request-shape or reasoning config changes, update tests in `tests/test_llm.py` and any batch fallback logic in `batch_run.py`.

Important invariants:
- `thinking_mode="off"` maps to no reasoning / zero-budget behavior.
- `thinking_mode` is the requested knob, not the full effective provider config.
- Persisted `thinking_level` and `thinking_budget` must come from `describe_effective_thinking_mode()`, which resolves them using both `model_name` and `thinking_mode`.
- Runtime validation of supported model/mode pairs comes from `model_thinking.json`.
- Append-only prompt mode may send structured user/assistant messages instead of one flat text prompt.
- OpenAI requests inline images as base64 data URLs.

Usual follow-up tests:
- `python -m unittest tests.test_llm tests.test_batch_run tests.test_pipeline`
