# LLM Agent Notes

Purpose:
- Own provider resolution and live model adapters.

Key files:
- `common.py`: thinking-mode metadata and provider inference
- `gemini.py`: Gemini client
- `openai_client.py`: OpenAI client
- `__init__.py`: public client factory

Edit guidance:
- Keep provider inference in `common.py` synchronized with `build_model_client()`.
- Preserve the `generate_turn(prompt_text, image_paths, model_name, thinking_mode)` interface across providers.
- If request-shape or reasoning config changes, update tests in `tests/test_llm.py` and any batch fallback logic in `batch_run.py`.

Important invariants:
- `thinking_mode="off"` maps to no reasoning / zero-budget behavior.
- Non-default reasoning modes flow through summary metadata via `describe_thinking_mode()`.
- OpenAI requests inline images as base64 data URLs.

Usual follow-up tests:
- `python -m unittest tests.test_llm tests.test_batch_run tests.test_pipeline`
