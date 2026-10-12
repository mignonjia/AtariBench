"""Retry helpers for transient model API failures."""

from __future__ import annotations

import os
import random
import re
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

MAX_RETRIES = 5
MAX_BACKOFF_SECONDS = 60.0
MAX_TRANSIENT_RETRY_WINDOW_SECONDS = 900.0
_RETRY_DELAY_PATTERNS = (
    re.compile(r"retry in (?P<seconds>\d+(?:\.\d+)?)s", re.IGNORECASE),
    re.compile(r"retrydelay['\"]?\s*[:=]\s*['\"]?(?P<seconds>\d+(?:\.\d+)?)s", re.IGNORECASE),
)


class RetryableResponseError(RuntimeError):
    """Raised when a provider returns a malformed or unusable response worth retrying."""


def call_with_retries(
    operation: Callable[[], T],
    max_retries: int = MAX_RETRIES,
    max_retry_window_seconds: float | None = None,
) -> T:
    """Run one operation with retryable failures kept inside the active turn."""

    started_at = time.monotonic()
    retry_index = 0
    retry_window_seconds = _resolve_retry_window_seconds(max_retry_window_seconds)

    while True:
        try:
            return operation()
        except Exception as exc:
            if not is_retryable_error(exc):
                raise
            sleep_seconds = compute_retry_delay_seconds(exc=exc, retry_index=retry_index)
            next_retry_index = retry_index + 1
            if next_retry_index > max_retries and retry_window_seconds is not None:
                elapsed_seconds = time.monotonic() - started_at
                if elapsed_seconds + sleep_seconds > retry_window_seconds:
                    raise
            time.sleep(sleep_seconds)
            retry_index = next_retry_index
    raise AssertionError("unreachable")


def compute_retry_delay_seconds(exc: Exception, retry_index: int) -> float:
    """Return a provider-guided retry delay or a capped exponential fallback."""

    hinted_delay = _extract_retry_delay_seconds(exc)
    if hinted_delay is not None:
        return hinted_delay
    exponential = min(2**retry_index, MAX_BACKOFF_SECONDS)
    upper_bound = min(2 ** (retry_index + 1), MAX_BACKOFF_SECONDS)
    return random.uniform(exponential, upper_bound)


def _resolve_retry_window_seconds(configured_value: float | None) -> float | None:
    if configured_value is not None:
        return configured_value

    raw_value = os.getenv(
        "ATARIBENCH_LLM_MAX_TRANSIENT_RETRY_WINDOW_SECONDS",
        str(MAX_TRANSIENT_RETRY_WINDOW_SECONDS),
    ).strip()
    if raw_value.lower() in {"none", "unbounded", "infinite"}:
        return None

    try:
        parsed = float(raw_value)
    except ValueError:
        return MAX_TRANSIENT_RETRY_WINDOW_SECONDS
    return None if parsed < 0 else parsed


def _extract_retry_delay_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            parsed = _coerce_delay_seconds(retry_after)
            if parsed is not None:
                return parsed

    message = str(exc)
    for pattern in _RETRY_DELAY_PATTERNS:
        match = pattern.search(message)
        if not match:
            continue
        parsed = _coerce_delay_seconds(match.group("seconds"))
        if parsed is not None:
            return parsed
    return None


def _coerce_delay_seconds(raw_value: object) -> float | None:
    try:
        parsed = float(str(raw_value).strip())
    except ValueError:
        return None
    return max(parsed, 0.0)


def is_retryable_error(exc: Exception) -> bool:
    """Return whether one provider error should be retried."""

    if isinstance(exc, RetryableResponseError):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    message = str(exc).lower()
    transient_markers = (
        "429",
        "408",
        "409",
        "500",
        "502",
        "503",
        "504",
        "resource_exhausted",
        "rate limit",
        "too many requests",
        "server error",
        "service unavailable",
        "temporarily unavailable",
        "high demand",
        "connecterror",
        "remoteprotocolerror",
        "readtimeout",
        "timeoutexception",
        "timed out",
        "server disconnected without sending a response",
        "connection reset",
        "temporary failure in name resolution",
        "nodename nor servname provided",
        "no route to host",
    )
    return any(marker in message for marker in transient_markers)
