"""Retry helpers for transient model API failures."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

MAX_RETRIES = 5


class RetryableResponseError(RuntimeError):
    """Raised when a provider returns a malformed or unusable response worth retrying."""


def call_with_retries(operation: Callable[[], T], max_retries: int = MAX_RETRIES) -> T:
    """Run one operation with jittered exponential backoff for transient errors."""

    for retry_index in range(max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            if retry_index >= max_retries or not is_retryable_error(exc):
                raise
            sleep_seconds = random.uniform(2**retry_index, 2 ** (retry_index + 1))
            time.sleep(sleep_seconds)
    raise AssertionError("unreachable")


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
