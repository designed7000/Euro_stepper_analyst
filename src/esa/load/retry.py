"""Retry with exponential backoff for calls to slow external services."""

import time
from collections.abc import Callable


class LoadError(Exception):
    """A load step failed and cannot continue."""


def call_with_retry[T](
    fn: Callable[[], T],
    *,
    attempts: int,
    base_delay_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn until it succeeds, at most `attempts` times.

    The delay before retry i (from 0) is base_delay_s * 2**i.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - any failure of the call is retried
            last = exc
            if i < attempts - 1:
                sleep(base_delay_s * 2**i)
    raise LoadError(f"Call failed after {attempts} attempts: {last}") from last
