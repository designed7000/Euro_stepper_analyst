"""
TTL cache to replace Streamlit's @st.cache_data decorator.
Thread-safe, configurable TTL per cached function.
"""

import time
import threading
from functools import wraps
from typing import Any, Dict, Tuple

_cache: Dict[tuple, Tuple[Any, float]] = {}
_cache_lock = threading.Lock()

DEFAULT_TTL = 3600  # 1 hour


def ttl_cache(ttl: int = DEFAULT_TTL):
    """Decorator that caches function results with a time-to-live.

    Args:
        ttl: Time to live in seconds (default: 3600)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (func.__module__, func.__name__) + args + tuple(sorted(kwargs.items()))

            with _cache_lock:
                if key in _cache:
                    result, timestamp = _cache[key]
                    if time.time() - timestamp < ttl:
                        return result

            result = func(*args, **kwargs)

            with _cache_lock:
                _cache[key] = (result, time.time())

            return result
        return wrapper
    return decorator


def clear_cache():
    """Clear all cached values."""
    with _cache_lock:
        _cache.clear()
