"""LLM response caching (Enhancement Phase 7).

Repeated market states produce near-identical bias calls; caching them
by a stable feature hash cuts token spend and latency. Works for both the
raw parse result and the parsed signal dict.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import Any

from config.constants import (
    LLM_CACHE_FEATURE_PRECISION,
    LLM_CACHE_SIZE,
    LLM_CACHE_TTL_SECONDS,
)
from core.logger import get_logger

log = get_logger(__name__)


def market_state_key(features: dict[str, Any], precision: int = LLM_CACHE_FEATURE_PRECISION) -> str:
    """Stable hash of normalized features; rounding groups near-identical states."""
    rounded = {
        key: f"{float(value):.{precision}f}" if isinstance(value, int | float) else str(value)
        for key, value in features.items()
    }
    blob = ",".join(f"{k}={rounded[k]}" for k in sorted(rounded))
    return hashlib.sha1(blob.encode()).hexdigest()


class LLMCache:
    """LRU + TTL cache keyed by a market-state hash."""

    def __init__(
        self,
        max_size: int = LLM_CACHE_SIZE,
        ttl_seconds: float = LLM_CACHE_TTL_SECONDS,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}  # key -> (stored_at, value)
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if time.time() - stored_at > self._ttl:
            self._store.pop(key, None)
            return None
        self._hits += 1
        self._store.pop(key)
        self._store[key] = entry  # move to MRU
        return value

    def put(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)
        while len(self._store) > self._max_size:
            self._store.pop(next(iter(self._store)))

    def get_or_call(self, key: str, producer: Callable[[], Any]) -> Any:
        cached = self.get(key)
        if cached is not None:
            log.info("llm_cache_hit", extra={"key": key[:8]})
            return cached
        self._misses += 1
        value = producer()
        self.put(key, value)
        return value

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0


def cached_maker_output(
    cache: LLMCache,
    features: dict[str, Any],
    producer: Callable[[], Any],
) -> Any:
    """Run the Maker producer through the cache keyed on market state."""
    key = market_state_key(features)
    return cache.get_or_call(key, producer)
