"""Dynamic strike list sync + expiry rollover (Enhancement Phase 1).

Strike ranges change daily and on expiry day (last Thursday of the month).
A provider fetches the current expiry + full strike chain; we keep a window
of strikes around the ATM and detect rollover so stale OI buffers are reset.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from config.constants import (
    KEY_CALL_OI_PREFIX,
    KEY_PUT_OI_PREFIX,
    STRIKES_RANGE_AROUND_ATM,
    STRIKES_SYNC_INTERVAL_SECONDS,
)
from config.settings import Settings
from core.logger import get_logger
from core.redis_manager import RedisManager

log = get_logger(__name__)


@runtime_checkable
class StrikesProvider(Protocol):
    """Abstract fetch of the current expiry + full strike chain."""

    def fetch_expiry_and_strikes(self, symbol: str) -> tuple[str, list[int]]:
        """Return (expiry_date 'YYYY-MM-DD', sorted strikes list)."""
        ...


class StrikesManager:
    def __init__(
        self,
        settings: Settings,
        redis: RedisManager,
        provider: StrikesProvider,
        sync_interval_seconds: int = STRIKES_SYNC_INTERVAL_SECONDS,
        range_around_atm: int = STRIKES_RANGE_AROUND_ATM,
    ) -> None:
        self._settings = settings
        self._redis = redis
        self._provider = provider
        self._interval = sync_interval_seconds
        self._range = range_around_atm
        self._last_sync_at = float("-inf")  # first call always due
        self.rollover_detected = False

    def is_due(self, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
        return (current - self._last_sync_at) >= self._interval

    def sync(self, now: float | None = None) -> dict:
        """Fetch latest expiry + strikes, persist, handle rollover.

        Returns a summary dict for logging/observability.
        """
        expiry, full_strikes = self._provider.fetch_expiry_and_strikes(self._settings.nifty_symbol)
        strikes = sorted(set(full_strikes))
        self._last_sync_at = now if now is not None else time.time()

        stored_expiry = self._redis.get_active_expiry()
        if stored_expiry is not None and stored_expiry != expiry:
            self.rollover_detected = True
            log.warning(
                "expiry_rollover",
                extra={"old": stored_expiry, "new": expiry},
            )
            self._reset_oi_windows(strikes)
        elif stored_expiry is None:
            self.rollover_detected = False

        self._redis.set_active_expiry(expiry)
        self._redis.set_strikes(strikes)

        summary = {
            "expiry": expiry,
            "strikes_count": len(strikes),
            "rollover_detected": self.rollover_detected,
        }
        log.info("strikes_synced", extra=summary)
        return summary

    def sync_if_due(self, now: float | None = None) -> bool:
        """Sync only if the refresh interval elapsed. Returns True if synced."""
        if not self.is_due(now):
            return False
        self.sync(now)
        return True

    def nearest_strikes(
        self,
        spot: float,
        range_around_atm: int | None = None,
    ) -> list[int]:
        """Nearest strikes around the current spot (window for OI tracking)."""
        strikes = self._redis.get_strikes()
        if not strikes:
            return []
        atm = int(round(spot / 100.0) * 100)
        lo, hi = (
            atm - (range_around_atm or self._range) * 100,
            atm + (range_around_atm or self._range) * 100,
        )
        return [s for s in strikes if lo <= s <= hi]

    def _reset_oi_windows(self, strikes: list[int]) -> None:
        """Drop stale per-strike OI buffers after rollover."""
        assert self._redis.client is not None
        client = self._redis.client
        for strike in strikes:
            client.delete(f"{KEY_CALL_OI_PREFIX}{strike}")
            client.delete(f"{KEY_PUT_OI_PREFIX}{strike}")
