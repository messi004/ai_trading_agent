"""REST backfill to warm Redis buffers on (re)connect (Enhancement Phase 1).

On boot the 1m/5m velocity windows are empty. Before live ticks arrive, a
snapshot provider fetches recent bars/ticks and we replay them through the
same Redis buffers so velocity calculations are warm immediately.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from config.settings import Settings
from core.logger import get_logger
from core.redis_manager import RedisManager

log = get_logger(__name__)


@runtime_checkable
class SnapshotProvider(Protocol):
    """Abstract historical/recent data fetch (Breeze REST)."""

    def fetch_recent_snapshot(self, symbol: str, strikes: list[int], lookback_seconds: int) -> dict:
        """Return {"spot_ticks": [tick...], "oi_samples": [tick...]} (oldest first)."""
        ...


class BackfillService:
    def __init__(
        self,
        settings: Settings,
        redis: RedisManager,
        provider: SnapshotProvider,
        lookback_seconds: int = 600,
    ) -> None:
        self._settings = settings
        self._redis = redis
        self._provider = provider
        self._lookback = lookback_seconds

    async def run(self) -> int:
        """Replay the snapshot into buffers. Returns number of ticks ingested."""
        strikes = self._redis.get_strikes()
        try:
            snapshot = self._provider.fetch_recent_snapshot(
                self._settings.nifty_symbol, strikes, self._lookback
            )
        except Exception as exc:  # noqa: BLE001
            log.error("backfill_fetch_failed", extra={"error": str(exc)})
            return 0

        spot_ticks = snapshot.get("spot_ticks", [])
        oi_samples = snapshot.get("oi_samples", [])

        for raw in spot_ticks:
            tick = raw.get("price")
            self._redis.push_spot_tick({"price": tick, "ts_epoch": raw.get("ts_epoch", 0.0)})
        for raw in oi_samples:
            strike = raw.get("strike")
            if raw.get("option_type") == "CALL":
                self._redis.push_call_oi(strike, raw.get("oi", 0.0))
            else:
                self._redis.push_put_oi(strike, raw.get("oi", 0.0))

        ingested = len(spot_ticks) + len(oi_samples)
        log.info(
            "backfill_complete",
            extra={
                "spot": len(spot_ticks),
                "oi": len(oi_samples),
                "lookback_seconds": self._lookback,
            },
        )
        return ingested
