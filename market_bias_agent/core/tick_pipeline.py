"""Tick ingestion pipeline (Phase 1): validate -> persist.

Every raw WebSocket tick flows: validate (stale/out-of-order/duplicate) ->
append to the raw source-of-truth stream -> update derived buffers.
"""

from __future__ import annotations

from typing import Any

from config.settings import Settings
from core.logger import get_logger
from core.redis_manager import RedisManager
from core.tick_validator import Tick, TickValidator

log = get_logger(__name__)


class TickPipeline:
    def __init__(
        self,
        settings: Settings,
        redis: RedisManager,
        validator: TickValidator,
        health: Any = None,
    ) -> None:
        self._settings = settings
        self._redis = redis
        self._validator = validator
        self._health = health

    def process(self, raw: dict) -> Tick | None:
        """Validate and persist a raw tick. Returns the canonical Tick or None."""
        tick = self._validator.validate(raw)
        if tick is None:
            return None
        if self._health is not None:
            self._health.record_tick()

        record = {
            "type": tick.type,
            "symbol": tick.symbol,
            "ts_epoch": tick.ts_epoch,
            "price": tick.price,
            "volume": tick.volume,
            "strike": tick.strike,
            "option_type": tick.option_type,
            "oi": tick.oi,
        }
        self._redis.append_raw_tick(record)

        if tick.type == "spot":
            self._redis.push_spot_tick(
                {"price": tick.price, "volume": tick.volume, "ts_epoch": tick.ts_epoch}
            )
        elif tick.type == "oi":
            if tick.option_type == "CALL":
                self._redis.push_call_oi(tick.strike, tick.oi)
            elif tick.option_type == "PUT":
                self._redis.push_put_oi(tick.strike, tick.oi)
            else:
                log.warning("unknown_option_type", extra={"option_type": tick.option_type})
        return tick

    @property
    def stats(self):
        return self._validator.stats
