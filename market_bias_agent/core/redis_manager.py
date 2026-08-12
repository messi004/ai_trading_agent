"""Redis manager — state storage engine (PRD Module 1).

Owns connection lifecycle and the tick/OI buffer data structures.
"""

from __future__ import annotations

import json
from typing import Any, cast

import redis

from config.constants import (
    AUDIT_TRAIL_KEY,
    EOD_STRUCTURAL_BIAS_TTL_SECONDS,
    EXPIRY_ROLLOVER_KEY,
    KEY_CALL_OI_PREFIX,
    KEY_EOD_STRUCTURAL_BIAS,
    KEY_PRE_MARKET_LEVELS,
    KEY_PUT_OI_PREFIX,
    KEY_RAW_TICK_STREAM,
    KEY_SPOT_TICKS,
    KEY_STRIKES,
    OI_BUFFER_TTL_SECONDS,
    OI_WINDOW_INTERVALS,
    PRE_MARKET_LEVELS_TTL_SECONDS,
    SPOT_TICK_BUFFER_SIZE,
    STREAM_MAXLEN,
)
from config.settings import Settings


class RedisManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.client: redis.Redis | None = None

    def connect(self) -> RedisManager:
        client = redis.Redis(
            host=self._settings.redis_host,
            port=self._settings.redis_port,
            db=self._settings.redis_db,
            password=self._settings.redis_password or None,
            decode_responses=True,
        )
        client.ping()
        self.client = client
        return self

    def push_spot_tick(self, tick: dict[str, Any]) -> None:
        """Append a tick to the circular spot buffer (keeps last 300)."""
        assert self.client is not None
        self.client.rpush(KEY_SPOT_TICKS, json.dumps(tick, default=str))
        self.client.ltrim(KEY_SPOT_TICKS, -SPOT_TICK_BUFFER_SIZE, -1)

    def get_spot_ticks(self, count: int = SPOT_TICK_BUFFER_SIZE) -> list[dict[str, Any]]:
        assert self.client is not None
        raw = cast(list[str], self.client.lrange(KEY_SPOT_TICKS, -count, -1))
        return [json.loads(item) for item in raw]

    def push_call_oi(self, strike: int, value: float) -> None:
        self._push_oi(KEY_CALL_OI_PREFIX, strike, value)

    def push_put_oi(self, strike: int, value: float) -> None:
        self._push_oi(KEY_PUT_OI_PREFIX, strike, value)

    def _push_oi(self, prefix: str, strike: int, value: float) -> None:
        assert self.client is not None
        key = f"{prefix}{strike}"
        was_empty = self.client.llen(key) == 0
        self.client.rpush(key, value)
        self.client.ltrim(key, -OI_WINDOW_INTERVALS, -1)
        if was_empty:
            # Phase 7: OI windows are intraday-only; give the key a TTL on first write
            self.client.expire(key, OI_BUFFER_TTL_SECONDS)

    def get_call_oi_window(self, strike: int) -> list[float]:
        return self._get_oi_window(KEY_CALL_OI_PREFIX, strike)

    def get_put_oi_window(self, strike: int) -> list[float]:
        return self._get_oi_window(KEY_PUT_OI_PREFIX, strike)

    def _get_oi_window(self, prefix: str, strike: int) -> list[float]:
        assert self.client is not None
        key = f"{prefix}{strike}"
        raw = cast(list[str], self.client.lrange(key, 0, -1))
        return [float(v) for v in raw]

    def last_tick_age_seconds(self) -> float | None:
        """Age of the newest spot tick in seconds (watchdog support)."""
        if self.client is None:
            return None
        ticks = self.get_spot_ticks(1)
        if not ticks:
            return None
        import time

        return time.time() - float(ticks[-1].get("ts_epoch", 0.0))

    # ------------------------------------------------------------------
    # Append-only raw tick stream (source of truth, Phase 1)
    # ------------------------------------------------------------------
    def append_raw_tick(self, tick: dict[str, Any]) -> None:
        """XADD the raw tick into the append-only stream (capped)."""
        assert self.client is not None
        self.client.xadd(KEY_RAW_TICK_STREAM, cast(dict, tick), maxlen=STREAM_MAXLEN)

    def read_raw_ticks(self, start: str = "-", count: int = 1000) -> list[dict[str, Any]]:
        """Read ticks from the raw stream, oldest-first (XRANGE order)."""
        assert self.client is not None
        entries = cast(list, self.client.xrange(KEY_RAW_TICK_STREAM, start, "+", count=count))
        ticks: list[dict[str, Any]] = []
        for _stream_id, fields in entries:
            item = dict(fields)
            item["_stream_id"] = _stream_id
            ticks.append(item)
        return ticks

    def stream_length(self) -> int:
        assert self.client is not None
        return int(cast(int, self.client.xlen(KEY_RAW_TICK_STREAM)))

    # ------------------------------------------------------------------
    # Decision audit trail (Phase 6)
    # ------------------------------------------------------------------
    def push_audit(self, decision: dict[str, Any], maxlen: int = 10_000) -> None:
        """Append a Maker/Checker decision to the capped audit list."""
        assert self.client is not None
        self.client.lpush(AUDIT_TRAIL_KEY, json.dumps(decision, default=str))
        self.client.ltrim(AUDIT_TRAIL_KEY, 0, maxlen - 1)

    def get_audit(self, count: int = 100) -> list[dict[str, Any]]:
        assert self.client is not None
        raw = cast(list[str], self.client.lrange(AUDIT_TRAIL_KEY, 0, count - 1))
        return [json.loads(item) for item in raw]

    def audit_length(self) -> int:
        assert self.client is not None
        return int(cast(int, self.client.llen(AUDIT_TRAIL_KEY)))

    # ------------------------------------------------------------------
    # Strikes management (Phase 1)
    # ------------------------------------------------------------------
    def set_strikes(self, strikes: list[int]) -> None:
        assert self.client is not None
        pipe = self.client.pipeline()
        pipe.delete(KEY_STRIKES)
        if strikes:
            pipe.zadd(KEY_STRIKES, {str(s): float(s) for s in strikes})
        pipe.execute()

    def get_strikes(self) -> list[int]:
        assert self.client is not None
        raw = cast(list, self.client.zrange(KEY_STRIKES, 0, -1))
        return [int(v) for v in raw]

    def set_active_expiry(self, expiry_date: str) -> None:
        assert self.client is not None
        self.client.set(EXPIRY_ROLLOVER_KEY, expiry_date)

    def get_active_expiry(self) -> str | None:
        assert self.client is not None
        return cast(str | None, self.client.get(EXPIRY_ROLLOVER_KEY))

    # ------------------------------------------------------------------
    # Pre-market levels (Phase 6): next-day S/R + max pain zones
    # ------------------------------------------------------------------
    def set_pre_market_levels(
        self, levels: dict[str, Any], ttl_seconds: int = PRE_MARKET_LEVELS_TTL_SECONDS
    ) -> None:
        """Persist next-day S/R + max-pain zones for the live intraday engine."""
        assert self.client is not None
        self.client.set(KEY_PRE_MARKET_LEVELS, json.dumps(levels, default=str))
        self.client.expire(KEY_PRE_MARKET_LEVELS, ttl_seconds)

    def get_pre_market_levels(self) -> dict[str, Any] | None:
        assert self.client is not None
        raw = cast(str | None, self.client.get(KEY_PRE_MARKET_LEVELS))
        if raw is None:
            return None
        return cast(dict[str, Any], json.loads(raw))

    # ------------------------------------------------------------------
    # EOD structural bias (Phase 8): next-day FII/PRO institutional bias
    # ------------------------------------------------------------------
    def set_eod_bias(
        self, bias: dict[str, Any], ttl_seconds: int = EOD_STRUCTURAL_BIAS_TTL_SECONDS
    ) -> None:
        """Persist the EOD institutional bias for the next trading session."""
        assert self.client is not None
        self.client.set(KEY_EOD_STRUCTURAL_BIAS, json.dumps(bias, default=str))
        self.client.expire(KEY_EOD_STRUCTURAL_BIAS, ttl_seconds)

    def get_eod_bias(self) -> dict[str, Any] | None:
        assert self.client is not None
        raw = cast(str | None, self.client.get(KEY_EOD_STRUCTURAL_BIAS))
        if raw is None:
            return None
        return cast(dict[str, Any], json.loads(raw))

    # ------------------------------------------------------------------
    # Memory monitoring (Phase 7)
    # ------------------------------------------------------------------
    def memory_usage_bytes(self) -> int | None:
        """Redis used_memory in bytes, or None when unavailable."""
        if self.client is None:
            return None
        try:
            info = cast(dict, self.client.info("memory"))
        except Exception:  # noqa: BLE001 - fakeredis / restricted servers lack INFO
            return None
        return int(info.get("used_memory", 0)) or None

    def dbsize(self) -> int:
        if self.client is None:
            return 0
        return int(cast(int, self.client.dbsize()))

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
