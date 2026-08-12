"""Tick normalization and integrity checks (Enhancement Phase 1).

Every raw WebSocket message passes through a TickValidator before it is
persisted. Invalid ticks are dropped and counted for observability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from config.constants import (
    MAX_TICK_AGE_SECONDS,
    OI_MAX_TICK_AGE_SECONDS,
    OI_TICK_SKEW_TOLERANCE_SECONDS,
    TICK_SKEW_TOLERANCE_SECONDS,
)


class TickError(ValueError):
    """Raised when a raw tick cannot be normalized."""


@dataclass(frozen=True)
class Tick:
    """Canonical normalized tick used across the pipeline."""

    type: str  # "spot" | "oi"
    symbol: str
    ts_epoch: float
    price: float = 0.0
    volume: float = 0.0
    strike: int = 0
    option_type: str = ""  # "CALL" | "PUT"
    oi: float = 0.0

    @property
    def signature(self) -> tuple[str, str, float, float, float, int]:
        """Signature used for consecutive-duplicate detection."""
        return (self.type, self.symbol, self.price, self.volume, self.oi, self.strike)


def _to_float(raw: Any, field: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise TickError(f"field '{field}' not numeric: {raw!r}") from None
    if value < 0:
        raise TickError(f"field '{field}' must be >= 0, got {value}")
    return value


def normalize_tick(raw: dict[str, Any], fallback_epoch: float | None = None) -> Tick:
    """Map a raw WebSocket payload into a canonical Tick.

    Raw shape (Breeze-like):
      {"type": "spot", "symbol": "NIFTY", "price": 24005.5, "volume": 123, "ts_epoch": ...}
      {"type": "oi", "symbol": "NIFTY", "strike": 24000, "option_type": "CALL",
       "oi": 123456, "ts_epoch": ...}
    """
    tick_type = str(raw.get("type", "")).lower()
    if tick_type not in ("spot", "oi"):
        raise TickError(f"unknown tick type: {raw.get('type')!r}")

    ts_epoch = _to_float(raw.get("ts_epoch", fallback_epoch), "ts_epoch")
    if ts_epoch <= 0:
        raise TickError(f"invalid ts_epoch: {ts_epoch}")

    symbol = str(raw.get("symbol", "")).strip() or "UNKNOWN"

    if tick_type == "spot":
        return Tick(
            type="spot",
            symbol=symbol,
            ts_epoch=ts_epoch,
            price=_to_float(raw.get("price"), "price"),
            volume=_to_float(raw.get("volume", 0), "volume"),
        )

    return Tick(
        type="oi",
        symbol=symbol,
        ts_epoch=ts_epoch,
        price=_to_float(raw.get("price", 0), "price"),
        strike=int(_to_float(raw.get("strike"), "strike")),
        option_type=str(raw.get("option_type", "")).upper(),
        oi=_to_float(raw.get("oi"), "oi"),
    )


def is_stale(
    tick: Tick, max_age_seconds: float = MAX_TICK_AGE_SECONDS, now: float | None = None
) -> bool:
    """True if the tick is older than max_age vs the local clock."""
    current = now if now is not None else time.time()
    return (current - tick.ts_epoch) > max_age_seconds


def is_out_of_order(
    tick: Tick,
    prev_tick: Tick | None,
    skew_tolerance_seconds: float = TICK_SKEW_TOLERANCE_SECONDS,
) -> bool:
    """True if the tick timestamp jumped unexpectedly (rejected as out-of-order).

    Only meaningful when comparing ticks of the same symbol/type sequence.
    """
    if prev_tick is None:
        return False
    if tick.symbol != prev_tick.symbol or tick.type != prev_tick.type:
        return False
    if tick.type == "oi" and (
        tick.strike != prev_tick.strike or tick.option_type != prev_tick.option_type
    ):
        return False
    return (tick.ts_epoch - prev_tick.ts_epoch) > skew_tolerance_seconds


def is_duplicate(prev_tick: Tick | None, tick: Tick) -> bool:
    """True if this tick repeats the previous tick's exact values."""
    if prev_tick is None:
        return False
    if tick.symbol != prev_tick.symbol or tick.type != prev_tick.type:
        return False
    if tick.type == "oi" and (
        tick.strike != prev_tick.strike or tick.option_type != prev_tick.option_type
    ):
        return False
    return tick.signature == prev_tick.signature


@dataclass
class ValidationStats:
    total: int = 0
    dropped_stale: int = 0
    dropped_out_of_order: int = 0
    dropped_duplicate: int = 0
    dropped_malformed: int = 0
    accepted: int = 0


class TickValidator:
    """Stateful validator maintaining per-symbol last-tick state."""

    def __init__(
        self,
        max_age_seconds: float = MAX_TICK_AGE_SECONDS,
        skew_tolerance_seconds: float = TICK_SKEW_TOLERANCE_SECONDS,
        drop_duplicates: bool = True,
        oi_max_age_seconds: float = OI_MAX_TICK_AGE_SECONDS,
        oi_skew_tolerance_seconds: float = OI_TICK_SKEW_TOLERANCE_SECONDS,
    ) -> None:
        self.max_age = max_age_seconds
        self.skew = skew_tolerance_seconds
        self.drop_duplicates = drop_duplicates
        self.oi_max_age = oi_max_age_seconds
        self.oi_skew = oi_skew_tolerance_seconds
        self._last_by_key: dict[tuple[str, str, int, str], Tick] = {}
        self.stats = ValidationStats()

    def validate(self, raw: dict[str, Any], now: float | None = None) -> Tick | None:
        """Validate a raw tick. Returns canonical Tick or None if dropped."""
        self.stats.total += 1
        try:
            tick = normalize_tick(raw, fallback_epoch=now)
        except TickError:
            self.stats.dropped_malformed += 1
            return None

        # OI ticks carry `ltt` = last TRADE time, not OI-update time. An
        # illiquid strike can trade once an hour yet push current OI every
        # few seconds; treating the trade timestamp as staleness would drop
        # live OI data. Spot prices, by contrast, only change on trade, so
        # their age is a faithful freshness signal.
        if tick.type == "oi":
            max_age = max(self.max_age, self.oi_max_age)
            skew = max(self.skew, self.oi_skew)
        else:
            max_age, skew = self.max_age, self.skew

        if is_stale(tick, max_age, now):
            self.stats.dropped_stale += 1
            return None

        key = (tick.type, tick.symbol, tick.strike, tick.option_type)
        prev = self._last_by_key.get(key)
        if is_out_of_order(tick, prev, skew):
            self.stats.dropped_out_of_order += 1
            return None
        if self.drop_duplicates and is_duplicate(prev, tick):
            self.stats.dropped_duplicate += 1
            return None

        self._last_by_key[key] = tick
        self.stats.accepted += 1
        return tick
