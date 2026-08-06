"""Candle construction, ATR, volatility regime & pattern detection (Phase 2).

Structural intraday pipeline works on 1-minute resampled candles. These
pure functions build candles from ticks and classify market state / patterns.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from config.constants import (
    ATR_PERIOD,
    PATTERN_BEAR_ENGULFING,
    PATTERN_BULL_ENGULFING,
    PATTERN_HAMMER,
    PATTERN_PIN_BAR,
    PATTERN_SHOOTING_STAR,
    PATTERN_SWEEP_HIGH,
    PATTERN_SWEEP_LOW,
    PIN_BAR_WICK_BODY_RATIO,
    REGIME_CALM_ATR_PCT,
    REGIME_HIGH_VOL_ATR_PCT,
    SWEEP_WICK_TOLERANCE_POINTS,
)


@dataclass(frozen=True)
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    ts_epoch: float = 0.0

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open


def build_candles_from_ticks(
    ticks: Sequence[dict],
    interval_seconds: int = 60,
) -> list[Candle]:
    """Resample ordered spot ticks into OHLCV candles of `interval_seconds`."""
    candles: dict[int, list[float]] = {}  # bucket -> [open, high, low, close, vol]
    for raw in ticks:
        price = float(raw["price"])
        volume = float(raw.get("volume", 0.0))
        ts = float(raw["ts_epoch"])
        bucket = int(ts // interval_seconds)
        agg = candles.setdefault(bucket, [price, price, price, price, 0.0])
        agg[1] = max(agg[1], price)
        agg[2] = min(agg[2], price)
        agg[3] = price
        agg[4] += volume

    result: list[Candle] = []
    for bucket in sorted(candles):
        o, hi, lo, c, v = candles[bucket]
        result.append(
            Candle(open=o, high=hi, low=lo, close=c, volume=v, ts_epoch=bucket * interval_seconds)
        )
    return result


def true_ranges(candles: Sequence[Candle]) -> np.ndarray:
    """True Range vector for a candle series (empty series -> empty array)."""
    n = len(candles)
    if n == 0:
        return np.array([])
    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    closes = np.array([c.close for c in candles])
    tr = highs - lows
    if n > 1:
        prev_close = np.roll(closes, 1)
        prev_close[0] = candles[0].open
        tr = np.maximum(tr, np.abs(highs - prev_close))
        tr = np.maximum(tr, np.abs(lows - prev_close))
    return tr


def atr_series(candles: Sequence[Candle], period: int = ATR_PERIOD) -> np.ndarray:
    """Wilder-smoothed ATR series (first `period` values use simple mean)."""
    tr = true_ranges(candles)
    n = len(tr)
    if n < period:
        return np.array([])
    out = np.empty(n)
    out[:period] = np.nan
    prev = float(np.mean(tr[:period]))
    out[period - 1] = prev
    for i in range(period, n):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def atr(candles: Sequence[Candle], period: int = ATR_PERIOD) -> float:
    """Latest ATR value, or 0.0 if insufficient data."""
    series = atr_series(candles, period)
    if series.size == 0:
        return 0.0
    last = series[-1]
    return float(last) if np.isfinite(last) else 0.0


def atr_percent(candles: Sequence[Candle], period: int = ATR_PERIOD) -> float:
    """ATR as a % of the latest close (0.0 if no data)."""
    if not candles:
        return 0.0
    value = atr(candles, period)
    return value / candles[-1].close * 100.0 if candles[-1].close else 0.0


def classify_volatility_regime(
    candles: Sequence[Candle],
    calm_pct: float = REGIME_CALM_ATR_PCT,
    high_vol_pct: float = REGIME_HIGH_VOL_ATR_PCT,
) -> str:
    """Classify current volatility regime: CALM | ACTIVE | HIGH_VOL."""
    atr_pct = atr_percent(candles)
    if atr_pct <= 0:
        return "ACTIVE"  # not enough data -> neutral default
    if atr_pct < calm_pct:
        return "CALM"
    if atr_pct > high_vol_pct:
        return "HIGH_VOL"
    return "ACTIVE"


# ---------------------------------------------------------------------------
# Single-candle patterns
# ---------------------------------------------------------------------------
def _is_pin_bar(candle: Candle) -> tuple[bool, str]:
    """Long wick vs small body. Returns (is_pin, subtype)."""
    if candle.body <= 0:
        return False, ""
    if (
        candle.lower_wick >= candle.upper_wick * PIN_BAR_WICK_BODY_RATIO
        and candle.lower_wick >= candle.body * PIN_BAR_WICK_BODY_RATIO
    ):
        return True, PATTERN_HAMMER
    if (
        candle.upper_wick >= candle.lower_wick * PIN_BAR_WICK_BODY_RATIO
        and candle.upper_wick >= candle.body * PIN_BAR_WICK_BODY_RATIO
    ):
        return True, PATTERN_SHOOTING_STAR
    return False, ""


def detect_single_candle_patterns(candle: Candle) -> list[str]:
    """Patterns visible on a single candle (pin bars)."""
    is_pin, subtype = _is_pin_bar(candle)
    if not is_pin:
        return []
    return [PATTERN_PIN_BAR, subtype]


def detect_engulfing(prev: Candle, curr: Candle) -> list[str]:
    """Bullish/bearish engulfing across the last two candles."""
    if prev.body <= 0 or curr.body <= 0:
        return []
    if prev.is_bullish != curr.is_bullish:
        if curr.is_bullish and curr.open <= prev.close and curr.close >= prev.open:
            return [PATTERN_BULL_ENGULFING]
        if not curr.is_bullish and curr.open >= prev.close and curr.close <= prev.open:
            return [PATTERN_BEAR_ENGULFING]
    return []


def detect_sweep(
    candles: Sequence[Candle], tolerance: float = SWEEP_WICK_TOLERANCE_POINTS
) -> list[str]:
    """Wick pierces the prior candle's extreme then closes back inside.

    Returns e.g. ['SWEEP_HIGH'] (bearish signal: failed breakout of prior high).
    """
    if len(candles) < 2:
        return []
    curr, prev = candles[-1], candles[-2]
    patterns: list[str] = []
    if curr.high > prev.high + tolerance and curr.close < prev.high:
        patterns.append(PATTERN_SWEEP_HIGH)
    if curr.low < prev.low - tolerance and curr.close > prev.low:
        patterns.append(PATTERN_SWEEP_LOW)
    return patterns


def detect_all_patterns(candles: Sequence[Candle]) -> list[str]:
    """All pattern signals across the latest candles."""
    if not candles:
        return []
    patterns: list[str] = []
    patterns.extend(detect_single_candle_patterns(candles[-1]))
    if len(candles) >= 2:
        patterns.extend(detect_engulfing(candles[-2], candles[-1]))
        patterns.extend(detect_sweep(candles))
    return patterns
