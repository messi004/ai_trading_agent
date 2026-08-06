"""Advanced feature primitives (Enhancement Phase 2).

Pure, vectorizable helpers layered on top of the PRD math engine:
  * Volume delta (tick rule)
  * Momentum acceleration (velocity of velocity)
  * OI + price divergence classification
  * Regime-scaled threshold application
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from config.constants import (
    DIVERGENCE,
    MOMENTUM_DT_SECONDS,
    REGIME_THRESHOLD_SCALING,
    VOLATILITY_REGIMES,
)


class FeatureError(ValueError):
    """Raised on invalid feature input."""


@dataclass(frozen=True)
class VolumeDelta:
    """Buying vs selling pressure from the tick rule."""

    buy_volume: float
    sell_volume: float

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def buy_ratio(self) -> float:
        """buy / sell; >= 1 means net buying."""
        if self.sell_volume <= 0:
            return float("inf") if self.buy_volume > 0 else 1.0
        return self.buy_volume / self.sell_volume

    @property
    def bias(self) -> str:
        """BUY | SELL | NEUTRAL."""
        if self.buy_ratio >= 1.2:
            return "BUY"
        if self.buy_ratio <= 0.8:
            return "SELL"
        return "NEUTRAL"


def compute_volume_delta(ticks: Sequence[dict]) -> VolumeDelta:
    """Tick-rule volume delta: up-tick volume = buying, down-tick = selling.

    `ticks` are ordered spot ticks with 'price' and 'volume'.
    """
    buy = 0.0
    sell = 0.0
    prev_price: float | None = None
    for raw in ticks:
        price = float(raw["price"])
        volume = float(raw.get("volume", 0.0))
        if prev_price is not None:
            if price > prev_price:
                buy += volume
            elif price < prev_price:
                sell += volume
            else:
                buy += volume / 2.0
                sell += volume / 2.0
        prev_price = price
    return VolumeDelta(buy_volume=buy, sell_volume=sell)


def momentum_acceleration(
    velocity_current: float,
    velocity_previous: float,
    dt_seconds: float = MOMENTUM_DT_SECONDS,
) -> float:
    """RoC of velocity = (v_now - v_prev) / dt. Positive = accelerating."""
    if dt_seconds <= 0:
        raise FeatureError(f"dt_seconds must be > 0, got {dt_seconds}")
    return (velocity_current - velocity_previous) / dt_seconds


def classify_oi_price_divergence(
    price_change_points: float,
    oi_change_contracts: float,
    price_tolerance: float = 0.5,
    oi_tolerance: float = 1000.0,
) -> str:
    """Classify the OI + price relationship.

    Returns one of: LONG_BUILD, SHORT_COVER, SHORT_BUILD, LONG_UNWIND, NEUTRAL.
    """
    price_up = price_change_points > price_tolerance
    price_down = price_change_points < -price_tolerance
    oi_up = oi_change_contracts > oi_tolerance
    oi_down = oi_change_contracts < -oi_tolerance

    if price_up and oi_up:
        return "LONG_BUILD"
    if price_up and oi_down:
        return "SHORT_COVER"
    if price_down and oi_up:
        return "SHORT_BUILD"
    if price_down and oi_down:
        return "LONG_UNWIND"
    return "NEUTRAL"


def divergence_description(label: str) -> str:
    """Human-readable meaning of a divergence label."""
    return DIVERGENCE.get(label, label)


def scale_thresholds_by_regime(base: dict, regime: str) -> dict:
    """Multiply base trigger thresholds by the regime's scaling factors."""
    if regime not in VOLATILITY_REGIMES:
        raise FeatureError(f"unknown regime {regime!r}; expected {VOLATILITY_REGIMES}")
    scaling = REGIME_THRESHOLD_SCALING[regime]
    return {key: round(value * scaling[key], 2) for key, value in base.items() if key in scaling}


@dataclass
class FeatureBundle:
    """One-tick bundle of all advanced features for the pipeline."""

    regime: str = "ACTIVE"
    atr_pct: float = 0.0
    volume_delta: VolumeDelta | None = None
    divergence: str = "NEUTRAL"
    divergence_description: str = ""
    acceleration: float = 0.0
    patterns: list[str] | None = None


def compute_feature_bundle(
    *,
    candles: Sequence,
    spot_ticks: Sequence[dict],
    price_change_points: float = 0.0,
    oi_change_contracts: float = 0.0,
    call_velocity_1m: float = 0.0,
    call_velocity_1m_prev: float = 0.0,
    dt_seconds: float = MOMENTUM_DT_SECONDS,
) -> FeatureBundle:
    """Aggregate regime, volume delta, divergence and momentum for a tick."""
    from core.candle_engine import (
        atr_percent,
        classify_volatility_regime,
        detect_all_patterns,
    )

    regime = classify_volatility_regime(candles) if candles else "ACTIVE"
    patterns = detect_all_patterns(candles)
    vd = compute_volume_delta(spot_ticks) if spot_ticks else None
    divergence = classify_oi_price_divergence(price_change_points, oi_change_contracts)

    return FeatureBundle(
        regime=regime,
        atr_pct=atr_percent(candles) if candles else 0.0,
        volume_delta=vd,
        divergence=divergence,
        divergence_description=divergence_description(divergence),
        acceleration=momentum_acceleration(call_velocity_1m, call_velocity_1m_prev, dt_seconds),
        patterns=patterns,
    )
