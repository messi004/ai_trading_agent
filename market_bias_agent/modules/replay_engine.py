"""Deterministic signal replay (Enhancement Phase 3).

Replays minute candles through the exact same feature functions used live
(evaluate_triggers_with_regime, divergence, patterns) so backtest == live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from config.constants import SCALP_SL_MAX_POINTS, SCALP_TARGET_MIN_POINTS
from config.settings import Settings
from core.candle_engine import Candle, classify_volatility_regime, detect_all_patterns
from core.features import classify_oi_price_divergence
from core.math_engine import OIMetrics, compute_oi_metrics, evaluate_triggers_with_regime


@runtime_checkable
class OIProvider(Protocol):
    """Yields per-bar OI metrics so the trigger matrix can be evaluated."""

    def oi_metrics_for_bar(self, index: int, candle: Candle) -> OIMetrics: ...


class SyntheticOIProvider:
    """Deterministic seeded OI so backtests are reproducible without real data.

    Occasional large velocity spikes make realistic signals fire.
    """

    def __init__(
        self, seed: int = 42, base_call_oi: float = 100_000, base_put_oi: float = 95_000
    ) -> None:
        import random

        self._rng = random.Random(seed)
        self._base_call = base_call_oi
        self._base_put = base_put_oi
        self._last_call = base_call_oi
        self._last_put = base_put_oi

    def _step(self, last: float) -> float:
        delta = self._rng.uniform(-15_000, 15_000)
        if self._rng.random() < 0.03:  # 3% chance of a velocity spike
            delta += self._rng.choice([-1, 1]) * self._rng.uniform(50_000, 120_000)
        return max(last + delta, 10_000)

    def oi_metrics_for_bar(self, index: int, candle: Candle) -> OIMetrics:
        call = self._step(self._last_call)
        put = self._step(self._last_put)
        metrics = compute_oi_metrics(
            total_call_oi=call,
            total_put_oi=put,
            call_oi_60s_ago=self._last_call,
            call_oi_300s_ago=self._last_call - self._rng.uniform(-30_000, 30_000),
            put_oi_60s_ago=self._last_put,
            put_oi_300s_ago=self._last_put - self._rng.uniform(-30_000, 30_000),
        )
        self._last_call, self._last_put = call, put
        return metrics


def direction_from_divergence(price_change_points: float, oi_change_contracts: float) -> str:
    """Map OI+price divergence to a trade direction ('' = no edge)."""
    label = classify_oi_price_divergence(price_change_points, oi_change_contracts)
    if label in ("LONG_BUILD", "SHORT_COVER"):
        return "LONG"
    if label in ("SHORT_BUILD", "LONG_UNWIND"):
        return "SHORT"
    return ""


@dataclass
class BacktestSignal:
    ts_epoch: float
    direction: str
    trigger_type: str
    entry: float
    sl: float
    target: float
    regime: str = "ACTIVE"
    divergence: str = "NEUTRAL"
    patterns: list[str] = field(default_factory=list)
    metrics: OIMetrics | None = None


class ReplayEngine:
    def __init__(
        self,
        settings: Settings,
        oi_provider: OIProvider,
        thresholds: dict | None = None,
        sl_points: float = SCALP_SL_MAX_POINTS,
        target_points: float = SCALP_TARGET_MIN_POINTS,
    ) -> None:
        self._settings = settings
        self._oi = oi_provider
        self._thresholds = thresholds or dict(settings.trigger)
        self._sl = sl_points
        self._target = target_points

    def run(self, candles: list[Candle]) -> list[BacktestSignal]:
        """Replay bars -> signals. One signal per triggering bar (no dedupe)."""
        signals: list[BacktestSignal] = []
        for i, candle in enumerate(candles):
            metrics = self._oi.oi_metrics_for_bar(i, candle)
            regime = classify_volatility_regime(candles[max(0, i - 40) : i + 1])
            trigger = evaluate_triggers_with_regime(
                metrics,
                spot=candle.close,
                regime=regime,
                base_thresholds=self._thresholds,
            )
            if not trigger.triggered:
                continue
            price_change = candle.close - candle.open
            oi_change = metrics.call_velocity_1m
            direction = direction_from_divergence(price_change, oi_change)
            if not direction:
                continue
            patterns = detect_all_patterns(candles[max(0, i - 2) : i + 1])
            divergence = classify_oi_price_divergence(price_change, oi_change)
            signals.append(
                BacktestSignal(
                    ts_epoch=candle.ts_epoch,
                    direction=direction,
                    trigger_type=trigger.trigger_type,
                    entry=candle.close,
                    sl=self._sl,
                    target=self._target,
                    regime=regime,
                    divergence=divergence,
                    patterns=patterns,
                    metrics=metrics,
                )
            )
        return signals
