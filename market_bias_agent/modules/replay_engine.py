"""Deterministic signal replay (Enhancement Phase 3).

Replays minute candles through the exact same feature functions used live
(evaluate_triggers_with_regime, divergence, patterns) so backtest == live.

OI metrics come from a real historical OI series (ingested from Breeze via
``scripts.ingest_history`` and read through ``core.data_store.DataStore``),
mirroring the live engine's total-OI + velocity computation.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from config.constants import SCALP_SL_MAX_POINTS, SCALP_TARGET_MIN_POINTS
from config.settings import Settings
from core.candle_engine import Candle, classify_volatility_regime, detect_all_patterns
from core.data_store import DataStore
from core.features import classify_oi_price_divergence
from core.math_engine import OIMetrics, compute_oi_metrics, evaluate_triggers_with_regime


@runtime_checkable
class OIProvider(Protocol):
    """Yields per-bar OI metrics so the trigger matrix can be evaluated."""

    def oi_metrics_for_bar(self, index: int, candle: Candle) -> OIMetrics: ...


class HistoricalOIProvider:
    """Per-bar OI metrics from a real ingested (ts, call_oi, put_oi) series.

    Mirrors the live engine: total OI is the sum across the tracked strikes,
    and 1m/5m velocity compares the current total against the total sampled
    60s / 300s earlier. ``series`` is ``[(ts_epoch, total_call_oi,
    total_put_oi), ...]`` in ascending order, as persisted by
    :class:`core.data_store.DataStore` from real Breeze history.
    """

    def __init__(self, series: list[tuple[float, float, float]]) -> None:
        if not series:
            raise ValueError(
                "HistoricalOIProvider requires a non-empty OI series — "
                "run `python -m scripts.ingest_history --with-oi` first"
            )
        self._ts: list[float] = []
        self._call: list[float] = []
        self._put: list[float] = []
        for ts, call, put in series:
            self._ts.append(ts)
            self._call.append(call)
            self._put.append(put)

    @classmethod
    def from_store(cls, store: DataStore, symbol: str) -> HistoricalOIProvider:
        series = store.load_oi_series(symbol)
        if not series:
            raise ValueError(
                f"no real OI series found for {symbol.upper()} — "
                "ingest it first via `python -m scripts.ingest_history --with-oi`"
            )
        return cls(series)

    def _at_or_before(self, ts: float) -> tuple[float, float]:
        """Latest (call, put) total at or before ``ts`` (0.0 when none)."""
        idx = bisect.bisect_right(self._ts, ts) - 1
        if idx < 0:
            return 0.0, 0.0
        return self._call[idx], self._put[idx]

    def oi_metrics_for_bar(self, index: int, candle: Candle) -> OIMetrics:
        call, put = self._at_or_before(candle.ts_epoch)
        call_60, put_60 = self._at_or_before(candle.ts_epoch - 60.0)
        call_300, put_300 = self._at_or_before(candle.ts_epoch - 300.0)
        return compute_oi_metrics(
            total_call_oi=call,
            total_put_oi=put,
            call_oi_60s_ago=call_60,
            call_oi_300s_ago=call_300,
            put_oi_60s_ago=put_60,
            put_oi_300s_ago=put_300,
        )


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
