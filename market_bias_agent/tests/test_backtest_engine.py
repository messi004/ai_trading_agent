"""Unit tests for replay + backtest engines (Phase 3)."""

from __future__ import annotations

import types

from core.candle_engine import Candle
from core.cost_model import CostModel
from core.math_engine import OIMetrics, compute_oi_metrics
from modules.backtest_engine import BacktestEngine
from modules.replay_engine import ReplayEngine, direction_from_divergence


class HugeOIProvider:
    """Always reports a big 1m call velocity -> scalp trigger."""

    def oi_metrics_for_bar(self, index: int, candle: Candle) -> OIMetrics:
        return compute_oi_metrics(
            total_call_oi=100_000,
            total_put_oi=95_000,
            call_oi_60s_ago=50_000,
            call_oi_300s_ago=100_000,
            put_oi_60s_ago=95_000,
            put_oi_300s_ago=95_000,
        )


def _candles(pairs: list[tuple[float, float]]) -> list[Candle]:
    """Build candles from (open, close) pairs; high/low extend beyond both."""
    candles: list[Candle] = []
    for i, (open_price, close) in enumerate(pairs):
        candles.append(
            Candle(
                open=open_price,
                high=max(open_price, close) + 2,
                low=min(open_price, close) - 2,
                close=close,
                volume=100_000,
                ts_epoch=float(i * 60),
            )
        )
    return candles


def _settings():
    return types.SimpleNamespace(
        trigger={
            "scalp_velocity_1m": 40_000,
            "intraday_velocity_5m": 150_000,
            "volume_vs_20ma": 1.5,
        }
    )


def test_direction_from_divergence() -> None:
    assert direction_from_divergence(10, 50_000) == "LONG"
    assert direction_from_divergence(-10, -50_000) == "SHORT"
    assert direction_from_divergence(0, 0) == ""


def test_replay_generates_signal_at_level_with_velocity() -> None:
    engine = ReplayEngine(_settings(), HugeOIProvider())
    # neutral bodies for first bars, then a bullish bar near 24000 -> LONG signal
    candles = _candles([(23855, 23855), (23860, 23860), (23999, 24005)])
    signals = engine.run(candles)
    assert len(signals) == 1
    assert signals[0].direction == "LONG"
    assert signals[0].trigger_type == "SCALP"


def test_backtest_target_hit() -> None:
    settings = _settings()
    replay = ReplayEngine(settings, HugeOIProvider(), sl_points=4, target_points=6)
    engine = BacktestEngine(
        settings,
        replay,
        cost_model=CostModel(slippage_points=0.0, cost_per_trade_points=0.0),
        max_hold_bars=10,
    )
    candles = _candles([(23855, 23855), (23860, 23860), (23999, 24005), (24020, 24030)])
    result = engine.run(candles)
    assert result.trades, "expected at least one trade"
    assert result.trades[0].exit_reason == "TARGET_HIT"
    assert result.trades[0].pnl_points > 0


def test_backtest_sl_hit() -> None:
    settings = _settings()
    replay = ReplayEngine(settings, HugeOIProvider(), sl_points=4, target_points=6)
    engine = BacktestEngine(
        settings,
        replay,
        cost_model=CostModel(slippage_points=0.0, cost_per_trade_points=0.0),
        max_hold_bars=10,
    )
    candles = _candles(
        [(23855, 23855), (23860, 23860), (23999, 24005), (23950, 23940), (23900, 23890)]
    )
    result = engine.run(candles)
    assert result.trades, "expected at least one trade"
    assert result.trades[0].exit_reason == "SL_HIT"
    assert result.trades[0].pnl_points < 0


def test_report_metrics_present() -> None:
    settings = _settings()
    engine = BacktestEngine(settings, ReplayEngine(settings, HugeOIProvider()))
    result = engine.run(_candles([(23855, 23855), (23860, 23860), (23999, 24005), (24020, 24030)]))
    report = engine.build_report(result)
    for key in (
        "signals",
        "trades",
        "hit_rate",
        "expectancy_points",
        "profit_factor",
        "max_drawdown_points",
        "max_consecutive_losses",
    ):
        assert key in report


def test_checker_integration_rejects_and_halts() -> None:
    """Wire a CheckerNode: rate limit rejects later signals, Rule D halts day."""
    from modules.checker_node import CheckerNode

    settings = _settings()
    checker = CheckerNode(
        settings=None,
        max_signals_per_hour=1,
        max_daily_loss_points=0.01,  # first loss triggers circuit
    )
    engine = BacktestEngine(
        settings,
        ReplayEngine(settings, HugeOIProvider()),
        cost_model=CostModel(slippage_points=0.0, cost_per_trade_points=0.0),
        max_hold_bars=10,
        checker=checker,
    )
    result = engine.run(
        _candles(
            [
                (23855, 23855),
                (23860, 23860),
                (23999, 24005),
                (24020, 24030),
                (24040, 24050),
                (24060, 24070),
            ]
        )
    )
    report = engine.build_report(result)
    assert report["rejected_by_checker"] >= 1  # rate limit / circuit block later signals
    assert result.rejected_signals  # rejected signals are surfaced


def test_walk_forward_splits_windows() -> None:
    settings = _settings()
    engine = BacktestEngine(settings, ReplayEngine(settings, HugeOIProvider()))
    candles = _candles([(23999, 24000 + (i % 3)) for i in range(50)])
    windows = engine.walk_forward(candles, window_bars=20)
    assert len(windows) == 3
    assert windows[0]["window"] == 1
    assert all(w["trades"] >= 0 for w in windows)
