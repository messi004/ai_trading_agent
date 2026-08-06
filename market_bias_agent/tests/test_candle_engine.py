"""Unit tests for candle engine (Phase 2)."""

from __future__ import annotations

from config.constants import (
    PATTERN_BEAR_ENGULFING,
    PATTERN_BULL_ENGULFING,
    PATTERN_HAMMER,
    PATTERN_SHOOTING_STAR,
    PATTERN_SWEEP_HIGH,
    PATTERN_SWEEP_LOW,
)
from core.candle_engine import (
    Candle,
    atr,
    atr_percent,
    build_candles_from_ticks,
    classify_volatility_regime,
    detect_engulfing,
    detect_single_candle_patterns,
    detect_sweep,
    true_ranges,
)


class TestCandleBuilding:
    def test_build_candles_from_ticks(self) -> None:
        ticks = [
            {"price": 100.0, "volume": 10, "ts_epoch": 0},
            {"price": 102.0, "volume": 20, "ts_epoch": 1},
            {"price": 101.0, "volume": 5, "ts_epoch": 59},
            {"price": 105.0, "volume": 50, "ts_epoch": 60},  # next minute
        ]
        candles = build_candles_from_ticks(ticks, interval_seconds=60)
        assert len(candles) == 2
        assert candles[0].open == 100.0
        assert candles[0].high == 102.0
        assert candles[0].low == 100.0
        assert candles[0].close == 101.0
        assert candles[0].volume == 35.0
        assert candles[1].open == 105.0


class TestATR:
    def test_true_ranges(self) -> None:
        candles = [
            Candle(open=10, high=12, low=9, close=11),
            Candle(open=11, high=13, low=10, close=12),
        ]
        tr = true_ranges(candles)
        assert tr[0] == 3.0

    def test_atr_requires_enough_data(self) -> None:
        candles = [Candle(open=10, high=12, low=9, close=11) for _ in range(3)]
        assert atr(candles, period=14) == 0.0

    def test_atr_positive_with_data(self) -> None:
        candles = [Candle(open=10, high=12, low=9, close=11 + i) for i in range(20)]
        assert atr(candles, period=14) > 0.0

    def test_atr_percent(self) -> None:
        candles = [Candle(open=10, high=11, low=9, close=10) for _ in range(20)]
        assert atr_percent(candles) > 0.0


class TestRegime:
    def test_calm_high_vol_classification(self) -> None:
        calm = [Candle(open=24000, high=24001, low=23999, close=24000) for _ in range(20)]
        wild = [Candle(open=24000, high=24030, low=23970, close=24000) for _ in range(20)]
        assert classify_volatility_regime(calm) == "CALM"
        assert classify_volatility_regime(wild) == "HIGH_VOL"

    def test_empty_defaults_active(self) -> None:
        assert classify_volatility_regime([]) == "ACTIVE"


class TestPatterns:
    def test_bullish_engulfing(self) -> None:
        prev = Candle(open=101, high=101, low=99, close=100)  # bearish, body 1
        curr = Candle(open=100, high=103, low=100, close=102.5)  # bullish, body 2.5
        assert PATTERN_BULL_ENGULFING in detect_engulfing(prev, curr)

    def test_bearish_engulfing(self) -> None:
        prev = Candle(open=100, high=102, low=100, close=101)
        curr = Candle(open=101, high=101, low=98, close=99)
        assert PATTERN_BEAR_ENGULFING in detect_engulfing(prev, curr)

    def test_hammer_pin_bar(self) -> None:
        # long lower wick, small body near top
        candle = Candle(open=100, high=100.5, low=98, close=100.2)
        patterns = detect_single_candle_patterns(candle)
        assert PATTERN_HAMMER in patterns

    def test_shooting_star_pin_bar(self) -> None:
        candle = Candle(open=100, high=102, low=99.5, close=99.8)
        patterns = detect_single_candle_patterns(candle)
        assert PATTERN_SHOOTING_STAR in patterns

    def test_sweep_high(self) -> None:
        prev = Candle(open=100, high=101, low=99, close=100.5)
        curr = Candle(open=100.5, high=105.5, low=100, close=100.8)
        assert PATTERN_SWEEP_HIGH in detect_sweep([prev, curr])

    def test_sweep_low(self) -> None:
        prev = Candle(open=100, high=101, low=99, close=100)
        curr = Candle(open=100, high=100.5, low=95, close=99.5)
        assert PATTERN_SWEEP_LOW in detect_sweep([prev, curr])

    def test_sweep_needs_two_candles(self) -> None:
        assert detect_sweep([Candle(open=1, high=2, low=0, close=1)]) == []
