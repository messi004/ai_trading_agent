"""Unit tests for pre-market level calculations + PreMarketEngine (PRD Module 6)."""

from __future__ import annotations

import types

import fakeredis
import pytest

from core.premarket_levels import (
    PreMarketLevelsError,
    combined_oi_by_strike,
    compute_pivot_sr,
    find_max_pain_strike,
    max_pain_zone,
    nearest_level,
    psychological_levels_around,
    session_bounds_from_ticks,
)
from core.redis_manager import RedisManager
from modules.premarket_engine import PreMarketEngine


class TestPivotSR:
    def test_classic_pivots(self) -> None:
        # H=24200, L=23800, C=24000 -> P=24000, R1=24200, S1=23800
        sr = compute_pivot_sr(24200.0, 23800.0, 24000.0)
        assert sr.pivot == pytest.approx(24000.0)
        assert sr.r1 == pytest.approx(24200.0)
        assert sr.s1 == pytest.approx(23800.0)
        assert sr.r2 == pytest.approx(24400.0)
        assert sr.s2 == pytest.approx(23600.0)

    def test_psych_levels_bracket_pivot(self) -> None:
        sr = compute_pivot_sr(24050.0, 23950.0, 24000.0)  # pivot = 24000
        assert sr.psych_support == 23900.0
        assert sr.psych_resistance == 24100.0

    def test_ordered_levels_ascending(self) -> None:
        sr = compute_pivot_sr(24200.0, 23800.0, 24000.0)
        levels = sr.ordered_levels()
        assert levels == sorted(levels)
        assert levels[0] < levels[-1]

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(PreMarketLevelsError):
            compute_pivot_sr(23800.0, 24200.0, 24000.0)

    def test_non_positive_rejected(self) -> None:
        with pytest.raises(PreMarketLevelsError):
            compute_pivot_sr(0.0, -1.0, 1.0)


class TestMaxPain:
    def test_combined_oi_union(self) -> None:
        combined = combined_oi_by_strike({24000: 100.0}, {24100: 50.0, 24000: 20.0})
        assert combined == {24000: 120.0, 24100: 50.0}

    def test_find_max_pain_strike(self) -> None:
        assert find_max_pain_strike({24000: 10.0, 24100: 99.0}) == 24100
        assert find_max_pain_strike({}) is None

    def test_max_pain_zone_band(self) -> None:
        zone = max_pain_zone({24000: 10.0, 24100: 99.0})
        assert zone == pytest.approx((24088.0, 24112.0))

    def test_empty_oi_no_zone(self) -> None:
        assert max_pain_zone({}) is None


class TestHelpers:
    def test_psych_levels_around(self) -> None:
        levels = psychological_levels_around(24005.0, span=1)
        assert 23900.0 in levels and 24000.0 in levels and 24100.0 in levels

    def test_nearest_level(self) -> None:
        assert nearest_level(24005.0, [23900.0, 24000.0, 24100.0]) == 24000.0
        assert nearest_level(5.0, []) is None

    def test_session_bounds_from_ticks(self) -> None:
        ticks = [{"price": 23800}, {"price": 24200}, {"price": 24000}]
        high, low, close = session_bounds_from_ticks(ticks)
        assert (high, low, close) == (24200.0, 23800.0, 24000.0)

    def test_session_bounds_empty(self) -> None:
        assert session_bounds_from_ticks([]) is None
        assert session_bounds_from_ticks([{"price": None}]) is None


class TestPreMarketEngine:
    def _make_engine(self) -> tuple[PreMarketEngine, RedisManager, fakeredis.FakeRedis]:
        settings = types.SimpleNamespace()
        mgr = RedisManager(settings)
        fake = fakeredis.FakeRedis(decode_responses=True)
        mgr.client = fake
        return PreMarketEngine(settings, redis=mgr), mgr, fake

    def test_run_computes_and_stores_levels(self) -> None:
        engine, mgr, _ = self._make_engine()
        mgr.push_spot_tick({"price": 23800, "ts_epoch": 1.0})
        mgr.push_spot_tick({"price": 24200, "ts_epoch": 2.0})
        mgr.push_spot_tick({"price": 24000, "ts_epoch": 3.0})
        mgr.set_strikes([23900, 24000, 24100])
        mgr.push_call_oi(24100, 900.0)
        mgr.push_put_oi(24100, 950.0)

        result = engine.run()
        assert result["prev_close"] == 24000.0
        assert result["pivot"] == pytest.approx(24000.0)
        assert result["max_pain"]["strike"] == 24100
        # persisted to Redis
        stored = mgr.get_pre_market_levels()
        assert stored is not None and stored["pivot"] == result["pivot"]

    def test_run_without_prev_data_returns_empty(self) -> None:
        engine, _, _ = self._make_engine()
        assert engine.run() == {}

    def test_run_ignores_missing_oi(self) -> None:
        engine, mgr, _ = self._make_engine()
        mgr.push_spot_tick({"price": 24000, "ts_epoch": 1.0})
        result = engine.run()
        assert "max_pain" not in result
        assert result["s1"] > 0
