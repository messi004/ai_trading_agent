"""Unit tests for advanced features (Phase 2)."""

from __future__ import annotations

import pytest

from core.features import (
    classify_oi_price_divergence,
    compute_feature_bundle,
    compute_volume_delta,
    divergence_description,
    momentum_acceleration,
    scale_thresholds_by_regime,
)
from core.math_engine import compute_oi_metrics


class TestVolumeDelta:
    def test_uptrend_is_buying_pressure(self) -> None:
        ticks = [
            {"price": 100.0, "volume": 10},
            {"price": 101.0, "volume": 20},
            {"price": 102.0, "volume": 30},
        ]
        vd = compute_volume_delta(ticks)
        assert vd.buy_volume == 50.0
        assert vd.sell_volume == 0.0
        assert vd.delta == 50.0
        assert vd.bias == "BUY"

    def test_downtrend_is_selling_pressure(self) -> None:
        ticks = [
            {"price": 102.0, "volume": 30},
            {"price": 101.0, "volume": 20},
            {"price": 100.0, "volume": 10},
        ]
        vd = compute_volume_delta(ticks)
        assert vd.bias == "SELL"

    def test_neutral_when_sideways(self) -> None:
        ticks = [
            {"price": 100.0, "volume": 10},
            {"price": 100.0, "volume": 10},
        ]
        assert compute_volume_delta(ticks).bias == "NEUTRAL"

    def test_empty_ticks(self) -> None:
        vd = compute_volume_delta([])
        assert vd.delta == 0.0
        assert vd.buy_ratio == 1.0


class TestMomentumAcceleration:
    def test_positive_acceleration(self) -> None:
        assert momentum_acceleration(200, 100, dt_seconds=60) == pytest.approx(100 / 60)

    def test_zero_dt_raises(self) -> None:
        with pytest.raises(ValueError):
            momentum_acceleration(100, 100, dt_seconds=0)


class TestDivergence:
    def test_long_build(self) -> None:
        assert classify_oi_price_divergence(10, 50_000) == "LONG_BUILD"

    def test_short_cover(self) -> None:
        assert classify_oi_price_divergence(10, -50_000) == "SHORT_COVER"

    def test_short_build(self) -> None:
        assert classify_oi_price_divergence(-10, 50_000) == "SHORT_BUILD"

    def test_long_unwind(self) -> None:
        assert classify_oi_price_divergence(-10, -50_000) == "LONG_UNWIND"

    def test_neutral(self) -> None:
        assert classify_oi_price_divergence(0.1, 100) == "NEUTRAL"

    def test_description(self) -> None:
        assert "new longs" in divergence_description("LONG_BUILD")


class TestRegimeScaling:
    def test_calm_raises_thresholds(self) -> None:
        base = {"scalp_velocity_1m": 40_000, "intraday_velocity_5m": 150_000, "volume_vs_20ma": 1.5}
        scaled = scale_thresholds_by_regime(base, "CALM")
        assert scaled["scalp_velocity_1m"] == 52_000
        assert scaled["intraday_velocity_5m"] == 195_000

    def test_high_vol_lowers_thresholds(self) -> None:
        base = {"scalp_velocity_1m": 40_000, "intraday_velocity_5m": 150_000, "volume_vs_20ma": 1.5}
        scaled = scale_thresholds_by_regime(base, "HIGH_VOL")
        assert scaled["scalp_velocity_1m"] == 32_000

    def test_unknown_regime_raises(self) -> None:
        with pytest.raises(ValueError):
            scale_thresholds_by_regime({}, "BANANAS")


class TestFeatureBundle:
    def test_bundle_computes_all_fields(self) -> None:
        ticks = [
            {"price": 24000.0, "volume": 100, "ts_epoch": 0},
            {"price": 24001.0, "volume": 200, "ts_epoch": 1},
        ]
        m = compute_oi_metrics(
            total_call_oi=100_000,
            total_put_oi=95_000,
            call_oi_60s_ago=100_000,
            call_oi_300s_ago=100_000,
            put_oi_60s_ago=95_000,
            put_oi_300s_ago=95_000,
        )
        bundle = compute_feature_bundle(
            candles=[],
            spot_ticks=ticks,
            price_change_points=10,
            oi_change_contracts=50_000,
            call_velocity_1m=m.call_velocity_1m,
            call_velocity_1m_prev=-10_000,
        )
        assert bundle.regime == "ACTIVE"
        assert bundle.divergence == "LONG_BUILD"
        assert bundle.volume_delta is not None
        assert bundle.volume_delta.bias == "BUY"
        assert bundle.acceleration > 0
        assert bundle.patterns == []
