"""Unit tests for the PRD mathematical feature engine."""

import pytest

from core.math_engine import (
    MathEngineError,
    compute_oi_metrics,
    compute_pcr,
    evaluate_triggers,
    evaluate_triggers_with_regime,
    guardrail_pcr_long,
    guardrail_scale_in_points,
    is_at_level,
    nearest_round_level,
    oi_velocity,
    round_levels_in_range,
    volume_ratio,
)


class TestPCR:
    def test_basic_pcr(self) -> None:
        assert compute_pcr(10_000, 9_500) == pytest.approx(0.95)

    def test_zero_call_oi_raises(self) -> None:
        with pytest.raises(MathEngineError):
            compute_pcr(0, 100)

    def test_negative_put_oi_raises(self) -> None:
        with pytest.raises(MathEngineError):
            compute_pcr(10_000, -1)


class TestVelocity:
    def test_oi_velocity_build_up(self) -> None:
        assert oi_velocity(100_000, 60_000) == 40_000

    def test_oi_velocity_unwind(self) -> None:
        assert oi_velocity(50_000, 150_000) == -100_000


class TestLevels:
    def test_nearest_round_level(self) -> None:
        assert nearest_round_level(24005) == 24000
        assert nearest_round_level(24050) == 24000  # banker's rounding
        assert nearest_round_level(24105) == 24100

    def test_is_at_level_within_tolerance(self) -> None:
        assert is_at_level(24012, 24000, tolerance=12.0) is True
        assert is_at_level(23988, 24000, tolerance=12.0) is True
        assert is_at_level(24013, 24000, tolerance=12.0) is False

    def test_round_levels_in_range(self) -> None:
        levels = round_levels_in_range(24005, tolerance=12.0)
        assert 24000 in levels

    def test_volume_ratio(self) -> None:
        assert volume_ratio(150_000, 100_000) == 1.5
        assert volume_ratio(100_000, 0) == float("inf")


class TestTriggers:
    def _metrics(self, call_vel_1m=0, put_vel_1m=0, call_vel_5m=0, put_vel_5m=0):
        return compute_oi_metrics(
            total_call_oi=100_000,
            total_put_oi=95_000,
            call_oi_60s_ago=100_000 - call_vel_1m,
            call_oi_300s_ago=100_000 - call_vel_5m,
            put_oi_60s_ago=95_000 - put_vel_1m,
            put_oi_300s_ago=95_000 - put_vel_5m,
        )

    def test_scalp_velocity_trigger(self) -> None:
        m = self._metrics(call_vel_1m=45_000)
        r = evaluate_triggers(m, spot=23900, scalp_velocity_1m=40_000)
        assert r.triggered
        assert r.trigger_type == "SCALP"

    def test_no_trigger_when_below_threshold(self) -> None:
        m = self._metrics(call_vel_1m=10_000)
        r = evaluate_triggers(m, spot=23900, scalp_velocity_1m=40_000)
        assert not r.triggered

    def test_intraday_level_plus_velocity(self) -> None:
        m = self._metrics(call_vel_5m=160_000)
        r = evaluate_triggers(m, spot=24005, intraday_velocity_5m=150_000)
        assert r.triggered
        assert r.trigger_type == "INTRADAY"

    def test_intraday_requires_level(self) -> None:
        m = self._metrics(call_vel_5m=160_000)
        r = evaluate_triggers(m, spot=24050, intraday_velocity_5m=150_000)
        assert not r.triggered

    def test_intraday_fires_at_premarket_extra_level(self) -> None:
        m = self._metrics(call_vel_5m=160_000)
        # Spot 24050 is NOT a round level, but S1 = 24045 is a premarket level.
        r = evaluate_triggers(m, spot=24050, intraday_velocity_5m=150_000, extra_levels=[24045.0])
        assert r.triggered
        assert r.trigger_type == "INTRADAY"
        assert r.details["at_level"] is True

    def test_extra_level_outside_tolerance_ignored(self) -> None:
        m = self._metrics(call_vel_5m=160_000)
        # 24045 is 5 pts away -> within 12; 24100 is 50 pts away -> ignored.
        r = evaluate_triggers(m, spot=24050, intraday_velocity_5m=150_000, extra_levels=[24100.0])
        assert not r.triggered

    def test_scalp_volume_cross_at_level(self) -> None:
        m = self._metrics()
        r = evaluate_triggers(
            m,
            spot=24005,
            volume=200_000,
            volume_20ma=100_000,
            volume_vs_20ma_multiplier=1.5,
        )
        assert r.triggered
        assert r.details["volume_cross"] is True

    def test_regime_scaling_changes_trigger(self) -> None:
        m = self._metrics(call_vel_1m=45_000)
        base = {"scalp_velocity_1m": 40_000, "intraday_velocity_5m": 150_000, "volume_vs_20ma": 1.5}

        # CALM raises scalp threshold to 52k -> no trigger at 45k
        calm = evaluate_triggers_with_regime(m, spot=23900, regime="CALM", base_thresholds=base)
        assert not calm.triggered

        # HIGH_VOL lowers it to 32k -> trigger
        hot = evaluate_triggers_with_regime(m, spot=23900, regime="HIGH_VOL", base_thresholds=base)
        assert hot.triggered
        assert hot.trigger_type == "SCALP"


class TestGuardrails:
    def test_rule_a_sl_too_wide(self) -> None:
        ok, _ = guardrail_scale_in_points(sl=5, target=6)
        assert not ok

    def test_rule_a_target_too_small(self) -> None:
        ok, _ = guardrail_scale_in_points(sl=3, target=5)
        assert not ok

    def test_rule_a_pass(self) -> None:
        ok, _ = guardrail_scale_in_points(sl=3, target=7)
        assert ok

    def test_rule_b_low_pcr_blocks_long(self) -> None:
        m = compute_oi_metrics(
            total_call_oi=100_000,
            total_put_oi=70_000,
            call_oi_60s_ago=100_000,
            call_oi_300s_ago=100_000,
            put_oi_60s_ago=70_000,
            put_oi_300s_ago=70_000,
        )
        ok, _ = guardrail_pcr_long(m)
        assert not ok

    def test_rule_b_unwind_override(self) -> None:
        m = compute_oi_metrics(
            total_call_oi=100_000,
            total_put_oi=70_000,
            call_oi_60s_ago=250_000,
            call_oi_300s_ago=250_000,
            put_oi_60s_ago=70_000,
            put_oi_300s_ago=70_000,
        )
        ok, _ = guardrail_pcr_long(m, unwind_override=100_000)
        assert ok
