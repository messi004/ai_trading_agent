"""Mathematical feature engine — PRD Module 2.

All functions are pure (no I/O) so the live pipeline and the backtest
replay harness execute identical logic.

Formulas implemented (exact from PRD):
  1. PCR            = Total Put OI / Total Call OI
  2. Velocity_1m    = OI_current - OI_60s_ago
     Velocity_5m    = OI_current - OI_300s_ago
  3. Level Condition = |Spot - Level| <= 12.0 points
  4. Trigger matrix  = Scalp / Intraday thresholds (profile-scaled)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from config.constants import (
    ATR_TARGET_REACHABLE_FACTOR,
    INTRADAY_VELOCITY_5M_MIN,
    LEVEL_DISTANCE_TOLERANCE,
    MAX_DAILY_LOSS_POINTS,
    MAX_SIGNALS_PER_HOUR,
    MAX_SPREAD_POINTS,
    SCALP_VELOCITY_1M_MIN,
    SIGNAL_RATE_WINDOW_SECONDS,
    STRIKE_COOLDOWN_SECONDS,
)


class MathEngineError(ValueError):
    """Raised on mathematically invalid input."""


@dataclass(frozen=True)
class OIMetrics:
    """Snapshot of option-chain derived metrics."""

    pcr: float
    total_call_oi: float
    total_put_oi: float
    call_velocity_1m: float = 0.0
    put_velocity_1m: float = 0.0
    call_velocity_5m: float = 0.0
    put_velocity_5m: float = 0.0
    max_call_unwind_1m: float = 0.0


@dataclass(frozen=True)
class TriggerResult:
    """Outcome of evaluating the trigger matrix for one tick."""

    triggered: bool
    trigger_type: str = ""  # "SCALP" | "INTRADAY" | ""
    scalp_reason: str = ""
    intraday_reason: str = ""
    metrics: OIMetrics | None = None
    near_level: float | None = None
    details: dict = field(default_factory=dict)


def compute_pcr(total_call_oi: float, total_put_oi: float) -> float:
    """PCR = Total Put OI / Total Call OI. Raises if call OI is zero."""
    if total_call_oi <= 0:
        raise MathEngineError(f"total_call_oi must be > 0, got {total_call_oi}")
    if total_put_oi < 0:
        raise MathEngineError(f"total_put_oi must be >= 0, got {total_put_oi}")
    return total_put_oi / total_call_oi


def oi_velocity(current_oi: float, past_oi: float) -> float:
    """Velocity = OI_current - OI_<window>_ago (positive = build-up, negative = unwind)."""
    return current_oi - past_oi


def compute_oi_metrics(
    total_call_oi: float,
    total_put_oi: float,
    call_oi_60s_ago: float,
    call_oi_300s_ago: float,
    put_oi_60s_ago: float,
    put_oi_300s_ago: float,
) -> OIMetrics:
    """Full OI metrics snapshot per PRD formulas."""
    call_vel_1m = oi_velocity(total_call_oi, call_oi_60s_ago)
    call_vel_5m = oi_velocity(total_call_oi, call_oi_300s_ago)
    put_vel_1m = oi_velocity(total_put_oi, put_oi_60s_ago)
    put_vel_5m = oi_velocity(total_put_oi, put_oi_300s_ago)
    return OIMetrics(
        pcr=compute_pcr(total_call_oi, total_put_oi),
        total_call_oi=total_call_oi,
        total_put_oi=total_put_oi,
        call_velocity_1m=call_vel_1m,
        put_velocity_1m=put_vel_1m,
        call_velocity_5m=call_vel_5m,
        put_velocity_5m=put_vel_5m,
        max_call_unwind_1m=max(-call_vel_1m, 0.0),
    )


def nearest_round_level(spot: float, base: int = 100) -> float:
    """Nearest round psychological level (e.g., 24000, 24100) at `base` spacing."""
    return round(spot / base) * base


def is_at_level(spot: float, level: float, tolerance: float = LEVEL_DISTANCE_TOLERANCE) -> bool:
    """Level Condition = |Spot - Level| <= tolerance."""
    return abs(spot - level) <= tolerance


def round_levels_in_range(
    spot: float, tolerance: float = LEVEL_DISTANCE_TOLERANCE, base: int = 100
) -> list[float]:
    """All round levels within `tolerance` points of the spot."""
    levels: set[float] = set()
    center = nearest_round_level(spot, base)
    step = base
    for candidate in (center, center - step, center + step):
        if is_at_level(spot, candidate, tolerance):
            levels.add(candidate)
    return sorted(levels)


def volume_ratio(volume: float, volume_20ma: float) -> float:
    """Volume / 20-bar moving average."""
    if volume_20ma <= 0:
        return float("inf") if volume > 0 else 0.0
    return volume / volume_20ma


def _scalp_trigger(
    call_vel_1m: float,
    put_vel_1m: float,
    scalp_velocity_threshold: float,
) -> tuple[bool, str]:
    """Scalp: |Velocity_1m| >= threshold (max of call/put)."""
    max_vel_1m = max(abs(call_vel_1m), abs(put_vel_1m))
    if max_vel_1m >= scalp_velocity_threshold:
        return True, f"Velocity_1m={max_vel_1m:,.0f} >= {scalp_velocity_threshold:,.0f}"
    return False, ""


def _intraday_trigger(
    call_vel_5m: float,
    put_vel_5m: float,
    intraday_velocity_threshold: float,
) -> tuple[bool, str]:
    """Intraday: |Velocity_5m| >= threshold (max of call/put)."""
    max_vel_5m = max(abs(call_vel_5m), abs(put_vel_5m))
    if max_vel_5m >= intraday_velocity_threshold:
        return True, f"Velocity_5m={max_vel_5m:,.0f} >= {intraday_velocity_threshold:,.0f}"
    return False, ""


def evaluate_triggers(
    metrics: OIMetrics,
    spot: float,
    *,
    volume: float | None = None,
    volume_20ma: float | None = None,
    scalp_velocity_1m: float = SCALP_VELOCITY_1M_MIN,
    intraday_velocity_5m: float = INTRADAY_VELOCITY_5M_MIN,
    volume_vs_20ma_multiplier: float = 1.5,
    tolerance: float = LEVEL_DISTANCE_TOLERANCE,
) -> TriggerResult:
    """Evaluate both trigger pipelines for a single tick.

    PRD matrix:
      * Scalp   = |Velocity_1m| >= threshold
                  OR (Spot crosses L AND Volume > 1.5x of 20-MA)
      * Intraday = Spot within Level Condition AND |Velocity_5m| >= threshold
    """
    levels = round_levels_in_range(spot, tolerance)
    at_level = len(levels) > 0
    near_level = levels[0] if levels else None

    scalp = False
    scalp_reason = ""
    scalp_vel, scalp_vel_reason = _scalp_trigger(
        metrics.call_velocity_1m, metrics.put_velocity_1m, scalp_velocity_1m
    )
    if scalp_vel:
        scalp, scalp_reason = True, scalp_vel_reason

    volume_cross = False
    if not scalp and at_level and volume is not None and volume_20ma is not None:
        vr = volume_ratio(volume, volume_20ma)
        if vr >= volume_vs_20ma_multiplier:
            volume_cross = True
            scalp = True
            scalp_reason = (
                f"Spot {spot} at level {near_level}, Volume ratio={vr:.2f} >= "
                f"{volume_vs_20ma_multiplier}"
            )

    intraday = False
    intraday_reason = ""
    if at_level:
        vel_5m, vel_5m_reason = _intraday_trigger(
            metrics.call_velocity_5m, metrics.put_velocity_5m, intraday_velocity_5m
        )
        if vel_5m:
            intraday = True
            intraday_reason = f"Spot {spot} within {tolerance}pt of {near_level}; {vel_5m_reason}"

    trigger_type = ""
    if scalp and intraday:
        trigger_type = "SCALP+INTRADAY"
    elif scalp:
        trigger_type = "SCALP"
    elif intraday:
        trigger_type = "INTRADAY"

    return TriggerResult(
        triggered=bool(trigger_type),
        trigger_type=trigger_type,
        scalp_reason=scalp_reason,
        intraday_reason=intraday_reason,
        metrics=metrics,
        near_level=near_level,
        details={
            "at_level": at_level,
            "volume_cross": volume_cross,
            "levels": levels,
        },
    )


def evaluate_triggers_with_regime(
    metrics: OIMetrics,
    spot: float,
    regime: str,
    base_thresholds: dict,
    *,
    volume: float | None = None,
    volume_20ma: float | None = None,
    tolerance: float = LEVEL_DISTANCE_TOLERANCE,
) -> TriggerResult:
    """Run the PRD trigger matrix with regime-scaled thresholds (Phase 2).

    In CALM markets thresholds are raised (filter noise); in HIGH_VOL markets
    they are lowered so fast moves get caught early.
    """
    from core.features import scale_thresholds_by_regime

    scaled = scale_thresholds_by_regime(base_thresholds, regime)
    return evaluate_triggers(
        metrics,
        spot,
        volume=volume,
        volume_20ma=volume_20ma,
        scalp_velocity_1m=scaled["scalp_velocity_1m"],
        intraday_velocity_5m=scaled["intraday_velocity_5m"],
        volume_vs_20ma_multiplier=scaled["volume_vs_20ma"],
        tolerance=tolerance,
    )


def guardrail_scale_in_points(sl: float, target: float) -> tuple[bool, str]:
    """Rule A: Scalp mode requires SL <= 4 pts and Target >= 6 pts."""
    if sl > 4.0 or target < 6.0:
        return False, f"Rule A failed: SL={sl} (max 4), Target={target} (min 6)"
    return True, ""


def guardrail_pcr_long(
    metrics: OIMetrics, pcr_block: float = 0.75, unwind_override: float = 100_000
) -> tuple[bool, str]:
    """Rule B: If PCR < 0.75, block Bullish unless Call Unwinding > 100k."""
    if metrics.pcr >= pcr_block:
        return True, ""
    if metrics.max_call_unwind_1m > unwind_override:
        return True, (
            f"PCR low ({metrics.pcr:.2f}) but Call Unwinding "
            f"{metrics.max_call_unwind_1m:,.0f} > {unwind_override:,.0f} — override allowed"
        )
    return False, f"Rule B: PCR {metrics.pcr:.2f} < {pcr_block} and no Call Unwind override"


def guardrail_duplicate(
    last_alert_ts: float | None, now_ts: float, cooldown: float = 120.0
) -> tuple[bool, str]:
    """Rule C: Block duplicates if an alert was sent in the last `cooldown` seconds."""
    if last_alert_ts is None:
        return True, ""
    if now_ts - last_alert_ts >= cooldown:
        return True, ""
    return False, (
        f"Rule C: duplicate blocked, last alert {now_ts - last_alert_ts:.0f}s ago "
        f"(cooldown {cooldown:.0f}s)"
    )


def guardrail_daily_loss(
    daily_pnl_points: float, max_daily_loss_points: float = MAX_DAILY_LOSS_POINTS
) -> tuple[bool, str]:
    """Rule D: Hard halt all signals if daily simulated loss exceeds the circuit."""
    if daily_pnl_points > -max_daily_loss_points:
        return True, ""
    return False, (
        f"Rule D: daily loss {daily_pnl_points:.1f} pts hit circuit "
        f"{-max_daily_loss_points:.1f} pts — trading halted"
    )


def guardrail_signal_rate(
    signals_this_hour: int,
    max_signals_per_hour: int = MAX_SIGNALS_PER_HOUR,
    window_seconds: int = SIGNAL_RATE_WINDOW_SECONDS,
) -> tuple[bool, str]:
    """Rule E.1: Reject if more than `max_signals_per_hour` already fired in the window."""
    if signals_this_hour < max_signals_per_hour:
        return True, ""
    return False, (
        f"Rule E: signal rate limit hit — {signals_this_hour} in {window_seconds}s "
        f"(max {max_signals_per_hour})"
    )


def guardrail_strike_cooldown(
    last_signal_ts: float | None,
    now_ts: float,
    cooldown_seconds: float = STRIKE_COOLDOWN_SECONDS,
) -> tuple[bool, str]:
    """Rule E.2: Cooldown per strike so a stuck level does not spam alerts."""
    if last_signal_ts is None:
        return True, ""
    if now_ts - last_signal_ts >= cooldown_seconds:
        return True, ""
    return False, (
        f"Rule E: strike cooldown {now_ts - last_signal_ts:.0f}s < {cooldown_seconds:.0f}s"
    )


def guardrail_spread(
    bid: float, ask: float, max_spread_points: float = MAX_SPREAD_POINTS
) -> tuple[bool, str]:
    """Rule F: Reject if current bid-ask spread exceeds threshold (execution risk)."""
    if ask <= bid:
        return False, f"Rule F: invalid quotes bid={bid} ask={ask}"
    spread = ask - bid
    if spread <= max_spread_points:
        return True, ""
    return False, f"Rule F: spread {spread:.2f} pts > {max_spread_points:.2f} pts — reject"


def guardrail_atr_sanity(
    target_distance_points: float,
    atr: float,
    factor: float = ATR_TARGET_REACHABLE_FACTOR,
) -> tuple[bool, str]:
    """Rule G: Reject if the implied target is unreachable within `factor` bars of ATR."""
    if atr <= 0:
        return True, ""  # no ATR data -> cannot sanity check
    if target_distance_points <= factor * atr:
        return True, ""
    return False, (
        f"Rule G: target {target_distance_points:.1f} pts > {factor} x ATR "
        f"{atr:.1f} pts — target unreachable"
    )


def sliding_window(series: Sequence[float], window: int = 30) -> list[float]:
    """Keep the last `window` values (Redis sliding window helper)."""
    values = list(series)
    return values[-window:]


def metrics_from_features(features: dict) -> OIMetrics:
    """Build an OIMetrics snapshot from a live feature dict.

    Accepts the feature dict shape used by the memory/pipeline layer
    (pcr, total_call_oi, call_oi_vel_1m, velocity_5m, ...) with sensible
    defaults so the Checker's PCR / unwind rules stay evaluable.
    """
    total_call = float(features.get("total_call_oi", 0.0))
    total_put = float(features.get("total_put_oi", 0.0))
    pcr = float(features.get("pcr", 1.0))
    call_vel_1m = float(features.get("call_oi_vel_1m", 0.0))
    put_vel_1m = float(features.get("put_oi_vel_1m", 0.0))
    call_vel_5m = float(features.get("call_oi_vel_5m", float(features.get("velocity_5m", 0.0))))
    put_vel_5m = float(features.get("put_oi_vel_5m", float(features.get("velocity_5m", 0.0))))
    if pcr <= 0 and total_call > 0:
        pcr = total_put / total_call if total_call else 1.0
    return OIMetrics(
        pcr=pcr,
        total_call_oi=total_call,
        total_put_oi=total_put,
        call_velocity_1m=call_vel_1m,
        put_velocity_1m=put_vel_1m,
        call_velocity_5m=call_vel_5m,
        put_velocity_5m=put_vel_5m,
        max_call_unwind_1m=max(-call_vel_1m, 0.0),
    )
