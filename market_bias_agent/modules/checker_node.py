"""Checker Node — risk guardrail rule engine (Enhancement Phase 4).

Runs all PRD guardrails A-G against a structured signal and returns a
single APPROVED/REJECTED verdict. Stateful: tracks daily PnL for the
Rule-D loss circuit, the per-hour signal rate, and per-strike cooldowns.
Every decision is appended to an audit trail for post-hoc review (Phase 6).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from config.constants import (
    ATR_TARGET_REACHABLE_FACTOR,
    DUPLICATE_ALERT_COOLDOWN_SECONDS,
    MAX_DAILY_LOSS_POINTS,
    MAX_SIGNALS_PER_HOUR,
    MAX_SPREAD_POINTS,
    SIGNAL_RATE_WINDOW_SECONDS,
    STRIKE_COOLDOWN_SECONDS,
)
from config.settings import Settings
from core.logger import get_logger
from core.math_engine import (
    OIMetrics,
    guardrail_atr_sanity,
    guardrail_daily_loss,
    guardrail_duplicate,
    guardrail_pcr_long,
    guardrail_signal_rate,
    guardrail_spread,
    guardrail_strike_cooldown,
)
from core.signals import StructuredSignal

log = get_logger(__name__)

RULE_IDS = ("A", "B", "C", "D", "E", "F", "G")


@dataclass
class RuleVerdict:
    rule_id: str
    passed: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckerVerdict:
    approved: bool
    verdicts: list[RuleVerdict]
    signal: StructuredSignal
    rejected_rules: list[str] = field(default_factory=list)
    overall_reason: str = ""

    @property
    def rejected(self) -> bool:
        return not self.approved


@dataclass
class CheckerContext:
    """Runtime inputs a rule may need beyond the signal itself."""

    metrics: OIMetrics | None = None
    bid: float | None = None
    ask: float | None = None
    atr: float = 0.0
    scalp_mode: bool = True


class CheckerNode:
    def __init__(
        self,
        settings: Settings,
        *,
        max_daily_loss_points: float = MAX_DAILY_LOSS_POINTS,
        max_signals_per_hour: int = MAX_SIGNALS_PER_HOUR,
        signal_rate_window: int = SIGNAL_RATE_WINDOW_SECONDS,
        strike_cooldown: float = STRIKE_COOLDOWN_SECONDS,
        duplicate_cooldown: float = DUPLICATE_ALERT_COOLDOWN_SECONDS,
        max_spread_points: float = MAX_SPREAD_POINTS,
        atr_factor: float = ATR_TARGET_REACHABLE_FACTOR,
    ) -> None:
        self._settings = settings
        self._max_daily_loss = max_daily_loss_points
        self._max_per_hour = max_signals_per_hour
        self._rate_window = signal_rate_window
        self._strike_cooldown = strike_cooldown
        self._dup_cooldown = duplicate_cooldown
        self._max_spread = max_spread_points
        self._atr_factor = atr_factor

        # Rule D state
        self._daily_pnl_points = 0.0
        self._daily_loss_halted = False

        # Rule E state
        self._signal_timestamps: list[float] = []

        # Rule C/E.2 state (strike -> last alert ts)
        self._last_alert_by_strike: dict[float, float] = {}

        self._audit: list[dict[str, Any]] = []
        self._audit_writer: Any = None

    def set_audit_writer(self, writer: Any) -> None:
        """Attach a callable(decision: dict) for Redis/DB write-back (Phase 6)."""
        self._audit_writer = writer

    def record_exit_pnl(self, pnl_points: float) -> None:
        """Feed realized PnL so the Rule-D circuit can halt the day."""
        self._daily_pnl_points += pnl_points
        if self._daily_pnl_points <= -self._max_daily_loss:
            self._daily_loss_halted = True
            log.warning(
                "checker_rule_d_circuit_hit",
                extra={"daily_pnl_points": round(self._daily_pnl_points, 2)},
            )

    def reset_daily(self) -> None:
        """New trading day: clear loss circuit, rate window, cooldowns."""
        self._daily_pnl_points = 0.0
        self._daily_loss_halted = False
        self._signal_timestamps = []
        self._last_alert_by_strike = {}

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------
    def check(
        self, signal: StructuredSignal, context: CheckerContext | None = None
    ) -> CheckerVerdict:
        """Evaluate all rules A-G. Returns APPROVED iff every rule passes."""
        ctx = context or CheckerContext()
        now = time.time()
        verdicts: list[RuleVerdict] = []

        verdicts.append(self._rule_a(signal))
        verdicts.append(self._rule_b(signal, ctx))
        verdicts.append(self._rule_c(signal, now))
        verdicts.append(self._rule_d())
        verdicts.append(self._rule_e(signal, now))
        verdicts.append(self._rule_f(ctx))
        verdicts.append(self._rule_g(signal, ctx))

        rejected = [v.rule_id for v in verdicts if not v.passed]
        approved = not rejected

        if approved:
            now = time.time()
            self._signal_timestamps.append(now)
            self._last_alert_by_strike[signal.strike] = now

        verdict = CheckerVerdict(
            approved=approved,
            verdicts=verdicts,
            signal=signal,
            rejected_rules=rejected,
            overall_reason="ALL_RULES_PASS" if approved else f"REJECTED: {','.join(rejected)}",
        )
        self._audit_append(verdict, now)
        return verdict

    def _rule_a(self, signal: StructuredSignal) -> RuleVerdict:
        """Scalp mode: SL <= 4 pts and Target >= 6 pts."""
        from core.math_engine import guardrail_scale_in_points

        passed, reason = guardrail_scale_in_points(signal.sl, signal.target)
        return RuleVerdict("A", passed, reason)

    def _rule_b(self, signal: StructuredSignal, ctx: CheckerContext) -> RuleVerdict:
        """PCR guard for bullish signals."""
        if signal.direction != "BULLISH" or ctx.metrics is None:
            return RuleVerdict("B", True, "")
        passed, reason = guardrail_pcr_long(ctx.metrics)
        return RuleVerdict("B", passed, reason)

    def _rule_c(self, signal: StructuredSignal, now: float) -> RuleVerdict:
        """Duplicate alert cooldown per strike (120s)."""
        last = self._last_alert_by_strike.get(signal.strike)
        passed, reason = guardrail_duplicate(last, now, self._dup_cooldown)
        return RuleVerdict("C", passed, reason)

    def _rule_d(self) -> RuleVerdict:
        """Daily loss circuit: hard halt when losses exceed the max."""
        if self._daily_loss_halted:
            return RuleVerdict("D", False, "Rule D: daily loss circuit already triggered")
        passed, reason = guardrail_daily_loss(self._daily_pnl_points, self._max_daily_loss)
        return RuleVerdict("D", passed, reason)

    def _rule_e(self, signal: StructuredSignal, now: float) -> RuleVerdict:
        """Signal rate limit + per-strike cooldown."""
        window_start = now - self._rate_window
        recent = sum(1 for ts in self._signal_timestamps if ts >= window_start)
        rate_ok, rate_reason = guardrail_signal_rate(recent, self._max_per_hour, self._rate_window)
        if not rate_ok:
            return RuleVerdict("E", False, rate_reason)
        last = self._last_alert_by_strike.get(signal.strike)
        cooldown_ok, cooldown_reason = guardrail_strike_cooldown(last, now, self._strike_cooldown)
        return RuleVerdict(
            "E",
            cooldown_ok,
            "" if cooldown_ok else cooldown_reason,
            metadata={"signals_in_window": recent},
        )

    def _rule_f(self, ctx: CheckerContext) -> RuleVerdict:
        """Spread guard — only evaluable when live quotes exist."""
        if ctx.bid is None or ctx.ask is None:
            return RuleVerdict("F", True, "")  # backtest: no quotes -> skip
        passed, reason = guardrail_spread(ctx.bid, ctx.ask, self._max_spread)
        return RuleVerdict("F", passed, reason)

    def _rule_g(self, signal: StructuredSignal, ctx: CheckerContext) -> RuleVerdict:
        """ATR sanity: target must be reachable within factor x 1-bar ATR."""
        target_distance = signal.target
        passed, reason = guardrail_atr_sanity(target_distance, ctx.atr, self._atr_factor)
        return RuleVerdict("G", passed, reason, metadata={"atr": round(ctx.atr, 2)})

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------
    def _audit_append(self, verdict: CheckerVerdict, now: float) -> None:
        entry = {
            "ts_epoch": now,
            "signal_id": verdict.signal.signal_id,
            "approved": verdict.approved,
            "rejected_rules": verdict.rejected_rules,
            "reason": verdict.overall_reason,
            "rules": [
                {"rule": v.rule_id, "passed": v.passed, "reason": v.reason}
                for v in verdict.verdicts
            ],
        }
        self._audit.append(entry)
        if self._audit_writer is not None:
            try:
                self._audit_writer(entry)
            except Exception as exc:  # noqa: BLE001 - audit must never block signals
                log.warning("checker_audit_write_failed", extra={"error": str(exc)})
        log.info(
            "checker_verdict",
            extra={
                "signal_id": verdict.signal.signal_id,
                "approved": verdict.approved,
                "rejected": verdict.rejected_rules,
            },
        )

    def audit_trail(self, limit: int = 100) -> list[dict[str, Any]]:
        return list(self._audit[-limit:])

    @property
    def daily_pnl_points(self) -> float:
        return self._daily_pnl_points

    @property
    def daily_loss_halted(self) -> bool:
        return self._daily_loss_halted
