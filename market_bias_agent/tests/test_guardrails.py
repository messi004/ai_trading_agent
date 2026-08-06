"""Phase 4 guardrail tests: Checker rules D-G, signal schema, LLM guardrails."""

import json

from core.llm_guardrails import (
    LLMTokenBudget,
    enforce_temperature,
    parse_maker_output,
    parse_maker_output_with_retry,
    should_use_llm,
)
from core.math_engine import (
    OIMetrics,
    guardrail_atr_sanity,
    guardrail_daily_loss,
    guardrail_duplicate,
    guardrail_signal_rate,
    guardrail_spread,
    guardrail_strike_cooldown,
)
from core.signals import StructuredSignal, side_to_direction, validate_maker_signal
from modules.checker_node import CheckerContext, CheckerNode

MAKER_OK = {
    "direction": "BULLISH",
    "confidence": 0.8,
    "entry_zone": [23999, 24002],
    "sl": 4.0,
    "target": 6.0,
    "rationale": "call build-up at level",
    "trap_type": "BREAKOUT",
}


def make_signal(**overrides) -> StructuredSignal:
    base = {
        "direction": "BULLISH",
        "confidence": 0.8,
        "entry_zone": (23999, 24002),
        "sl": 4.0,
        "target": 6.0,
        "rationale": "test",
        "trap_type": "NONE",
        "ts_epoch": 1000.0,
        "strike": 24000,
    }
    base.update(overrides)
    return StructuredSignal(**base)


def make_checker(**kwargs) -> CheckerNode:
    return CheckerNode(settings=None, **kwargs)


class TestSignalSchema:
    def test_valid_signal(self) -> None:
        assert validate_maker_signal(dict(MAKER_OK)) == []

    def test_missing_field(self) -> None:
        raw = dict(MAKER_OK)
        raw.pop("rationale")
        errors = validate_maker_signal(raw)
        assert any("missing field: rationale" in e for e in errors)

    def test_bad_direction(self) -> None:
        raw = dict(MAKER_OK, direction="UP")
        assert any("direction must be one of" in e for e in validate_maker_signal(raw))

    def test_confidence_range(self) -> None:
        raw = dict(MAKER_OK, confidence=1.5)
        assert any("confidence must be in [0,1]" in e for e in validate_maker_signal(raw))

    def test_entry_zone_order(self) -> None:
        raw = dict(MAKER_OK, entry_zone=[24005, 23995])
        errors = validate_maker_signal(raw)
        assert any("entry_zone low" in e and "high" in e for e in errors)

    def test_bad_trap_type(self) -> None:
        raw = dict(MAKER_OK, trap_type="DOUBLE_TOP")
        assert any("trap_type must be one of" in e for e in validate_maker_signal(raw))

    def test_side_mapping(self) -> None:
        assert side_to_direction("LONG") == "BULLISH"
        assert side_to_direction("SHORT") == "BEARISH"
        assert side_to_direction("flat") == "NEUTRAL"
        assert make_signal().side() == "LONG"
        assert make_signal(direction="BEARISH").side() == "SHORT"
        assert make_signal(direction="NEUTRAL").side() == ""


class TestRuleDEfg:
    def test_daily_loss_ok(self) -> None:
        ok, _ = guardrail_daily_loss(-50.0, max_daily_loss_points=100.0)
        assert ok

    def test_daily_loss_circuit(self) -> None:
        ok, reason = guardrail_daily_loss(-120.0, max_daily_loss_points=100.0)
        assert not ok
        assert "Rule D" in reason

    def test_signal_rate_limit(self) -> None:
        assert guardrail_signal_rate(4, max_signals_per_hour=5)[0]
        ok, reason = guardrail_signal_rate(5, max_signals_per_hour=5)
        assert not ok
        assert "Rule E" in reason

    def test_strike_cooldown(self) -> None:
        assert guardrail_strike_cooldown(None, 500.0)[0]
        assert guardrail_strike_cooldown(100.0, 500.0, cooldown_seconds=300)[0]
        ok, reason = guardrail_strike_cooldown(450.0, 500.0, cooldown_seconds=300)
        assert not ok
        assert "cooldown" in reason

    def test_duplicate_cooldown(self) -> None:
        ok, _ = guardrail_duplicate(100.0, 500.0, cooldown=120.0)
        assert ok
        ok, reason = guardrail_duplicate(450.0, 500.0, cooldown=120.0)
        assert not ok
        assert "Rule C" in reason

    def test_spread_guard(self) -> None:
        ok, _ = guardrail_spread(23999.0, 24000.5, max_spread_points=2.0)
        assert ok
        ok, reason = guardrail_spread(23999.0, 24005.0, max_spread_points=2.0)
        assert not ok
        assert "Rule F" in reason
        ok, reason = guardrail_spread(24005.0, 24000.0)
        assert not ok
        assert "invalid quotes" in reason

    def test_atr_sanity(self) -> None:
        ok, _ = guardrail_atr_sanity(4.0, atr=3.0, factor=2.0)
        assert ok
        ok, reason = guardrail_atr_sanity(20.0, atr=3.0, factor=2.0)
        assert not ok
        assert "Rule G" in reason
        ok, _ = guardrail_atr_sanity(20.0, atr=0.0)
        assert ok  # no ATR data -> cannot sanity check


class TestCheckerNode:
    def test_approves_clean_signal(self) -> None:
        checker = make_checker()
        verdict = checker.check(make_signal())
        assert verdict.approved
        assert verdict.rejected_rules == []

    def test_rejects_rule_a(self) -> None:
        checker = make_checker()
        verdict = checker.check(make_signal(sl=6.0, target=5.0))
        assert not verdict.approved
        assert "A" in verdict.rejected_rules

    def test_rejects_rule_b_on_low_pcr(self) -> None:
        checker = make_checker()
        low_pcr = OIMetrics(
            pcr=0.5, total_call_oi=100_000, total_put_oi=50_000, max_call_unwind_1m=0.0
        )
        verdict = checker.check(make_signal(direction="BULLISH"), CheckerContext(metrics=low_pcr))
        assert not verdict.approved
        assert "B" in verdict.rejected_rules

    def test_rule_b_bypasses_for_bearish(self) -> None:
        checker = make_checker()
        low_pcr = OIMetrics(
            pcr=0.5, total_call_oi=100_000, total_put_oi=50_000, max_call_unwind_1m=0.0
        )
        verdict = checker.check(make_signal(direction="BEARISH"), CheckerContext(metrics=low_pcr))
        assert verdict.approved

    def test_rule_c_duplicate_same_strike(self) -> None:
        checker = make_checker(duplicate_cooldown=3600)
        assert checker.check(make_signal(strike=24000)).approved
        verdict = checker.check(make_signal(strike=24000))
        assert not verdict.approved
        assert "C" in verdict.rejected_rules

    def test_rule_d_circuit_halt(self) -> None:
        checker = make_checker(max_daily_loss_points=100.0)
        checker.record_exit_pnl(-60.0)
        assert checker.check(make_signal()).approved
        checker.record_exit_pnl(-60.0)
        verdict = checker.check(make_signal())
        assert not verdict.approved
        assert "D" in verdict.rejected_rules
        assert checker.daily_loss_halted

    def test_rule_e_rate_limit(self) -> None:
        checker = make_checker(max_signals_per_hour=2, signal_rate_window=3600)
        checker.check(make_signal(strike=24000))
        checker.check(make_signal(strike=24100))
        verdict = checker.check(make_signal(strike=24200))
        assert not verdict.approved
        assert "E" in verdict.rejected_rules

    def test_rule_f_spread(self) -> None:
        checker = make_checker(max_spread_points=2.0)
        verdict = checker.check(make_signal(), CheckerContext(bid=23999.0, ask=24005.0))
        assert not verdict.approved
        assert "F" in verdict.rejected_rules
        # no quotes -> rule skipped
        assert checker.check(make_signal()).approved

    def test_rule_g_atr_sanity(self) -> None:
        checker = make_checker(atr_factor=2.0)
        verdict = checker.check(make_signal(target=20.0), CheckerContext(atr=3.0))
        assert not verdict.approved
        assert "G" in verdict.rejected_rules

    def test_reset_daily(self) -> None:
        checker = make_checker(max_daily_loss_points=100.0)
        checker.record_exit_pnl(-200.0)
        assert checker.daily_loss_halted
        checker.reset_daily()
        assert not checker.daily_loss_halted
        assert checker.check(make_signal()).approved

    def test_audit_trail(self) -> None:
        checker = make_checker()
        checker.check(make_signal())
        trail = checker.audit_trail()
        assert len(trail) == 1
        assert trail[0]["approved"] is True
        assert trail[0]["rules"][0]["rule"] == "A"


class TestLLMGuardrails:
    def test_temperature_clamped(self) -> None:
        assert enforce_temperature(0.3) == 0.3
        assert enforce_temperature(0.1) == 0.2
        assert enforce_temperature(1.0) == 0.4

    def test_parse_bare_json(self) -> None:
        result = parse_maker_output(json.dumps(MAKER_OK))
        assert result.parsed is not None
        assert result.parsed["direction"] == "BULLISH"

    def test_parse_fenced_json(self) -> None:
        raw = "```json\n" + json.dumps(MAKER_OK) + "\n```"
        result = parse_maker_output(raw)
        assert result.parsed is not None

    def test_parse_malformed(self) -> None:
        result = parse_maker_output("{not json")
        assert result.parsed is None
        assert "malformed JSON" in result.error

    def test_parse_schema_invalid(self) -> None:
        bad = dict(MAKER_OK, confidence=3.0)
        result = parse_maker_output(json.dumps(bad))
        assert result.parsed is None
        assert "confidence" in result.error

    def test_retry_once_then_fail(self) -> None:
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            return "{broken"

        result = parse_maker_output_with_retry(flaky, max_retries=1)
        assert result.parsed is None
        assert calls["n"] == 2  # initial + one retry

    def test_retry_succeeds_on_second(self) -> None:
        calls = {"n": 0}

        def recover() -> str:
            calls["n"] += 1
            return json.dumps(MAKER_OK) if calls["n"] > 1 else "{broken"

        result = parse_maker_output_with_retry(recover, max_retries=1)
        assert result.parsed is not None
        assert calls["n"] == 2

    def test_budget_cap(self) -> None:
        budget = LLMTokenBudget(daily_budget=100)
        assert budget.consume(60)
        assert not budget.consume(60)
        assert budget.remaining == 40
        assert should_use_llm(budget) is True
        budget.consume(40)
        assert budget.exhausted
        assert should_use_llm(budget) is False
        budget.reset_daily()
        assert not budget.exhausted
