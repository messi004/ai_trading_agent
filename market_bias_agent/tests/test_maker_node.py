"""Unit tests for the Maker node (LLM bias producer + rule-only fallback)."""

from __future__ import annotations

import json

from config.settings import Settings
from core.llm_cache import LLMCache
from core.llm_guardrails import LLMTokenBudget, MakerParseResult
from modules.maker_node import (
    MAKER_SYSTEM_PROMPT,
    MakerNode,
    build_maker_prompt,
    rule_only_signal,
)

VALID_OUTPUT = {
    "direction": "BULLISH",
    "confidence": 0.75,
    "entry_zone": [23550.0, 23580.0],
    "sl": 4.0,
    "target": 6.0,
    "rationale": "pcr 1.05 below 1.2 with call velocity 12k",
    "trap_type": "BREAKOUT",
}


def _settings() -> Settings:
    return Settings(gemini_api_key="test-key", llm_model="gemini-3.5-flash")


def test_rule_only_signal_pcr_lt_1_bullish() -> None:
    signal = rule_only_signal(
        {"pcr": 0.8, "call_oi_vel_1m": 5000.0, "put_oi_vel_1m": -200.0, "spot": 23500.0}
    )
    assert signal["direction"] == "BULLISH"
    assert signal["confidence"] == 0.5
    assert signal["trap_type"] == "NONE"


def test_rule_only_signal_pcr_gt_1_bearish() -> None:
    signal = rule_only_signal(
        {"pcr": 1.3, "call_oi_vel_1m": 100.0, "put_oi_vel_1m": 8000.0, "spot": 23500.0}
    )
    assert signal["direction"] == "BEARISH"


def test_rule_only_signal_neutral_without_velocity() -> None:
    signal = rule_only_signal({"pcr": 1.0, "spot": 23500.0})
    assert signal["direction"] == "NEUTRAL"


def test_build_maker_prompt_includes_features_and_memory() -> None:
    prompt = build_maker_prompt(
        {"pcr": 1.05, "spot": 23500.0},
        {"count": 2, "win_rate": 0.6, "avg_move_points": 12.0},
    )
    assert "pcr=1.05" in prompt
    assert "win-rate 60%" in prompt
    assert MAKER_SYSTEM_PROMPT  # sanity: module-level prompt exists


def test_build_maker_prompt_includes_institutional_context() -> None:
    prompt = build_maker_prompt(
        {
            "pcr": 1.05,
            "spot": 23500.0,
            "structural_bias": "BEARISH",
            "institutional_signals": ["FII net short index futures"],
        },
        None,
    )
    assert "Institutional context (EOD): bias=BEARISH" in prompt
    assert "FII net short index futures" in prompt
    # the raw signals list is not dumped into the features line
    assert "institutional_signals=[" not in prompt


def test_build_maker_prompt_neutral_context_when_no_bias() -> None:
    prompt = build_maker_prompt({"pcr": 1.05, "spot": 23500.0, "structural_bias": "NEUTRAL"}, None)
    assert "No institutional context available." in prompt


def test_build_maker_prompt_includes_premarket_sr_fan() -> None:
    prompt = build_maker_prompt(
        {
            "pcr": 1.05,
            "spot": 24100.0,
            "premarket_pivot": 24100.0,
            "premarket_r1": 24200.0,
            "premarket_s1": 24000.0,
            "premarket_max_pain": 24100.0,
            "premarket_max_pain_zone": [24088.0, 24112.0],
        },
        None,
    )
    assert "Premarket S/R fan" in prompt
    assert "PIVOT = 24,100.0" in prompt
    assert "R1 = 24,200.0" in prompt
    assert "S1 = 24,000.0" in prompt
    assert "MAX_PAIN zone = 24,088.0 – 24,112.0" in prompt


def test_build_maker_prompt_includes_oi_walls() -> None:
    prompt = build_maker_prompt(
        {
            "pcr": 1.05,
            "spot": 24100.0,
            "premarket_oi_resistance": [24200.0, 24250.0],
            "premarket_oi_support": [24000.0, 23950.0],
            "premarket_oi_max_pain": 24100.0,
        },
        None,
    )
    assert "Live OI resistance (call walls): 24,200, 24,250" in prompt
    assert "Live OI support (put walls): 24,000, 23,950" in prompt


def test_build_maker_prompt_no_premarket_when_absent() -> None:
    prompt = build_maker_prompt({"pcr": 1.05, "spot": 23500.0}, None)
    assert "No premarket S/R levels available." in prompt


def test_maker_uses_injected_llm_and_parses_json() -> None:
    node = MakerNode(
        _settings(),
        llm_call=lambda _prompt: json.dumps(VALID_OUTPUT),
    )
    result = node.generate({"pcr": 1.05, "spot": 23500.0})
    assert isinstance(result, MakerParseResult)
    assert result.parsed == VALID_OUTPUT


def test_maker_falls_back_on_invalid_llm_output() -> None:
    node = MakerNode(_settings(), llm_call=lambda _prompt: "not json")
    result = node.generate({"pcr": 0.8, "call_oi_vel_1m": 5000.0, "spot": 23500.0})
    assert isinstance(result, dict)
    assert result["direction"] in ("BULLISH", "BEARISH", "NEUTRAL")


def test_maker_falls_back_on_llm_exception() -> None:
    def boom(_prompt: str) -> str:
        raise RuntimeError("network down")

    node = MakerNode(_settings(), llm_call=boom)
    result = node.generate({"pcr": 1.0, "spot": 23500.0})
    assert isinstance(result, dict)
    assert result["trap_type"] == "NONE"


def test_maker_falls_back_when_budget_exhausted() -> None:
    budget = LLMTokenBudget(daily_budget=10)
    budget.consume(10)
    calls: list[str] = []

    def record(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(VALID_OUTPUT)

    node = MakerNode(_settings(), budget=budget, llm_call=record)
    result = node.generate({"pcr": 1.0, "spot": 23500.0})
    assert isinstance(result, dict)
    assert calls == []  # LLM must not be called when budget exhausted


def test_maker_caches_identical_market_state() -> None:
    cache = LLMCache()
    node = MakerNode(_settings(), cache=cache, llm_call=lambda _p: json.dumps(VALID_OUTPUT))
    node.generate({"pcr": 1.05, "spot": 23500.0, "atr": 40.0})
    node.generate({"pcr": 1.05, "spot": 23500.0, "atr": 40.0})
    assert cache.size == 1


def test_maker_consumes_budget_tokens() -> None:
    budget = LLMTokenBudget(daily_budget=1000)
    node = MakerNode(_settings(), budget=budget, llm_call=lambda _p: json.dumps(VALID_OUTPUT))
    node.generate({"pcr": 1.05, "spot": 23500.0})
    assert budget.spent > 0
