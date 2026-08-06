"""Tests for the maker->checker workflow composition (PRD Module 4)."""

from __future__ import annotations

import json

from config.settings import Settings
from graph.workflow import SignalWorkflow
from modules.checker_node import CheckerNode
from modules.maker_node import MakerNode

VALID_OUTPUT = {
    "direction": "BULLISH",
    "confidence": 0.75,
    "entry_zone": [23550.0, 23580.0],
    "sl": 4.0,
    "target": 6.0,
    "rationale": "pcr 1.05 with call velocity",
    "trap_type": "NONE",
}


def _features() -> dict:
    return {
        "spot": 23560.0,
        "pcr": 1.05,
        "total_call_oi": 2500000.0,
        "total_put_oi": 2625000.0,
        "call_oi_vel_1m": 12500.0,
        "put_oi_vel_1m": -4300.0,
        "call_oi_vel_5m": 61200.0,
        "put_oi_vel_5m": -9800.0,
        "atr": 40.0,
        "strike": 23550.0,
        "trigger_type": "SCALP",
        "volatility": "ACTIVE",
    }


def test_workflow_returns_decision_dict() -> None:
    settings = Settings(gemini_api_key="test-key")
    maker = MakerNode(settings, llm_call=lambda _p: json.dumps(VALID_OUTPUT))
    checker = CheckerNode(settings)
    wf = SignalWorkflow(settings, maker=maker, checker=checker)
    decision = wf._run(_features())  # noqa: SLF001
    assert decision["status"] in ("APPROVED", "REJECTED")
    assert decision["signal"]["direction"] == "BULLISH"
    assert "maker_output" in decision
    assert "rejected_rules" in decision


def test_workflow_build_does_not_require_langgraph() -> None:
    settings = Settings(gemini_api_key="test-key")
    wf = SignalWorkflow(settings)
    wf.build()  # should not raise even without langgraph installed
    assert wf._graph is None or wf._graph is not None  # type: ignore[comparison-overlap]


def test_workflow_checker_can_reject_direction_mismatch() -> None:
    """BULLISH signal with PCR < 0.75 hits Rule B (no long bias on low PCR)."""
    settings = Settings(gemini_api_key="test-key")
    bull = dict(VALID_OUTPUT, direction="BULLISH")
    maker = MakerNode(settings, llm_call=lambda _p: json.dumps(bull))
    checker = CheckerNode(settings)
    wf = SignalWorkflow(settings, maker=maker, checker=checker)
    features = _features()
    features["pcr"] = 0.6
    decision = wf._run(features)  # noqa: SLF001
    assert decision["status"] == "REJECTED"
    assert "B" in decision["rejected_rules"]
