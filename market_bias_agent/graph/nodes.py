"""LangGraph nodes (PRD Module 4). Phase 0 skeletons."""

from __future__ import annotations

from typing import Any


def maker_node(state: dict[str, Any]) -> dict[str, Any]:
    """TODO(Phase 4): Analyst agent -> bias, target, SL, trap_type."""
    return {"maker_output": None}


def checker_node(state: dict[str, Any]) -> dict[str, Any]:
    """TODO(Phase 4): apply risk guardrails A-G. Output APPROVED|REJECTED."""
    return {"checker_verdict": "REJECTED"}
