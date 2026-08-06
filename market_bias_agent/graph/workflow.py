"""LangGraph maker-checker workflow (PRD Module 4).

Composes the LLM Maker node -> rule-based Checker node into a single
`invoke()` that returns an APPROVED/REJECTED decision dict. Uses the
installed langgraph primitives when available but keeps a synchronous,
dependency-light path so the pipeline runs offline and in tests.
"""

from __future__ import annotations

from typing import Any

from config.settings import Settings
from core.logger import get_logger
from core.math_engine import metrics_from_features
from core.signals import StructuredSignal
from modules.checker_node import CheckerContext, CheckerNode
from modules.maker_node import MakerNode, MakerParseResult, rule_only_signal

log = get_logger(__name__)


class SignalWorkflow:
    def __init__(
        self,
        settings: Settings,
        maker: MakerNode | None = None,
        checker: CheckerNode | None = None,
        memory: Any | None = None,
    ) -> None:
        self._settings = settings
        self._maker = maker or MakerNode(settings)
        self._checker = checker or CheckerNode(settings)
        self._memory = memory  # MemoryService (optional)
        self._graph: Any = None

    def build(self) -> None:
        """Compose the node graph (kept for langgraph compatibility)."""
        try:
            from langgraph.graph import StateGraph

            graph = StateGraph(dict)

            def maker(state: dict) -> dict:
                return self._maker_node(state)

            def checker(state: dict) -> dict:
                return self._checker_node(state)

            graph.add_node("maker", maker)
            graph.add_node("checker", checker)
            graph.set_entry_point("maker")
            graph.add_edge("maker", "checker")
            self._graph = graph.compile()
        except Exception as exc:  # noqa: BLE001 - fall back to plain invoke
            log.warning("workflow_graph_unavailable", extra={"error": str(exc)})
            self._graph = None

    async def invoke(
        self,
        features: dict[str, Any],
        memory_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the workflow. Returns a decision dict with APPROVED|REJECTED."""
        if self._graph is not None:
            return await self._graph.ainvoke(
                {"features": features, "memory_context": memory_context}
            )
        return self._run(features, memory_context)

    def run_sync(
        self,
        features: dict[str, Any],
        memory_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synchronous variant for the hot tick path (blocks on the LLM call)."""
        return self._run(features, memory_context)

    # ------------------------------------------------------------------
    # Pipeline (sync, dependency-light)
    # ------------------------------------------------------------------
    def _run(
        self,
        features: dict[str, Any],
        memory_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = {"features": features, "memory_context": memory_context}
        self._maker_node(state)
        return self._checker_node(state)

    def _maker_node(self, state: dict[str, Any]) -> dict[str, Any]:
        features = state.get("features") or {}
        memory_context = state.get("memory_context")
        result = self._maker.generate(features, memory_context)
        if isinstance(result, MakerParseResult) and result.parsed is not None:
            raw = result.parsed
        elif isinstance(result, dict):
            raw = result
        else:
            raw = rule_only_signal(features)
        raw_zone = raw.get("entry_zone")
        if isinstance(raw_zone, list | tuple) and len(raw_zone) == 2:
            try:
                entry_zone = (float(raw_zone[0]), float(raw_zone[1]))
            except (TypeError, ValueError):
                entry_zone = (0.0, 0.0)
        else:
            entry_zone = (0.0, 0.0)
        signal = StructuredSignal(
            direction=raw.get("direction", "NEUTRAL"),
            confidence=float(raw.get("confidence", 0.5)),
            entry_zone=entry_zone,
            sl=float(raw.get("sl", 4.0)),
            target=float(raw.get("target", 6.0)),
            rationale=str(raw.get("rationale", "")),
            trap_type=raw.get("trap_type", "NONE"),
            trigger_type=str(features.get("trigger_type", "SCALP")),
            strike=float(features.get("strike", features.get("spot", 0.0))),
            regime=str(features.get("volatility", "ACTIVE")),
            metadata={
                "pcr": features.get("pcr"),
                "spot": features.get("spot"),
                "near_level": features.get("near_level"),
            },
        )
        state["signal"] = signal
        state["maker_output"] = raw
        return state

    def _checker_node(self, state: dict[str, Any]) -> dict[str, Any]:
        signal: StructuredSignal = state["signal"]
        features = state.get("features") or {}
        context = CheckerContext(
            metrics=metrics_from_features(features),
            atr=float(features.get("atr", 0.0)),
            scalp_mode=signal.trigger_type in ("SCALP", "SCALP+INTRADAY"),
        )
        verdict = self._checker.check(signal, context)
        return {
            "status": "APPROVED" if verdict.approved else "REJECTED",
            "signal": signal.to_dict(),
            "maker_output": state.get("maker_output"),
            "checker_verdict": verdict.overall_reason,
            "rejected_rules": verdict.rejected_rules,
        }

    def maker(self) -> MakerNode:
        return self._maker

    def checker(self) -> CheckerNode:
        return self._checker
