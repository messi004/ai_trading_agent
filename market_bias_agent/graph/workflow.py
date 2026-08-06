"""LangGraph maker-checker workflow (PRD Module 4).

Skeleton for Phase 0. Real node implementations land in Phase 4.
"""

from __future__ import annotations

from typing import Any

from config.settings import Settings


class SignalWorkflow:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._graph = None

    def build(self) -> None:
        """TODO(Phase 4): compose Maker (gemini-3.5-flash) -> Checker (rules A-G) graph."""

    async def invoke(
        self,
        features: dict[str, Any],
        memory_context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the workflow. Returns decision dict with 'status': APPROVED|REJECTED."""
        return {"status": "REJECTED", "reason": "workflow not built (Phase 4 stub)"}
