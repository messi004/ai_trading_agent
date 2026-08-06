"""LLM guardrails for the Maker node (Enhancement Phase 4).

When the gemini-3.5-flash bias call is wired in, these wrappers enforce:

* Temperature clamped to the deterministic band [0.2, 0.4].
* Strict JSON schema validation with a single retry on malformed output.
* A per-day token/cost budget; once exhausted the pipeline falls back to a
  pure rule-only mode instead of spending beyond cap.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from config.constants import (
    LLM_DAILY_TOKEN_BUDGET,
    LLM_MAX_RETRIES,
    LLM_TEMPERATURE_MAX,
    LLM_TEMPERATURE_MIN,
)
from core.logger import get_logger
from core.signals import validate_maker_signal

log = get_logger(__name__)


def enforce_temperature(temperature: float) -> float:
    """Clamp to [LLM_TEMPERATURE_MIN, LLM_TEMPERATURE_MAX] for deterministic bias."""
    return max(LLM_TEMPERATURE_MIN, min(temperature, LLM_TEMPERATURE_MAX))


@dataclass
class MakerParseResult:
    parsed: dict[str, Any] | None
    error: str = ""


def parse_maker_output(raw: str) -> MakerParseResult:
    """Parse a Maker model response into a validated signal dict.

    Handles JSON fenced in ```json ... ``` blocks as well as bare JSON.
    Returns `parsed=None` + error on any malformed/schema-invalid output.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.strip("`").splitlines()
        if lines and lines[0].strip().lower() in ("json", "jsonc"):
            lines = lines[1:]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return MakerParseResult(None, f"malformed JSON: {exc}")
    if not isinstance(payload, dict):
        return MakerParseResult(None, f"expected JSON object, got {type(payload).__name__}")
    errors = validate_maker_signal(payload)
    if errors:
        return MakerParseResult(None, "; ".join(errors))
    return MakerParseResult(payload)


def parse_maker_output_with_retry(
    call_llm: Any, *, max_retries: int = LLM_MAX_RETRIES
) -> MakerParseResult:
    """Call `call_llm()` and retry-once (or up to `max_retries`) on bad output."""
    last_result = MakerParseResult(None, "no attempt made")
    for attempt in range(max_retries + 1):
        try:
            raw = call_llm()
        except Exception as exc:  # noqa: BLE001 - caller decides how to surface
            last_result = MakerParseResult(None, f"LLM call failed: {exc}")
            continue
        result = parse_maker_output(raw)
        if result.parsed is not None:
            return result
        last_result = result
        if attempt < max_retries:
            log.warning(
                "maker_retry",
                extra={"attempt": attempt + 1, "reason": last_result.error},
            )
    return last_result


class LLMTokenBudget:
    """Daily token spend cap; exhausted -> rule-only fallback."""

    def __init__(
        self, daily_budget: int = LLM_DAILY_TOKEN_BUDGET, day_start_ts: float | None = None
    ) -> None:
        self._budget = daily_budget
        self._day_start = day_start_ts or time.time()
        self._spent = 0
        self._calls = 0

    def consume(self, tokens: int) -> bool:
        """Reserve `tokens` against the budget. False if it would exceed cap."""
        if tokens < 0:
            raise ValueError(f"tokens must be >= 0, got {tokens}")
        if self._spent + tokens > self._budget:
            return False
        self._spent += tokens
        self._calls += 1
        return True

    @property
    def spent(self) -> int:
        return self._spent

    @property
    def remaining(self) -> int:
        return max(self._budget - self._spent, 0)

    @property
    def exhausted(self) -> bool:
        return self._spent >= self._budget

    def reset_daily(self, day_start_ts: float | None = None) -> None:
        self._day_start = day_start_ts or time.time()
        self._spent = 0
        self._calls = 0


def should_use_llm(budget: LLMTokenBudget) -> bool:
    """Fall back to rule-only mode once the daily token budget is exhausted."""
    return not budget.exhausted
