"""Maker Node — LLM analyst (PRD Module 4).

Turns mathematical features + Qdrant memory context into a structured
trade-bias signal (direction, confidence, entry zone, SL, target, trap type)
using Google Gemini via its OpenAI-compatible endpoint.

Guardrails (Enhancement Phase 4 / 6.3):
  * Temperature clamped to the deterministic band [0.2, 0.4].
  * Strict JSON schema validation with retry-once on malformed output.
  * Per-day token budget; once exhausted the pipeline falls back to a pure
    rule-only signal instead of spending beyond the cap.
  * Response cache keyed on the market state (Phase 7).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from config.constants import (
    GEMINI_BASE_URL,
    LLM_DAILY_TOKEN_BUDGET,
    LLM_MAX_RETRIES,
    QDRANT_SIMILAR_LIMIT,
)
from config.settings import Settings
from core.llm_cache import LLMCache, cached_maker_output
from core.llm_guardrails import (
    LLMTokenBudget,
    MakerParseResult,
    enforce_temperature,
    parse_maker_output_with_retry,
    should_use_llm,
)
from core.logger import get_logger

log = get_logger(__name__)


class MakerOutputError(Exception):
    """Raised when the LLM output fails schema validation (not cached)."""


MAKER_SYSTEM_PROMPT = """You are an institutional Nifty 50 index-options analyst.

You are given live mathematical features (PCR, OI velocity, spot vs levels,
volume delta, volatility regime), similar historical trap situations from a
vector memory, an optional EOD institutional (FII/PRO) structural bias, and
the next-day premarket S/R fan + max-pain pinning zone.

The institutional bias is advisory only: treat it as a tilt in your confidence
and rationale, never as a mandate. When the live features conflict with the
bias, prefer the live features. The premarket S/R levels mark likely
rejection/bounce zones — factor them into your entry/exit reasoning, but never
invent levels far from the given spot.

Return STRICT JSON only (no markdown, no commentary) with this exact schema:
{
  "direction": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": 0.0-1.0,
  "entry_zone": [low, high],
  "sl": float,
  "target": float,
  "rationale": "short reasoning string",
  "trap_type": "BULL_TRAP" | "BEAR_TRAP" | "BREAKOUT" | "NONE"
}
Rules: sl and target are RISK DISTANCES in index points measured from the
entry mid-point (scalp example: sl=4, target=6). Do NOT output absolute
index price levels. target must be positive. Never invent levels far from
the given spot. Be conservative: NEUTRAL is valid when confluence is weak."""


def _memory_context_text(memory_context: dict[str, Any] | None) -> str:
    """Compact text of similar historical situations for the prompt."""
    if not memory_context:
        return "No similar historical situations available."
    count = memory_context.get("count", 0)
    win_rate = memory_context.get("win_rate", 0.0)
    avg_move = memory_context.get("avg_move_points", 0.0)
    hits = memory_context.get("hits", [])
    lines = [
        f"Similar historical situations: {count} | win-rate {win_rate:.0%} | "
        f"avg move {avg_move:+.1f} pts",
    ]
    for hit in hits[:QDRANT_SIMILAR_LIMIT]:
        lines.append(f"  - outcome={hit.get('outcome')} score={hit.get('score', 0.0):.3f}")
    return "\n".join(lines)


def _institutional_context_text(features: dict[str, Any]) -> str:
    """EOD institutional (FII/PRO) bias for the prompt.

    The bias is advisory only — it tilts the analysis, it is never a mandate.
    """
    bias = str(features.get("structural_bias") or "NEUTRAL").upper()
    signals = features.get("institutional_signals") or []
    if bias == "NEUTRAL" and not signals:
        return "No institutional context available."
    lines = [f"Institutional context (EOD): bias={bias}"]
    for signal in signals[:5]:
        lines.append(f"  • {signal}")
    lines.append(
        "Treat this bias as a tilt in your confidence, not an instruction. "
        "Confirm with live features before any directional call."
    )
    return "\n".join(lines)


def _premarket_context_text(features: dict[str, Any]) -> str:
    """Structured S/R fan + live OI-profile walls + max-pain band for the prompt."""
    keys = (
        "premarket_pivot",
        "premarket_r1",
        "premarket_r2",
        "premarket_s1",
        "premarket_s2",
        "premarket_max_pain",
    )
    present = [(k, features.get(k)) for k in keys if features.get(k) is not None]
    zone = features.get("premarket_max_pain_zone")
    oi_r = features.get("premarket_oi_resistance")
    oi_s = features.get("premarket_oi_support")
    if not present and not zone and not oi_r and not oi_s:
        return "No premarket S/R levels available."
    lines = ["Premarket S/R fan (from previous session):"]
    for key, value in present:
        label = key.replace("premarket_", "").upper()
        lines.append(f"  • {label} = {value:,.1f}")
    if zone:
        lines.append(f"  • MAX_PAIN zone = {zone[0]:,.1f} – {zone[1]:,.1f}")
    if oi_r:
        lines.append("Live OI resistance (call walls): " + ", ".join(f"{lv:,.0f}" for lv in oi_r))
    if oi_s:
        lines.append("Live OI support (put walls): " + ", ".join(f"{lv:,.0f}" for lv in oi_s))
    lines.append(
        "The S/R fan marks likely rejection/bounce zones; OI walls show where "
        "option writers currently defend price. Weight price action around both."
    )
    return "\n".join(lines)


def build_maker_prompt(features: dict[str, Any], memory_context: dict[str, Any] | None) -> str:
    """Human-readable feature snapshot for the LLM."""
    prompt_features = {k: v for k, v in features.items() if k not in ("institutional_signals",)}
    parts = [f"{key}={value}" for key, value in sorted(prompt_features.items())]
    features_text = ", ".join(parts) if parts else "{}"
    return (
        "Live features:\n"
        f"{features_text}\n\n"
        "Historical memory:\n"
        f"{_memory_context_text(memory_context)}\n\n"
        "Institutional context:\n"
        f"{_institutional_context_text(features)}\n\n"
        "Premarket context:\n"
        f"{_premarket_context_text(features)}\n\n"
        "Produce the trade-bias JSON now."
    )


def rule_only_signal(features: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fallback when the LLM budget is exhausted.

    Pure rule-based bias so the pipeline never blocks on a dead budget:
      * PCR > 1.0 with put OI velocity -> mildly bearish bias
      * PCR < 1.0 with call OI velocity -> mildly bullish bias
      * otherwise NEUTRAL.
    """
    pcr = float(features.get("pcr", 1.0))
    call_vel = float(features.get("call_oi_vel_1m", 0.0))
    put_vel = float(features.get("put_oi_vel_1m", 0.0))
    spot = float(features.get("spot", 0.0))

    if pcr <= 0:
        direction = "NEUTRAL"
    elif pcr < 1.0 and call_vel > 0 and call_vel >= abs(put_vel):
        direction = "BULLISH"
    elif pcr > 1.0 and put_vel > 0 and put_vel >= abs(call_vel):
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    return {
        "direction": direction,
        "confidence": 0.5,
        "entry_zone": [spot - 2.0, spot + 2.0],
        "sl": 4.0,
        "target": 6.0,
        "rationale": "rule-only fallback (LLM budget exhausted)",
        "trap_type": "NONE",
    }


class MakerNode:
    """LLM maker with deterministic rule-only fallback."""

    def __init__(
        self,
        settings: Settings,
        *,
        budget: LLMTokenBudget | None = None,
        cache: LLMCache | None = None,
        llm_call: Callable[[str], str] | None = None,
    ) -> None:
        self._settings = settings
        self._budget = budget or LLMTokenBudget(daily_budget=LLM_DAILY_TOKEN_BUDGET)
        self._cache = cache or LLMCache()
        self._llm_call = llm_call  # injectable for tests
        self._client: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        features: dict[str, Any],
        memory_context: dict[str, Any] | None = None,
    ) -> MakerParseResult | dict[str, Any]:
        """Produce a maker signal (cached + budget-guarded).

        Returns a `MakerParseResult` from the LLM path, or a plain rule-only
        signal dict when the budget is exhausted / LLM unavailable.
        """
        if not should_use_llm(self._budget):
            log.warning("maker_llm_budget_exhausted_fallback_to_rule")
            return rule_only_signal(features)

        try:
            result = cached_maker_output(
                self._cache,
                features,
                lambda: self._cached_producer(features, memory_context),
            )
        except Exception as exc:  # noqa: BLE001 - never let the maker crash the tick path
            log.error("maker_llm_error", extra={"error": str(exc)})
            return rule_only_signal(features)

        if isinstance(result, MakerParseResult) and result.parsed is None:
            log.warning("maker_llm_invalid_fallback_to_rule", extra={"error": result.error})
            return rule_only_signal(features)
        return result

    def budget(self) -> LLMTokenBudget:
        return self._budget

    def cache(self) -> LLMCache:
        return self._cache

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------
    def _cached_producer(
        self, features: dict[str, Any], memory_context: dict[str, Any] | None
    ) -> MakerParseResult:
        """Call the LLM and raise when the output is invalid so it is not cached."""
        result = self._call_llm(features, memory_context)
        if isinstance(result, MakerParseResult) and result.parsed is None:
            raise MakerOutputError(result.error)
        return result

    def _call_llm(
        self, features: dict[str, Any], memory_context: dict[str, Any] | None
    ) -> MakerParseResult:
        prompt = build_maker_prompt(features, memory_context)
        tokens: int = 0
        if self._llm_call is not None:
            raw = self._llm_call(prompt)
        else:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self._settings.llm_model,
                messages=[
                    {"role": "system", "content": MAKER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=enforce_temperature(self._settings.llm_temperature),
                max_tokens=2000,
            )
            raw = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            tokens = int(usage.total_tokens) if usage else 0
        if not tokens:
            tokens = max(len(raw) // 4, 1)
        if self._budget.consume(tokens):
            log.debug("maker_llm_tokens_consumed", extra={"tokens": tokens})
        return parse_maker_output_with_retry(lambda: raw, max_retries=LLM_MAX_RETRIES)

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._settings.gemini_api_key, base_url=GEMINI_BASE_URL)
        return self._client
