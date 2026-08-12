"""Pre-market engine (PRD Module 6).

Runs on a daily 08:30 IST cron and pre-computes the next trading session's
key levels from the previous session's data:

  * Classic pivot-point Support/Resistance fan (P, R1, R2, S1, S2)
  * Psychological round levels around the pivot
  * Max-Pain pinning zone from the option chain's highest combined OI

The levels are persisted to Redis where the live intraday engine reads them
so the |spot - level| <= tolerance rule has warm S/R zones at market open.
"""

from __future__ import annotations

from typing import Any

from config.constants import KEY_PRE_MARKET_LEVELS
from config.settings import Settings
from core.logger import get_logger
from core.premarket_levels import (
    PreMarketLevelsError,
    combined_oi_by_strike,
    compute_pivot_sr,
    max_pain_zone,
    psychological_levels_around,
    session_bounds_from_ticks,
)
from core.redis_manager import RedisManager
from utils.time_utils import iso_ist, now_ist

log = get_logger(__name__)


class PreMarketEngine:
    def __init__(self, settings: Settings, redis: RedisManager | None = None) -> None:
        self._settings = settings
        self._redis = redis
        self.last_result: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        """Compute next-day S/R + max pain zones and persist to Redis.

        Returns a summary dict (empty when no prior-session data is available).
        """
        redis = self._redis or self._get_redis()
        ticks = redis.get_spot_ticks()
        bounds = session_bounds_from_ticks(ticks)
        if bounds is None:
            log.warning("premarket_no_prev_session_data")
            self.last_result = {}
            return {}

        high, low, close = bounds
        levels: dict[str, Any] = {
            "computed_at_ist": iso_ist(now_ist()),
            "session_date": now_ist().date().isoformat(),
            "prev_high": high,
            "prev_low": low,
            "prev_close": close,
        }
        try:
            sr = compute_pivot_sr(high, low, close)
        except PreMarketLevelsError as exc:
            log.error("premarket_pivot_error", extra={"error": str(exc)})
            self.last_result = {}
            return {}
        levels.update(sr.to_dict())
        levels["levels"] = sr.ordered_levels()

        psych = psychological_levels_around(close)
        levels["psych_levels"] = psych

        pain = self._max_pain_summary(redis)
        if pain:
            levels["max_pain"] = pain

        redis.set_pre_market_levels(levels)
        self.last_result = levels
        log.info(
            "premarket_levels_ready",
            extra={
                "key": KEY_PRE_MARKET_LEVELS,
                "pivot": levels["pivot"],
                "r1": levels["r1"],
                "s1": levels["s1"],
                "max_pain": pain.get("strike") if pain else None,
            },
        )
        return levels

    def report_text(self) -> str:
        """Structured premarket levels summary for the Telegram ops channel."""
        levels = self.last_result or {}
        if not levels:
            return "<b>🌅 Premarket</b>\nNo prior-session data available — levels skipped."
        lines = [
            "<b>🌅 Premarket Levels</b>",
            f"Session: {levels.get('session_date', '')} | "
            f"Prev H/L/C: {levels.get('prev_high', 0.0):,.1f} / "
            f"{levels.get('prev_low', 0.0):,.1f} / {levels.get('prev_close', 0.0):,.1f}",
            "",
            f"Pivot: <b>{levels.get('pivot', 0.0):,.1f}</b>",
            f"R1 {levels.get('r1', 0.0):,.1f} | R2 {levels.get('r2', 0.0):,.1f}",
            f"S1 {levels.get('s1', 0.0):,.1f} | S2 {levels.get('s2', 0.0):,.1f}",
        ]
        pain = levels.get("max_pain")
        if isinstance(pain, dict):
            zone = pain.get("zone")
            if isinstance(zone, list | tuple) and zone:
                zone_txt = f" ({zone[0]:,.1f}–{zone[1]:,.1f})"
            else:
                zone_txt = ""
            lines.append(f"Max Pain: {pain.get('strike', 0.0):,.0f}{zone_txt}")
        return "\n".join(lines)

    def _max_pain_summary(self, redis: RedisManager) -> dict[str, Any] | None:
        """Max pain from per-strike Call/Put OI windows (best-effort)."""
        strikes = redis.get_strikes()
        if not strikes:
            return None
        call_oi: dict[int, float] = {}
        put_oi: dict[int, float] = {}
        for strike in strikes:
            call_window = redis.get_call_oi_window(strike)
            put_window = redis.get_put_oi_window(strike)
            if call_window:
                call_oi[strike] = call_window[-1]
            if put_window:
                put_oi[strike] = put_window[-1]
        if not call_oi and not put_oi:
            return None
        combined = combined_oi_by_strike(call_oi, put_oi)
        zone = max_pain_zone(combined)
        if zone is None:
            return None
        return {
            "strike": int(max(combined, key=lambda s: combined[s])),
            "zone": [round(zone[0], 2), round(zone[1], 2)],
        }

    def _get_redis(self) -> RedisManager:
        if self._redis is None:
            self._redis = RedisManager(self._settings).connect()
        return self._redis
