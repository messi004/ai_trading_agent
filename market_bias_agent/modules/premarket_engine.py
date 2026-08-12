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
    OIWallLevels,
    PreMarketLevelsError,
    combined_oi_by_strike,
    compute_pivot_sr,
    max_pain_zone,
    oi_wall_levels,
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

        pain = self._max_pain_summary(redis, close)
        if pain:
            levels["max_pain"] = pain
        walls = self._oi_walls_summary(redis, close)
        if walls:
            levels["oi_walls"] = walls

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

    def refresh_intraday(self) -> dict[str, Any]:
        """Recompute live OI-profile levels (walls + max pain) during the session.

        Pivot S/R stays as computed at premarket (structural, static). Only the
        OI-based fields are refreshed from the live OI buffers and merged into
        the existing Redis dict so the signal engine always reads fresh levels.
        """
        redis = self._redis or self._get_redis()
        levels = self._read_levels(redis)
        if not levels:
            return self.run()
        spot = self._last_spot(redis)
        if spot is None:
            return levels
        pain = self._max_pain_summary(redis, spot)
        walls = self._oi_walls_summary(redis, spot)
        levels["computed_at_ist"] = iso_ist(now_ist())
        if pain:
            levels["max_pain"] = pain
        else:
            levels.pop("max_pain", None)
        if walls:
            levels["oi_walls"] = walls
        else:
            levels.pop("oi_walls", None)
        redis.set_pre_market_levels(levels)
        self.last_result = levels
        log.info(
            "premarket_levels_refreshed",
            extra={
                "max_pain": pain.get("strike") if pain else None,
                "oi_resistance": walls.get("resistance") if walls else None,
                "oi_support": walls.get("support") if walls else None,
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
        walls = levels.get("oi_walls")
        if isinstance(walls, dict):
            resistance = walls.get("resistance") or []
            support = walls.get("support") or []
            if resistance:
                lines.append("OI Resistance: " + ", ".join(f"{lv:,.0f}" for lv in resistance))
            if support:
                lines.append("OI Support: " + ", ".join(f"{lv:,.0f}" for lv in support))
        lines.append(f"Updated: {levels.get('computed_at_ist', '')}")
        return "\n".join(lines)

    def _read_levels(self, redis: RedisManager) -> dict[str, Any]:
        try:
            stored = redis.get_pre_market_levels()
            return dict(stored) if isinstance(stored, dict) else {}
        except Exception as exc:  # noqa: BLE001
            log.warning("premarket_levels_read_failed", extra={"error": str(exc)})
            return {}

    def _last_spot(self, redis: RedisManager) -> float | None:
        ticks = redis.get_spot_ticks()
        for tick in reversed(ticks):
            price = tick.get("price")
            if price is not None:
                try:
                    return float(price)
                except (TypeError, ValueError):
                    continue
        return None

    def _oi_by_side(self, redis: RedisManager) -> tuple[dict[int, float], dict[int, float]]:
        """Latest per-strike Call/Put OI from the live intraday buffers."""
        strikes = redis.get_strikes()
        call_oi: dict[int, float] = {}
        put_oi: dict[int, float] = {}
        for strike in strikes:
            call_window = redis.get_call_oi_window(strike)
            put_window = redis.get_put_oi_window(strike)
            if call_window:
                call_oi[strike] = call_window[-1]
            if put_window:
                put_oi[strike] = put_window[-1]
        return call_oi, put_oi

    def _max_pain_summary(self, redis: RedisManager, spot: float) -> dict[str, Any] | None:
        """Max pain from per-strike Call/Put OI windows (best-effort)."""
        call_oi, put_oi = self._oi_by_side(redis)
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

    def _oi_walls_summary(self, redis: RedisManager, spot: float) -> dict[str, Any] | None:
        """Live OI-profile S/R walls (call walls = resistance, put walls = support)."""
        call_oi, put_oi = self._oi_by_side(redis)
        walls: OIWallLevels = oi_wall_levels(call_oi, put_oi, spot)
        if not walls.resistance and not walls.support and walls.max_pain is None:
            return None
        return {
            "resistance": [round(lv, 2) for lv in walls.resistance],
            "support": [round(lv, 2) for lv in walls.support],
            "max_pain": round(walls.max_pain, 2) if walls.max_pain is not None else None,
            "computed_at_ist": iso_ist(now_ist()),
        }

    def _get_redis(self) -> RedisManager:
        if self._redis is None:
            self._redis = RedisManager(self._settings).connect()
        return self._redis
