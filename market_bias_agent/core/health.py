"""Runtime health registry (Enhancement Phase 6).

Tracks live pipeline telemetry in one place and renders it for the /health
endpoint, a JSON status page, and the daily ops report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from config.constants import WATCHDOG_IDLE_SECONDS as _WATCHDOG_IDLE_SECONDS
from config.settings import Settings
from core.logger import get_logger
from utils.time_utils import iso_ist, market_status

log = get_logger(__name__)


@dataclass
class HealthRegistry:
    settings: Settings
    started_at: float = field(default_factory=time.time)
    last_tick_ts: float = 0.0
    ticks_processed: int = 0
    ws_connected: bool = False
    reconnect_count: int = 0
    buffer_fill_pct: float = 0.0
    triggers: int = 0
    approved: int = 0
    rejected: int = 0
    llm_tokens_spent: int = 0
    last_cron_success: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Telemetry updates
    # ------------------------------------------------------------------
    def record_tick(self) -> None:
        self.last_tick_ts = time.time()
        self.ticks_processed += 1

    def record_trigger(self) -> None:
        self.triggers += 1

    def record_verdict(self, approved: bool) -> None:
        if approved:
            self.approved += 1
        else:
            self.rejected += 1

    def record_llm_tokens(self, tokens: int) -> None:
        self.llm_tokens_spent += max(tokens, 0)

    def set_ws(self, connected: bool, reconnect_count: int = 0) -> None:
        self.ws_connected = connected
        self.reconnect_count = reconnect_count

    def set_buffer_fill(self, pct: float) -> None:
        self.buffer_fill_pct = max(0.0, min(pct, 100.0))

    def record_cron_success(self, name: str) -> None:
        self.last_cron_success[name] = iso_ist()

    def last_tick_age_seconds(self) -> float | None:
        if self.last_tick_ts <= 0:
            return None
        return time.time() - self.last_tick_ts

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def status(self, *, redis_connected: bool | None = None, profile: str = "") -> dict[str, Any]:
        return {
            "status": "ok",
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "market": market_status(),
            "redis_connected": redis_connected,
            "ws_connected": self.ws_connected,
            "reconnect_count": self.reconnect_count,
            "last_tick_age_seconds": self.last_tick_age_seconds(),
            "buffer_fill_pct": self.buffer_fill_pct,
            "ticks_processed": self.ticks_processed,
            "triggers": self.triggers,
            "approved": self.approved,
            "rejected": self.rejected,
            "llm_tokens_spent": self.llm_tokens_spent,
            "last_cron_success": self.last_cron_success,
            "profile": profile,
        }

    def daily_summary(self) -> dict[str, Any]:
        """Compact numbers used by the daily Telegram report."""
        return {
            "ticks": self.ticks_processed,
            "triggers": self.triggers,
            "approved": self.approved,
            "rejected": self.rejected,
            "rejection_rate": round(self.rejected / self.triggers, 3) if self.triggers else 0.0,
            "llm_tokens_spent": self.llm_tokens_spent,
            "last_cron_success": self.last_cron_success,
        }

    def watchdog_expired(
        self, idle_seconds: float = _WATCHDOG_IDLE_SECONDS, market: str | None = None
    ) -> bool:
        """True when no tick arrived for `idle_seconds` during market hours.

        `market` overrides the auto-detected PRE/OPEN/POST state (testability).
        """
        current_market = market if market is not None else market_status()
        if current_market != "OPEN":
            return False
        age = self.last_tick_age_seconds()
        if age is None:
            return self.ticks_processed > 0  # started but never got a tick
        return age > idle_seconds
