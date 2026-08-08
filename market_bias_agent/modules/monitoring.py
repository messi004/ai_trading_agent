"""Observability & alerting (Enhancement Phase 6).

* Watchdog: alert ops via Telegram when no tick arrives for 5 min in market hours.
* Daily report: structured summary of triggers / approvals / rejections / LLM cost.
* Decision audit: Checker verdicts are written to Redis for post-hoc review.
"""

from __future__ import annotations

from typing import Any

from config.constants import WATCHDOG_IDLE_SECONDS
from config.settings import Settings
from core.health import HealthRegistry
from core.logger import get_logger
from modules.checker_node import CheckerNode
from utils.telegram_bot import TelegramBot

log = get_logger(__name__)


class MonitoringService:
    def __init__(
        self,
        settings: Settings,
        health: HealthRegistry,
        telegram: TelegramBot | None = None,
        checker: CheckerNode | None = None,
        audit_writer: Any | None = None,
    ) -> None:
        self._settings = settings
        self._health = health
        self._telegram = telegram or TelegramBot(settings)
        self._checker = checker
        if checker is not None and audit_writer is not None:
            checker.set_audit_writer(audit_writer)

    # ------------------------------------------------------------------
    # 8.1 Watchdog
    # ------------------------------------------------------------------
    def watchdog_check(self, idle_seconds: float = WATCHDOG_IDLE_SECONDS) -> bool:
        """Alert ops when ticks stall during market hours. Returns True if alerted."""
        if not self._health.watchdog_expired(idle_seconds):
            return False
        age = self._health.last_tick_age_seconds()
        age_text = f"{age:.0f}s" if age is not None else "unknown"
        text = (
            "⚠️ <b>OPS ALERT</b>\n"
            f"Market is OPEN but no ticks for {age_text} (> {idle_seconds:.0f}s).\n"
            f"WS connected: {self._health.ws_connected} | "
            f"reconnects: {self._health.reconnect_count}\n"
            f"Ticks today: {self._health.ticks_processed}"
        )
        log.warning(
            "ops_watchdog_alert",
            extra={"idle_seconds": round(age or 0, 1), "ws_connected": self._health.ws_connected},
        )
        return self._telegram.send_ops(text)

    # ------------------------------------------------------------------
    # 8.2 Daily report
    # ------------------------------------------------------------------
    def build_daily_report(self) -> str:
        summary = self._health.daily_summary()
        lines = [
            "<b>📊 Daily Ops Report</b>",
            f"Ticks processed: <b>{summary['ticks']:,}</b>",
            f"Triggers: <b>{summary['triggers']}</b> "
            f"(approved {summary['approved']} / rejected {summary['rejected']})",
            f"Rejection rate: {summary['rejection_rate']:.1%}",
            f"LLM tokens spent: {summary['llm_tokens_spent']:,}",
        ]
        if self._checker is not None:
            by_rule = self._rejections_by_rule()
            if by_rule:
                lines.append("Rejections by rule:")
                lines.extend(f"  {rule}: {count}" for rule, count in by_rule)
        if summary["last_cron_success"]:
            cron = ", ".join(f"{k}@{v}" for k, v in summary["last_cron_success"].items())
            lines.append(f"Last cron success: {cron}")
        return "\n".join(lines)

    def send_daily_report(self) -> bool:
        report = self.build_daily_report()
        log.info("ops_daily_report", extra={"report": report})
        return self._telegram.send_ops(report)

    def send_text(self, text: str) -> bool:
        log.info("ops_send_text", extra={"report": text})
        return self._telegram.send_ops(text)

    def _rejections_by_rule(self) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for verdict in self._checker.audit_trail(limit=1000) if self._checker else []:
            for rule in verdict.get("rejected_rules", []):
                counts[rule] = counts.get(rule, 0) + 1
        return sorted(counts.items(), key=lambda item: item[1], reverse=True)


def status_page(
    health: HealthRegistry,
    *,
    redis_connected: bool | None = None,
    profile: str = "",
    audit_recent: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """JSON status page combining health + recent audit decisions."""
    page = health.status(redis_connected=redis_connected, profile=profile)
    page["audit_recent"] = audit_recent or []
    return page
