"""EOD post-market engine (PRD Module 6).

Runs on the daily 18:00 IST cron and performs two real, data-backed tasks:

1. **Institutional footprint** — fetches the NSE participant-wise OI report
   (FII/DII/Pro/Client index futures + options) through the NiftyTrader API
   and dispatches a structured Telegram report with the next-day structural
   bias.
2. **Memory ingestion** — indexes the day's trap events (real signals from
   the SQLite signal store) into Qdrant for auto-learning.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from config.settings import Settings
from core.logger import get_logger
from core.participant_oi import (
    NiftyTraderParticipantOIProvider,
    ParticipantOIError,
    compute_structural_bias,
)
from core.redis_manager import RedisManager
from core.signal_store import SignalLogStore
from memory.memory_service import MemoryService
from memory.trap_records import TrapRecord, compute_subsequent_move
from utils.telegram_bot import TelegramBot
from utils.time_utils import now_ist

log = get_logger(__name__)


class EODEngine:
    def __init__(
        self,
        settings: Settings,
        memory: MemoryService | None = None,
        store: SignalLogStore | None = None,
        telegram: TelegramBot | None = None,
        redis: RedisManager | None = None,
        participant_provider: Any | None = None,
    ) -> None:
        self._settings = settings
        self._memory = memory
        self._store = store or SignalLogStore()
        self._telegram = telegram
        self._redis = redis
        self._participant = participant_provider or NiftyTraderParticipantOIProvider()

    def run(self) -> dict[str, Any]:
        """Full EOD: institutional footprint report + memory ingestion.

        Returns a summary dict. A participant-OI fetch failure does not abort
        trap indexing — it is logged and reported as unavailable.
        """
        log.info("eod_run_start")
        result: dict[str, Any] = {"participant_report": {}, "traps_indexed": 0}

        report = self._participant_report()
        result["participant_report"] = report
        if report.get("status") == "ok":
            self._persist_bias(report)

        indexed = self.index_day_traps()
        result["traps_indexed"] = len(indexed)

        if self._telegram is not None:
            self._telegram.send_ops(self._report_text(report, indexed))

        log.info("eod_run_done", extra={"traps_indexed": len(indexed)})
        return result

    def _persist_bias(self, report: dict[str, Any]) -> None:
        """Store the structural bias for the next-day signal engine."""
        try:
            redis = self._redis or self._get_redis()
            bias = {
                "bias": report.get("bias", "NEUTRAL"),
                "signals": report.get("signals") or [],
                "nifty50": report.get("nifty50", 0.0),
                "session_date": report.get("date", ""),
                "computed_at_ist": now_ist().isoformat(),
                "participants": report.get("participants") or {},
            }
            redis.set_eod_bias(bias)
            log.info("eod_bias_persisted", extra={"bias": bias["bias"]})
        except Exception as exc:  # noqa: BLE001 - bias persistence must not abort EOD
            log.warning("eod_bias_persist_failed", extra={"error": str(exc)})

    # ------------------------------------------------------------------
    # FII/PRO institutional footprint
    # ------------------------------------------------------------------
    def _participant_report(self) -> dict[str, Any]:
        try:
            positions = self._participant.fetch_latest()
        except ParticipantOIError as exc:
            log.warning("eod_participant_oi_unavailable", extra={"error": str(exc)})
            return {"status": "unavailable", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - never fail the whole EOD
            log.warning("eod_participant_oi_failed", extra={"error": str(exc)})
            return {"status": "unavailable", "error": str(exc)}

        analysis = compute_structural_bias(positions)
        latest = max(positions, key=lambda p: p.date) if positions else None
        report = {
            "status": "ok",
            "date": latest.date if latest else "",
            "nifty50": latest.nifty50 if latest else 0.0,
            "bias": analysis["bias"],
            "signals": analysis["signals"],
            "participants": analysis["participants"],
        }
        log.info(
            "eod_participant_report",
            extra={"bias": report["bias"], "signals": len(report["signals"])},
        )
        return report

    def _report_text(self, report: dict[str, Any], indexed: list[str]) -> str:
        """Detailed EOD report: positioning table + signals + day recap + traps."""
        lines = ["<b>📉 EOD Institutional Report</b>"]
        if report.get("status") != "ok":
            lines.append("Participant OI: unavailable (skipped)")
        else:
            date = report.get("date") or ""
            nifty = float(report.get("nifty50") or 0.0)
            lines.append(f"Date: {date} | Nifty {nifty:,.0f}")
            lines.append(f"Structural bias: <b>{report.get('bias')}</b>")

            participants = report.get("participants") or {}
            cohort_lines: list[str] = []
            for cohort in ("FII", "PRO", "CLIENT", "DII"):
                row = participants.get(cohort) or participants.get(cohort.lower())
                if not row:
                    continue
                fut_net = float(row.get("future_index_net") or 0.0)
                call_short = float(row.get("option_index_call_short") or 0.0)
                put_short = float(row.get("option_index_put_short") or 0.0)
                fut_txt = f"{fut_net:+,.0f}"
                cohort_lines.append(
                    f"  • {cohort:<7}: futures net {fut_txt} | "
                    f"calls written {call_short:,.0f} | puts written {put_short:,.0f}"
                )
            if cohort_lines:
                lines.append("")
                lines.append("📊 Positioning (contracts):")
                lines.extend(cohort_lines)

            signals = report.get("signals") or []
            if signals:
                lines.append("")
                lines.append("🎯 Signals:")
                lines.extend(f"  • {s}" for s in signals)

        recap = self._day_recap()
        if recap:
            lines.append("")
            lines.append("📋 Day recap:")
            lines.extend(f"  • {item}" for item in recap)

        if indexed:
            lines.append("")
            lines.append(f"🧠 Traps indexed to memory: {len(indexed)}")
        return "\n".join(lines)

    def _day_recap(self) -> list[str]:
        """Today's signal/outcome summary from the signal store (best-effort)."""
        try:
            rows = self._store.get_analysed() + self._store.get_closed()
        except Exception as exc:  # noqa: BLE001 - recap must never break the EOD report
            log.warning("eod_day_recap_failed", extra={"error": str(exc)})
            return []
        today = datetime.now(timezone.utc).date().isoformat()
        signals = wins = losses = 0
        pnl = 0.0
        for row in rows:
            generated = float(row.get("generated_at") or 0.0)
            session_date = datetime.fromtimestamp(generated, tz=timezone.utc).date().isoformat()
            if session_date != today:
                continue
            signals += 1
            outcome = row.get("outcome")
            if outcome == "WIN":
                wins += 1
            elif outcome == "LOSS":
                losses += 1
            pnl += float(row.get("pnl_points") or 0.0)
        if signals == 0:
            return []
        return [
            f"signals {signals} | win {wins} | loss {losses} | " f"paper PnL {pnl:+.1f} pts",
        ]

    # ------------------------------------------------------------------
    # Memory ingestion (live trap collector)
    # ------------------------------------------------------------------
    def index_day_traps(self, events: list[dict[str, Any]] | None = None) -> list[str]:
        """Index the day's real trap events into Qdrant memory.

        `events` is normally collected from the signal store (live). When
        passed explicitly (tests/replay) it is used as-is.
        """
        memory = self._memory or self._get_memory()
        if events is None:
            events = self._collect_day_events()
        if not events:
            log.warning("eod_no_trap_events")
            return []
        records = [_record_from_event(event) for event in events]
        return memory.index_records(records)

    def _collect_day_events(self) -> list[dict[str, Any]]:
        """Collect today's closed/analysed signals from the SQLite store.

        These are real approved signals that reached a terminal outcome this
        session — the live trap-event source for EOD memory ingestion.
        """
        rows = self._store.get_analysed() + self._store.get_closed()
        today = datetime.now(timezone.utc).date().isoformat()
        events: list[dict[str, Any]] = []
        for row in rows:
            generated = float(row["generated_at"] or 0.0)
            session_date = datetime.fromtimestamp(generated, tz=timezone.utc).date().isoformat()
            if session_date != today:
                continue
            events.append(_event_from_row(row))
        if events:
            log.info("eod_collected_traps", extra={"count": len(events)})
        return events

    def weekly_compact(self) -> int:
        """Weekly compaction: drop stale / low-quality vectors."""
        return self._get_memory().compact_stale()

    def _get_redis(self) -> RedisManager:
        if self._redis is None:
            self._redis = RedisManager(self._settings).connect()
        return self._redis

    def _get_memory(self) -> MemoryService:
        if self._memory is None:
            from memory.memory_service import build_memory_service

            self._memory = build_memory_service(self._settings, force_memory=True)
        return self._memory


def _record_from_event(event: dict[str, Any]) -> TrapRecord:
    return TrapRecord(
        features=event["features"],
        historical_outcome=event["outcome"],
        subsequent_move=event["subsequent_move"],
        session_date=event["session_date"],
        expiry_week=event["expiry_week"],
    )


def _event_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map a signal-log row (real outcome) to a trap event for memory."""
    generated = float(row["generated_at"] or 0.0)
    entry_zone = json.loads(row["entry_zone"]) if row.get("entry_zone") else []
    entry = float(row["entry_fill"] or 0.0) or (sum(entry_zone) / 2 if entry_zone else 0.0)
    outcome = row["outcome"] or "BE"
    actual = "TARGET_HIT" if outcome == "WIN" else "SL_HIT"
    trap = row["trap_type"] or "NONE"
    if trap == "BULL_TRAP" and outcome == "LOSS":
        actual = "BULL_TRAP_REJECTION"
    elif trap == "BEAR_TRAP" and outcome == "LOSS":
        actual = "BEAR_TRAP_REJECTION"
    exit_ts = float(row["updated_at"] or generated)
    minutes = max(int((exit_ts - generated) / 60.0), 0)
    subsequent = compute_subsequent_move(entry, float(row["exit_price"] or 0.0), minutes)
    day = datetime.fromtimestamp(generated, tz=timezone.utc).isocalendar()
    expiry_week = (day[1] + 1) // 2
    session_date = datetime.fromtimestamp(generated, tz=timezone.utc).date().isoformat()
    features = {
        "spot": entry,
        "pcr": 1.0,
        "call_oi_vel_1m": 0.0,
        "put_oi_vel_1m": 0.0,
        "velocity_5m": 0.0,
        "volatility": "ACTIVE",
    }
    features["trap_type"] = trap
    return {
        "features": features,
        "outcome": actual,
        "subsequent_move": subsequent,
        "session_date": session_date,
        "expiry_week": expiry_week,
    }
