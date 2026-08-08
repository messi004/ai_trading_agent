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
from core.signal_store import SignalLogStore
from memory.memory_service import MemoryService
from memory.trap_records import TrapRecord, compute_subsequent_move
from utils.telegram_bot import TelegramBot

log = get_logger(__name__)


class EODEngine:
    def __init__(
        self,
        settings: Settings,
        memory: MemoryService | None = None,
        store: SignalLogStore | None = None,
        telegram: TelegramBot | None = None,
        participant_provider: Any | None = None,
    ) -> None:
        self._settings = settings
        self._memory = memory
        self._store = store or SignalLogStore()
        self._telegram = telegram
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

        indexed = self.index_day_traps()
        result["traps_indexed"] = len(indexed)

        if self._telegram is not None:
            self._telegram.send_ops(self._report_text(report, indexed))

        log.info("eod_run_done", extra={"traps_indexed": len(indexed)})
        return result

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

    @staticmethod
    def _report_text(report: dict[str, Any], indexed: list[str]) -> str:
        lines = ["<b>📉 EOD Institutional Report</b>"]
        if report.get("status") != "ok":
            lines.append("Participant OI: unavailable (skipped)")
        else:
            lines.append(f"Date: {report['date']} | Nifty {report['nifty50']:.0f}")
            lines.append(f"Structural bias: <b>{report['bias']}</b>")
            signals = report.get("signals") or []
            if signals:
                lines.append("Signals:")
                lines.extend(f"  • {s}" for s in signals)
        lines.append(f"Traps indexed to memory: {len(indexed)}")
        return "\n".join(lines)

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
