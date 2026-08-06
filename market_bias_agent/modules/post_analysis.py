"""Post-trade signal analysis loop (Enhancement Phase 8).

Every approved signal is tracked through the lifecycle, its outcome is
computed on exit (PnL, MFE/MAE, direction validity), stored in SQLite,
and written back to Qdrant memory with the *actual* outcome so future
similarity searches calibrate win-rates against what really happened.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from config.constants import EXIT_REASONS, OUTCOMES
from config.settings import Settings
from core.logger import get_logger
from core.signal_store import SignalLogStore
from core.signals import StructuredSignal
from memory.memory_service import MemoryService
from memory.trap_records import TrapRecord, compute_subsequent_move

log = get_logger(__name__)

BE_WINRATE_THRESHOLD = 0.40  # bias-correction loop: below this, tighten the rule

RESULT_WIN, RESULT_LOSS, RESULT_BE = OUTCOMES


def outcome_from_pnl(pnl_points: float) -> str:
    """Map PnL points to WIN / LOSS / BE."""
    if pnl_points > 0:
        return RESULT_WIN
    if pnl_points < 0:
        return RESULT_LOSS
    return RESULT_BE


@dataclass
class TradeSummary:
    """Concise per-trade result for the Telegram exit message."""

    signal_id: str
    exit_reason: str
    pnl_points: float
    mfe: float
    mae: float
    duration_minutes: float

    def to_text(self) -> str:
        reason = self.exit_reason.replace("_", " ")
        return (
            f"SIGNAL {self.signal_id[:8]} → <b>{reason}</b> | "
            f"{self.pnl_points:+.1f} pts | MFE {self.mfe:.1f} | MAE {self.mae:.1f} | "
            f"time {self.duration_minutes:.0f} min"
        )


class PostAnalysisEngine:
    def __init__(
        self,
        settings: Settings,
        store: SignalLogStore | None = None,
        memory: MemoryService | None = None,
    ) -> None:
        self._settings = settings
        self._store = store or SignalLogStore()
        self._memory = memory

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def register_signal(self, signal: StructuredSignal) -> str:
        return self._store.register_signal(signal)

    def approve(self, signal_id: str) -> None:
        self._store.mark_approved(signal_id)

    def monitor(self, signal_id: str, entry_fill: float) -> None:
        self._store.mark_monitoring(signal_id, entry_fill)

    def record_exit(
        self,
        signal_id: str,
        *,
        exit_price: float,
        exit_reason: str,
        pnl_points: float,
        mfe: float,
        mae: float,
        ts: float | None = None,
    ) -> TradeSummary:
        """Record the exit, compute the outcome, and return a Telegram summary."""
        if exit_reason not in EXIT_REASONS:
            raise ValueError(f"exit_reason must be one of {EXIT_REASONS}, got {exit_reason!r}")
        outcome = outcome_from_pnl(pnl_points)
        self._store.mark_exited(
            signal_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl_points=pnl_points,
            mfe=mfe,
            mae=mae,
            outcome=outcome,
            ts=ts or time.time(),
        )
        row = self._store.get_signal(signal_id)
        assert row is not None
        generated = float(row["generated_at"] or 0.0)
        duration_minutes = ((ts or time.time()) - generated) / 60.0
        summary = TradeSummary(
            signal_id=signal_id,
            exit_reason=exit_reason,
            pnl_points=pnl_points,
            mfe=mfe,
            mae=mae,
            duration_minutes=duration_minutes,
        )
        log.info(
            "post_trade_recorded",
            extra={"signal_id": signal_id, "outcome": outcome, "pnl_points": round(pnl_points, 2)},
        )
        return summary

    def close_and_write_back(
        self, signal_id: str, *, features: dict[str, Any] | None = None
    ) -> str | None:
        """Mark ANALYZED -> CLOSED and upsert the actual outcome to memory."""
        row = self._store.get_signal(signal_id)
        if row is None or row["status"] != "ANALYZED":
            log.warning("post_trade_writeback_skipped", extra={"signal_id": signal_id})
            return None
        if self._memory is None:
            self._memory = self._build_memory()
        record = self._trap_record_from_row(row, features)
        self._memory.index_records([record])
        self._store.mark_closed(signal_id)
        log.info(
            "post_trade_writeback",
            extra={"signal_id": signal_id, "actual_outcome": record.actual_outcome},
        )
        return record.vector_id

    # ------------------------------------------------------------------
    # Weekly performance feedback (10.5.5)
    # ------------------------------------------------------------------
    def weekly_report(self) -> dict[str, Any]:
        """Per-trigger-type hit-rate, expectancy, profit factor, MFE/MAE."""
        rows = self._store.get_closed() + self._store.get_analysed()
        by_trigger: dict[str, dict[str, Any]] = {}
        for row in rows:
            trigger = row["trigger_type"] or "UNKNOWN"
            bucket = by_trigger.setdefault(
                trigger,
                {"count": 0, "wins": 0, "losses": 0, "pnl": 0.0, "mfe": 0.0, "mae": 0.0},
            )
            bucket["count"] += 1
            pnl = float(row["pnl_points"] or 0.0)
            bucket["pnl"] += pnl
            bucket["mfe"] += float(row["mfe"] or 0.0)
            bucket["mae"] += float(row["mae"] or 0.0)
            if row["outcome"] == RESULT_WIN:
                bucket["wins"] += 1
            elif row["outcome"] == RESULT_LOSS:
                bucket["losses"] += 1
        report: dict[str, Any] = {}
        for trigger, bucket in by_trigger.items():
            decided = bucket["wins"] + bucket["losses"]
            report[trigger] = {
                "signals": bucket["count"],
                "hit_rate": round(bucket["wins"] / decided, 3) if decided else 0.0,
                "expectancy_points": round(bucket["pnl"] / bucket["count"], 2),
                "avg_mfe_points": round(bucket["mfe"] / bucket["count"], 2),
                "avg_mae_points": round(bucket["mae"] / bucket["count"], 2),
            }
        return report

    def bias_correction_suggestions(self) -> list[str]:
        """Suggest a Checker rule when a trap type's win-rate stays below threshold."""
        rows = self._store.get_closed() + self._store.get_analysed()
        by_trap: dict[str, dict[str, int]] = {}
        for row in rows:
            trap = row["trap_type"] or "NONE"
            bucket = by_trap.setdefault(trap, {"wins": 0, "decided": 0})
            bucket["decided"] += 1
            if row["outcome"] == RESULT_WIN:
                bucket["wins"] += 1
        suggestions: list[str] = []
        for trap, bucket in by_trap.items():
            if bucket["decided"] < 5:
                continue
            win_rate = bucket["wins"] / bucket["decided"]
            if trap != "NONE" and win_rate < BE_WINRATE_THRESHOLD:
                suggestions.append(
                    f"{trap} win-rate {win_rate:.0%} < {BE_WINRATE_THRESHOLD:.0%} "
                    f"({bucket['wins']}/{bucket['decided']}) — tighten/suppress {trap} signals"
                )
        return suggestions

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _build_memory(self) -> MemoryService:
        from memory.memory_service import build_memory_service

        return build_memory_service(self._settings, force_memory=True)

    def _trap_record_from_row(
        self, row: dict[str, Any], features: dict[str, Any] | None
    ) -> TrapRecord:
        from datetime import datetime, timezone

        entry_zone = json.loads(row["entry_zone"]) if row.get("entry_zone") else []
        entry = float(row["entry_fill"] or 0.0) or (sum(entry_zone) / 2 if entry_zone else 0.0)
        outcome = row["outcome"] or "BE"
        actual = "TARGET_HIT" if outcome == RESULT_WIN else "SL_HIT"
        trap = row["trap_type"] or "NONE"
        if trap == "BULL_TRAP" and outcome == RESULT_LOSS:
            actual = "BULL_TRAP_REJECTION"
        elif trap == "BEAR_TRAP" and outcome == RESULT_LOSS:
            actual = "BEAR_TRAP_REJECTION"
        generated = float(row["generated_at"] or 0.0)
        session_date = datetime.fromtimestamp(generated, tz=timezone.utc).date().isoformat()
        exit_ts = float(row["updated_at"] or generated)
        minutes = max(int((exit_ts - generated) / 60.0), 0)
        subsequent = compute_subsequent_move(entry, float(row["exit_price"] or 0.0), minutes)
        base_features = features or {
            "spot": entry,
            "pcr": 1.0,
            "call_oi_vel_1m": 0.0,
            "put_oi_vel_1m": 0.0,
            "velocity_5m": 0.0,
            "volatility": "ACTIVE",
        }
        base_features["trap_type"] = trap
        return TrapRecord(
            features=base_features,
            historical_outcome=actual,
            actual_outcome=actual,
            subsequent_move=subsequent,
            session_date=session_date,
            expiry_week=self._expiry_week(generated),
        )

    def _expiry_week(self, ts_epoch: float) -> int:
        from datetime import datetime, timezone

        day = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isocalendar()
        return (day[1] + 1) // 2  # 1..2 per month as a coarse expiry-week band
