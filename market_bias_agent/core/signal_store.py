"""Signal log store (Enhancement Phase 8, section 10.5.3).

SQLite-backed persistence for the post-trade analysis loop. Every signal
walks the lifecycle:
    SIGNAL_GENERATED → APPROVED → MONITORING → EXITED → ANALYZED → CLOSED
and the store enforces legal transitions, so analysis is always complete
before a signal is written back to memory.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from config.constants import SIGNAL_LIFECYCLE
from core.logger import get_logger

log = get_logger(__name__)

_CREATE = """
CREATE TABLE IF NOT EXISTS signal_log (
    signal_id       TEXT PRIMARY KEY,
    generated_at    REAL,
    trigger_type    TEXT,
    direction       TEXT,
    entry_zone      TEXT,
    sl              REAL,
    target          REAL,
    entry_fill      REAL,
    exit_price      REAL,
    exit_reason     TEXT,
    mfe             REAL,
    mae             REAL,
    pnl_points      REAL,
    outcome         TEXT,
    confidence      REAL,
    trap_type       TEXT,
    maker_rationale TEXT,
    status          TEXT,
    updated_at      REAL
);
"""

_VALID_NEXT = {
    "SIGNAL_GENERATED": ("APPROVED",),
    "APPROVED": ("MONITORING",),
    "MONITORING": ("EXITED",),
    "EXITED": ("ANALYZED",),
    "ANALYZED": ("CLOSED",),
    "CLOSED": (),
}


class SignalTransitionError(ValueError):
    """Raised when a signal tries an illegal lifecycle transition."""


class SignalLogStore:
    def __init__(self, db_path: str | Path = "data/signal_log.db") -> None:
        self._path = str(db_path)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_CREATE)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle transitions (enforced)
    # ------------------------------------------------------------------
    def register_signal(self, signal: Any) -> str:
        """Insert a new signal row in SIGNAL_GENERATED state."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO signal_log (
                    signal_id, generated_at, trigger_type, direction, entry_zone,
                    sl, target, confidence, trap_type, maker_rationale, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.signal_id,
                    signal.ts_epoch,
                    signal.trigger_type,
                    signal.direction,
                    json.dumps(list(signal.entry_zone)),
                    signal.sl,
                    signal.target,
                    signal.confidence,
                    signal.trap_type,
                    signal.rationale,
                    "SIGNAL_GENERATED",
                    signal.ts_epoch,
                ),
            )
        return signal.signal_id

    def transition(self, signal_id: str, new_status: str) -> None:
        """Move a signal to `new_status`, rejecting illegal transitions."""
        if new_status not in SIGNAL_LIFECYCLE:
            raise SignalTransitionError(f"unknown status {new_status!r}")
        row = self.get_signal(signal_id)
        if row is None:
            raise SignalTransitionError(f"unknown signal_id {signal_id!r}")
        if new_status not in _VALID_NEXT[row["status"]]:
            raise SignalTransitionError(
                f"illegal transition {row['status']} -> {new_status} for {signal_id}"
            )
        self._update_fields(signal_id, status=new_status)

    def mark_approved(self, signal_id: str) -> None:
        self.transition(signal_id, "APPROVED")

    def mark_monitoring(self, signal_id: str, entry_fill: float) -> None:
        self.transition(signal_id, "MONITORING")
        self._update_fields(signal_id, entry_fill=entry_fill)

    def mark_exited(
        self,
        signal_id: str,
        *,
        exit_price: float,
        exit_reason: str,
        pnl_points: float,
        mfe: float,
        mae: float,
        outcome: str | None = None,
        ts: float | None = None,
    ) -> None:
        """Record the exit + computed outcome, moving EXITED -> ANALYZED."""
        if outcome is None:
            outcome = self._outcome_from_pnl(pnl_points)
        self.transition(signal_id, "EXITED")
        self.transition(signal_id, "ANALYZED")
        self._update_fields(
            signal_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl_points=pnl_points,
            mfe=mfe,
            mae=mae,
            outcome=outcome,
            updated_at=ts,
        )

    def mark_closed(self, signal_id: str) -> None:
        self.transition(signal_id, "CLOSED")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_signal(self, signal_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM signal_log WHERE signal_id = ?", (signal_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_by_status(self, status: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM signal_log WHERE status = ? ORDER BY generated_at", (status,)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_analysed(self) -> list[dict[str, Any]]:
        return self.get_by_status("ANALYZED")

    def get_closed(self) -> list[dict[str, Any]]:
        return self.get_by_status("CLOSED")

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM signal_log").fetchone()
        return int(row[0])

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _outcome_from_pnl(self, pnl_points: float) -> str:
        if pnl_points > 0:
            return "WIN"
        if pnl_points < 0:
            return "LOSS"
        return "BE"

    def _update_fields(self, signal_id: str, **fields: Any) -> None:
        if not fields:
            return
        pairs = ", ".join(f"{key} = ?" for key in fields)
        values = [fields[key] for key in fields]
        with self._conn:
            self._conn.execute(
                f"UPDATE signal_log SET {pairs} WHERE signal_id = ?",
                (*values, signal_id),
            )

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        keys = [
            description[0]
            for description in self._conn.execute("SELECT * FROM signal_log").description
        ]
        return {key: value for key, value in zip(keys, row, strict=False)}
