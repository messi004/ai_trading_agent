"""EOD post-market engine (PRD Module 6).

Phase 5 wiring: indexes the day's trap events into Qdrant memory after the
close. Participant-wise OI analysis (FII/PRO vs retail) remains a stub until
the NSE data source is connected.
"""

from __future__ import annotations

from typing import Any

from config.settings import Settings
from core.logger import get_logger
from memory.memory_service import MemoryService

log = get_logger(__name__)


class EODEngine:
    def __init__(self, settings: Settings, memory: MemoryService | None = None) -> None:
        self._settings = settings
        self._memory = memory

    def run(self) -> None:
        """Full EOD: institutional footprint + memory ingestion + report."""
        log.info("eod_run_start")
        indexed = self.index_day_traps()
        log.info("eod_run_done", extra={"traps_indexed": len(indexed)})

    def index_day_traps(self, events: list[dict[str, Any]] | None = None) -> list[str]:
        """Index the day's trap events into Qdrant memory.

        `events` is normally collected from the Redis buffer; until the live
        collector is wired it may be supplied by tests or the replay path.
        """
        memory = self._memory or self._get_memory()
        if not events:
            log.warning("eod_no_trap_events")
            return []
        records = [_record_from_event(event) for event in events]
        return memory.index_records(records)

    def weekly_compact(self) -> int:
        """Weekly compaction: drop stale / low-quality vectors."""
        return self._get_memory().compact_stale()

    def _get_memory(self) -> MemoryService:
        if self._memory is None:
            from memory.memory_service import build_memory_service

            self._memory = build_memory_service(self._settings, force_memory=True)
        return self._memory


def _record_from_event(event: dict[str, Any]):
    from memory.trap_records import TrapRecord

    return TrapRecord(
        features=event["features"],
        historical_outcome=event["outcome"],
        subsequent_move=event["subsequent_move"],
        session_date=event["session_date"],
        expiry_week=event["expiry_week"],
    )
