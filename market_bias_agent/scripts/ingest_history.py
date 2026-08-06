"""Historical data ingestion CLI (Enhancement Phase 3).

Usage:
    python -m scripts.ingest_history --symbol NIFTY --start 2026-01-01 --end 2026-08-01

Fetches minute candles from a HistoryProvider and persists them into the
parquet data store for the backtest harness.
"""

from __future__ import annotations

import argparse
from typing import Protocol, runtime_checkable

from config.settings import Settings, get_settings
from core.data_store import DataStore
from core.logger import get_logger, setup_logging

log = get_logger(__name__)


@runtime_checkable
class HistoryProvider(Protocol):
    def fetch_candles(self, symbol: str, start: str, end: str) -> list:
        """Return Candle objects in the [start, end) range."""
        ...


class StubHistoryProvider:
    """Placeholder until the Breeze REST minute-history integration lands."""

    def fetch_candles(self, symbol: str, start: str, end: str) -> list:
        log.warning("stub_history_fetch", extra={"symbol": symbol, "start": start, "end": end})
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest historical minute candles")
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    setup_logging("INFO", json_output=False)
    settings: Settings = get_settings()

    provider: HistoryProvider = StubHistoryProvider()
    candles = provider.fetch_candles(args.symbol, args.start, args.end)

    store = DataStore(settings, base_dir=args.data_dir)
    if not candles:
        log.warning("no_candles_fetched", extra={"symbol": args.symbol})
        return
    store.save_candles(args.symbol, candles)
    log.info("ingest_done", extra={"symbol": args.symbol, "count": len(candles)})


if __name__ == "__main__":
    main()
