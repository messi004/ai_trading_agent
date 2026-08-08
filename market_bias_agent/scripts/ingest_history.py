"""Historical data ingestion CLI (Enhancement Phase 3).

Usage:
    python -m scripts.ingest_history --symbol NIFTY --start 2026-01-01 --end 2026-08-01
    python -m scripts.ingest_history --symbol NIFTY --with-oi \
        --start 2026-01-01 --end 2026-08-01

Fetches minute candles (and, with ``--with-oi``, per-strike option OI) from
the real Breeze REST history endpoint (through a live ICICI session) and
persists them into the parquet data store for the backtest harness. The
provider raises when no live session is configured so ingestion never
silently returns empty data. OI series are real sums across the ATM strike
window — never synthetic.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from config.constants import BREEZE_HISTORY_INTERVAL, STRIKES_RANGE_AROUND_ATM
from config.settings import Settings, get_settings
from core.breeze_providers import BreezeStrikesProvider
from core.breeze_session import BreezeSessionError, BreezeSessionManager
from core.candle_engine import Candle
from core.data_store import DataStore
from core.logger import get_logger, setup_logging

log = get_logger(__name__)


@runtime_checkable
class HistoryProvider(Protocol):
    def fetch_candles(self, symbol: str, start: str, end: str) -> list[Candle]:
        """Return Candle objects in the [start, end) range."""
        ...

    def fetch_oi_series(
        self, symbol: str, strikes: list[int], expiry: str, start: str, end: str
    ) -> list[tuple[float, float, float]]:
        """Return real (ts_epoch, total_call_oi, total_put_oi) rows."""
        ...


class BreezeHistoryProvider:
    """Real minute-history provider backed by the Breeze REST API."""

    def __init__(self, settings: Settings, session: BreezeSessionManager) -> None:
        self._settings = settings
        self._session = session

    def _client(self) -> Any:
        if not self._session.has_token and not self._session.has_credentials:
            raise BreezeSessionError(
                "No ICICI session token configured — push one via Telegram "
                "(`/session <token>`) before ingesting history"
            )
        return self._session.get_client()

    def fetch_candles(self, symbol: str, start: str, end: str) -> list[Candle]:
        """Fetch 1-minute candles from Breeze and map them to Candle objects."""
        client = self._client()
        raw = client.get_historical_data(
            interval=BREEZE_HISTORY_INTERVAL,
            from_date=start,
            to_date=end,
            stock_code=symbol.upper(),
            exchange_code="NSE",
        )
        rows = (raw or {}).get("Success") or []
        candles: list[Candle] = []
        for row in rows:
            ts = _parse_iso(row.get("datetime"))
            close = _to_float(row.get("close"))
            if ts is None or close <= 0:
                continue
            candles.append(
                Candle(
                    open=_to_float(row.get("open")),
                    high=_to_float(row.get("high")),
                    low=_to_float(row.get("low")),
                    close=close,
                    volume=_to_float(row.get("volume")),
                    ts_epoch=ts,
                )
            )
        candles.sort(key=lambda c: c.ts_epoch)
        log.info(
            "breeze_history_fetched",
            extra={"symbol": symbol, "start": start, "end": end, "count": len(candles)},
        )
        return candles

    def fetch_oi_series(
        self,
        symbol: str,
        strikes: list[int],
        expiry: str,
        start: str,
        end: str,
    ) -> list[tuple[float, float, float]]:
        """Fetch real per-strike OI history and aggregate to total call/put.

        Returns ``[(ts_epoch, total_call_oi, total_put_oi), ...]`` — the sum of
        per-strike ``open_interest`` across the given strikes for each minute,
        matching how the live engine totals OI. Raises when no live session is
        configured.
        """
        client = self._client()
        call_series: dict[float, float] = {}
        put_series: dict[float, float] = {}
        for strike in strikes:
            for right, bucket in (("call", call_series), ("put", put_series)):
                raw = client.get_historical_data(
                    interval=BREEZE_HISTORY_INTERVAL,
                    from_date=start,
                    to_date=end,
                    stock_code=symbol.upper(),
                    exchange_code="NFO",
                    product_type="options",
                    expiry_date=expiry,
                    option_type=right,
                    strike_price=str(strike),
                )
                rows = (raw or {}).get("Success") or []
                for row in rows:
                    ts = _parse_iso(row.get("datetime"))
                    oi = _to_float(row.get("open_interest"))
                    if ts is None or oi <= 0:
                        continue
                    bucket[ts] = bucket.get(ts, 0.0) + oi
        timestamps = sorted(set(call_series) | set(put_series))
        series = [
            (ts, call_series.get(ts, 0.0), put_series.get(ts, 0.0))
            for ts in timestamps
        ]
        log.info(
            "breeze_oi_fetched",
            extra={
                "symbol": symbol,
                "strikes": len(strikes),
                "expiry": expiry,
                "count": len(series),
            },
        )
        return series


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_iso(value: Any) -> float | None:
    if not value:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).replace("T", " ").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return None


def _atm_window(
    strikes: list[int], spot: float, range_around_atm: int = STRIKES_RANGE_AROUND_ATM
) -> list[int]:
    """Strikes within `range_around_atm` of the spot's ATM strike."""
    if spot <= 0:
        return strikes
    atm = int(round(spot / 100.0) * 100)
    lo, hi = atm - range_around_atm * 100, atm + range_around_atm * 100
    return [s for s in strikes if lo <= s <= hi]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest historical minute candles")
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--with-oi",
        action="store_true",
        help="Also fetch real per-strike OI history and persist the total call/put series",
    )
    args = parser.parse_args()

    setup_logging("INFO", json_output=False)
    settings: Settings = get_settings()
    session = BreezeSessionManager(settings)
    provider: HistoryProvider = BreezeHistoryProvider(settings, session)
    candles = provider.fetch_candles(args.symbol, args.start, args.end)

    store = DataStore(settings, base_dir=args.data_dir)
    if not candles:
        log.warning("no_candles_fetched", extra={"symbol": args.symbol})
        return
    store.save_candles(args.symbol, candles)
    log.info("ingest_done", extra={"symbol": args.symbol, "count": len(candles)})

    if args.with_oi:
        strikes_provider = BreezeStrikesProvider(settings, session)
        expiry, full_chain = strikes_provider.fetch_expiry_and_strikes(args.symbol)
        if not expiry or not full_chain:
            log.warning("no_strikes_for_oi", extra={"symbol": args.symbol})
            return
        spot = candles[-1].close if candles else 0.0
        window = _atm_window(full_chain, spot)
        series = provider.fetch_oi_series(args.symbol, window, expiry, args.start, args.end)
        if not series:
            log.warning("no_oi_fetched", extra={"symbol": args.symbol})
            return
        store.save_oi_series(args.symbol, series)
        log.info("oi_ingest_done", extra={"symbol": args.symbol, "count": len(series)})


if __name__ == "__main__":
    main()
