"""Local historical data store (Enhancement Phase 3).

Stores minute candles and per-minute total option OI series as parquet
files under a data directory, keyed by symbol. Used by the backtest +
walk-forward harness and the ingestion script. Both files are written
from real Breeze REST history (never synthetic).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import Settings
from core.candle_engine import Candle
from core.logger import get_logger

log = get_logger(__name__)

_COLUMNS = ["ts_epoch", "open", "high", "low", "close", "volume"]
_OI_COLUMNS = ["ts_epoch", "total_call_oi", "total_put_oi"]


class DataStore:
    def __init__(self, settings: Settings, base_dir: str = "data") -> None:
        self._settings = settings
        self._base = Path(base_dir)

    def _path_for(self, symbol: str) -> Path:
        path = self._base / f"{symbol.upper()}_1m.parquet"
        return path

    def save_candles(self, symbol: str, candles: list[Candle]) -> Path:
        """Write candles to parquet (append-friendly: replaces the file)."""
        path = self._path_for(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not candles:
            pd.DataFrame(columns=_COLUMNS).to_parquet(path)
            return path
        frame = pd.DataFrame(
            [
                {
                    "ts_epoch": c.ts_epoch,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles
            ]
        )
        frame = frame.sort_values("ts_epoch").drop_duplicates("ts_epoch", keep="last")
        frame.to_parquet(path, index=False)
        log.info("candles_saved", extra={"symbol": symbol, "count": len(frame), "path": str(path)})
        return path

    def load_candles(self, symbol: str, start_ts: float | None = None) -> list[Candle]:
        """Load candles for a symbol (empty if file missing).

        `start_ts` filters to candles at or after that epoch — used by the
        backtest range windows (days/months/years).
        """
        path = self._path_for(symbol)
        if not path.exists():
            return []
        frame = pd.read_parquet(path)
        frame = frame.sort_values("ts_epoch")
        if start_ts is not None:
            frame = frame[frame["ts_epoch"] >= start_ts]
        candles = [
            Candle(
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                ts_epoch=float(row[0]),
            )
            for row in frame.itertuples(index=False, name=None)
        ]
        return candles

    def candle_count(self, symbol: str) -> int:
        return len(self.load_candles(symbol))

    def _oi_path_for(self, symbol: str) -> Path:
        return self._base / f"{symbol.upper()}_oi_1m.parquet"

    def save_oi_series(self, symbol: str, series: list[tuple[float, float, float]]) -> Path:
        """Persist (ts_epoch, total_call_oi, total_put_oi) rows to parquet."""
        path = self._oi_path_for(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not series:
            pd.DataFrame(columns=_OI_COLUMNS).to_parquet(path)
            return path
        frame = pd.DataFrame(
            [
                {"ts_epoch": ts, "total_call_oi": call, "total_put_oi": put}
                for ts, call, put in series
            ]
        )
        frame = frame.sort_values("ts_epoch").drop_duplicates("ts_epoch", keep="last")
        frame.to_parquet(path, index=False)
        log.info(
            "oi_series_saved",
            extra={"symbol": symbol, "count": len(frame), "path": str(path)},
        )
        return path

    def load_oi_series(
        self, symbol: str, start_ts: float | None = None
    ) -> list[tuple[float, float, float]]:
        """Load (ts_epoch, total_call_oi, total_put_oi) rows (empty if missing).

        `start_ts` filters to rows at or after that epoch so the backtest OI
        series stays aligned with the candle range.
        """
        path = self._oi_path_for(symbol)
        if not path.exists():
            return []
        frame = pd.read_parquet(path)
        frame = frame.sort_values("ts_epoch")
        if start_ts is not None:
            frame = frame[frame["ts_epoch"] >= start_ts]
        return [
            (float(row[0]), float(row[1]), float(row[2]))
            for row in frame.itertuples(index=False, name=None)
        ]
