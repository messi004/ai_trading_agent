"""Backtest CLI on real ingested data (Enhancement Phase 3).

Usage:
    python -m scripts.ingest_history --symbol NIFTY --start 2026-01-01 --end 2026-08-01 --with-oi
    python -m scripts.run_backtest --symbol NIFTY

Replays real minute candles + the real per-minute total OI series through the
same trigger/divergence/pattern functions used live, then simulates trades
(SL/Target/time exits). No synthetic data is used anywhere in this path.
"""

from __future__ import annotations

import argparse

from config.constants import SCALP_SL_MAX_POINTS, SCALP_TARGET_MIN_POINTS
from config.settings import Settings, get_settings
from core.data_store import DataStore
from core.logger import get_logger, setup_logging
from modules.backtest_engine import BacktestEngine
from modules.backtest_runner import _window_label, _window_start_ts
from modules.replay_engine import HistoricalOIProvider, ReplayEngine

log = get_logger(__name__)


def _print_report(label: str, report: dict) -> None:
    print(f"\n=== {label} ===")
    for key, value in report.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest on real data")
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sl-points", type=float, default=None)
    parser.add_argument("--target-points", type=float, default=None)
    parser.add_argument("--max-hold-bars", type=int, default=None)
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Also run walk-forward windows over the data",
    )
    parser.add_argument("--days", type=float, default=None, help="Trailing window: last N days")
    parser.add_argument("--months", type=float, default=None, help="Trailing window: last N months")
    parser.add_argument("--years", type=float, default=None, help="Trailing window: last N years")
    args = parser.parse_args()

    setup_logging("INFO", json_output=False)
    settings: Settings = get_settings()
    store = DataStore(settings, base_dir=args.data_dir)

    start_ts = _window_start_ts(days=args.days, months=args.months, years=args.years)
    candles = store.load_candles(args.symbol, start_ts=start_ts)
    if not candles:
        raise SystemExit(
            f"no candles for {args.symbol.upper()} in window "
            f"({_window_label(days=args.days, months=args.months, years=args.years)}) — "
            "run ingest_history first (with a live session token)"
        )
    try:
        oi = HistoricalOIProvider.from_store(store, args.symbol, start_ts=start_ts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    replay = ReplayEngine(
        settings,
        oi,
        sl_points=args.sl_points if args.sl_points else SCALP_SL_MAX_POINTS,
        target_points=args.target_points if args.target_points else SCALP_TARGET_MIN_POINTS,
    )
    engine = BacktestEngine(
        settings,
        replay,
        max_hold_bars=args.max_hold_bars or 15,
    )
    result = engine.run(candles)
    _print_report("BACKTEST REPORT (real candles + real OI)", engine.build_report(result))

    if args.walk_forward:
        windows = engine.walk_forward(candles)
        print("\n=== WALK-FORWARD WINDOWS ===")
        for window in windows:
            _print_report(f"Window {window['window']}", window)


if __name__ == "__main__":
    main()
