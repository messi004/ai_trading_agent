"""Backtest report builder for the Telegram /backtest command.

Runs the same real-data backtest path as ``scripts.run_backtest`` and renders
the report as plain text so it can be pushed to the ops channel. The heavy
work happens in a worker thread (via the Telegram listener) so the event loop
never blocks.
"""

from __future__ import annotations

from typing import Any

from config.constants import SCALP_SL_MAX_POINTS, SCALP_TARGET_MIN_POINTS
from config.settings import Settings, get_settings
from core.data_store import DataStore
from core.logger import get_logger
from modules.backtest_engine import BacktestEngine
from modules.replay_engine import HistoricalOIProvider, ReplayEngine

log = get_logger(__name__)

DEFAULT_MAX_HOLD_BARS = 15
WALK_FORWARD_WINDOW_BARS = 390


def run_backtest_report(
    *,
    symbol: str = "NIFTY",
    data_dir: str = "data",
    sl_points: float | None = None,
    target_points: float | None = None,
    max_hold_bars: int | None = None,
    walk_forward: bool = False,
    days: float | None = None,
    months: float | None = None,
    years: float | None = None,
    settings: Settings | None = None,
) -> str:
    """Run the backtest on real ingested data and return a text report.

    `days`/`months`/`years` filter the data to a trailing window ending now
    (e.g. ``/backtest --days 30``). Raises when no candles/OI are available
    so the listener can surface a clear "ingest first" message.
    """
    settings = settings or get_settings()
    store = DataStore(settings, base_dir=data_dir)

    start_ts = _window_start_ts(days=days, months=months, years=years)
    candles = store.load_candles(symbol, start_ts=start_ts)
    if not candles:
        raise ValueError(
            f"no candles for {symbol.upper()} in the requested window "
            f"({_window_label(days=days, months=months, years=years)}) — "
            "run `python -m scripts.ingest_history --symbol NIFTY --with-oi` first"
        )
    try:
        oi = HistoricalOIProvider.from_store(store, symbol, start_ts=start_ts)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    replay = ReplayEngine(
        settings,
        oi,
        sl_points=sl_points if sl_points else SCALP_SL_MAX_POINTS,
        target_points=target_points if target_points else SCALP_TARGET_MIN_POINTS,
    )
    engine = BacktestEngine(
        settings,
        replay,
        max_hold_bars=max_hold_bars or DEFAULT_MAX_HOLD_BARS,
    )

    result = engine.run(candles)
    lines = [
        "<b>📊 Backtest Report</b>",
        f"Data: {len(candles)} bars | {symbol} "
        f"({_window_label(days=days, months=months, years=years)})",
    ]
    lines.extend(_report_lines(engine.build_report(result)))

    if walk_forward:
        windows = engine.walk_forward(candles, window_bars=WALK_FORWARD_WINDOW_BARS)
        lines.append("")
        lines.append("<b>Walk-forward windows:</b>")
        for window in windows:
            lines.extend(_report_lines(window))
            lines.append("")

    log.info(
        "backtest_report_built",
        extra={"symbol": symbol, "bars": len(candles), "walk_forward": walk_forward},
    )
    return "\n".join(lines)


def _window_start_ts(
    *, days: float | None = None, months: float | None = None, years: float | None = None
) -> float | None:
    """Epoch seconds for the start of the trailing window (None = all data)."""
    if days is None and months is None and years is None:
        return None
    from datetime import timedelta

    from utils.time_utils import now_ist

    total_days = float(days or 0.0) + 30.0 * float(months or 0.0) + 365.0 * float(years or 0.0)
    return now_ist().timestamp() - timedelta(days=total_days).total_seconds()


def _window_label(
    *, days: float | None = None, months: float | None = None, years: float | None = None
) -> str:
    if days is None and months is None and years is None:
        return "all history"
    parts: list[str] = []
    for value, unit in ((years, "y"), (months, "mo"), (days, "d")):
        if value:
            parts.append(f"{value:g}{unit}")
    return "last " + " ".join(parts)


def _report_lines(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if "window" in report:
        lines.append(f"— window {report['window']} —")
    for key in ("signals", "trades", "rejected_by_checker"):
        if key in report:
            lines.append(f"{key.replace('_', ' ')}: {report[key]}")
    for key in (
        "hit_rate",
        "expectancy_points",
        "profit_factor",
        "max_drawdown_points",
        "max_consecutive_losses",
        "avg_win_points",
        "avg_loss_points",
    ):
        if key in report:
            lines.append(f"{key.replace('_', ' ')}: {report[key]}")
    exits = report.get("exit_reason_counts")
    if exits:
        parts = [f"{k}: {v}" for k, v in exits.items() if v]
        if parts:
            lines.append("exits: " + ", ".join(parts))
    return lines
