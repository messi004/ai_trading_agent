"""Unit tests for paper trader (Phase 3)."""

from __future__ import annotations

import types

from core.cost_model import CostModel
from modules.paper_trader import PaperTrader


def _trader() -> PaperTrader:
    return PaperTrader(
        types.SimpleNamespace(),
        cost_model=CostModel(slippage_points=1.0, cost_per_trade_points=0.5),
        max_hold_seconds=300,
    )


def test_submit_and_open_position() -> None:
    trader = _trader()
    sid = trader.submit_signal(
        {
            "signal_id": "s1",
            "direction": "LONG",
            "entry": 24000,
            "sl": 4,
            "target": 6,
            "ts_epoch": 0,
        }
    )
    assert sid == "s1"
    positions = trader.open_positions()
    assert len(positions) == 1
    assert positions[0].entry_fill == 24001.0  # slippage up for LONG


def test_target_hit_closes_short() -> None:
    trader = _trader()
    trader.submit_signal(
        {
            "signal_id": "s1",
            "direction": "SHORT",
            "entry": 24000,
            "sl": 4,
            "target": 6,
            "ts_epoch": 0,
        }
    )
    closed = trader.update_price(23993, ts_epoch=60)  # entry_fill 23999, target 23993
    assert len(closed) == 1
    assert closed[0].exit_reason == "TARGET_HIT"


def test_sl_hit_closes_long() -> None:
    trader = _trader()
    trader.submit_signal(
        {
            "signal_id": "s1",
            "direction": "LONG",
            "entry": 24000,
            "sl": 4,
            "target": 6,
            "ts_epoch": 0,
        }
    )
    closed = trader.update_price(23996, ts_epoch=60)  # entry_fill 24001, SL 23997
    assert len(closed) == 1
    assert closed[0].exit_reason == "SL_HIT"
    assert closed[0].pnl_points < 0


def test_time_exit() -> None:
    trader = _trader()
    trader.submit_signal(
        {
            "signal_id": "s1",
            "direction": "LONG",
            "entry": 24000,
            "sl": 4,
            "target": 6,
            "ts_epoch": 0,
        }
    )
    closed = trader.update_price(24001, ts_epoch=301)
    assert len(closed) == 1
    assert closed[0].exit_reason == "TIME_EXIT"


def test_report_aggregation() -> None:
    trader = _trader()
    trader.submit_signal(
        {"signal_id": "w", "direction": "LONG", "entry": 24000, "sl": 4, "target": 6, "ts_epoch": 0}
    )
    trader.update_price(24010, ts_epoch=60)
    trader.submit_signal(
        {"signal_id": "l", "direction": "LONG", "entry": 24000, "sl": 4, "target": 6, "ts_epoch": 0}
    )
    trader.update_price(23990, ts_epoch=120)
    report = trader.report()
    assert report["closed"] == 2
    assert report["target_hits"] == 1
    assert report["sl_hits"] == 1
    assert report["total_pnl_points"] < 0


def test_premium_tracked_and_used_for_pnl() -> None:
    trader = _trader()
    trader.update_premium(24000, "CALL", 95.0)
    trader.submit_signal(
        {
            "signal_id": "s1",
            "direction": "LONG",
            "entry": 24000,
            "sl": 4,
            "target": 6,
            "ts_epoch": 0,
            "strike": 24000,
            "option_type": "CALL",
            "entry_premium": 95.0,
        }
    )
    assert trader.latest_premium(24000, "CALL") == 95.0
    # premium moves up with the underlying -> target exit books premium profit
    trader.update_premium(24000, "CALL", 104.0)
    closed = trader.update_price(24010, ts_epoch=60)
    assert len(closed) == 1
    assert closed[0].exit_reason == "TARGET_HIT"
    assert closed[0].entry_premium == 95.0
    assert closed[0].exit_premium == 104.0
    assert closed[0].pnl_premium_points == 9.0


def test_premium_pnl_skipped_without_exit_tick() -> None:
    trader = _trader()
    trader.submit_signal(
        {
            "signal_id": "s1",
            "direction": "SHORT",
            "entry": 24000,
            "sl": 4,
            "target": 6,
            "ts_epoch": 0,
            "strike": 24000,
            "option_type": "PUT",
            "entry_premium": 60.0,
        }
    )
    # no exit premium tick arrives -> falls back to index points only
    closed = trader.update_price(23993, ts_epoch=60)
    assert len(closed) == 1
    assert closed[0].pnl_premium_points is None
    assert closed[0].pnl_points is not None


def test_update_premium_rejects_bad_contract() -> None:
    trader = _trader()
    trader.update_premium(24000, "GAMMA", 10.0)
    trader.update_premium(24000, "CALL", 0.0)
    assert trader.latest_premium(24000, "CALL") is None
