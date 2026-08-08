"""Unit tests for parquet data store + cost model (Phase 3)."""

from __future__ import annotations

import types

import pytest

from core.candle_engine import Candle
from core.cost_model import CostModel
from core.data_store import DataStore


def _store(tmp_path):
    return DataStore(types.SimpleNamespace(), base_dir=str(tmp_path))


def test_save_and_load_candles(tmp_path) -> None:
    store = _store(tmp_path)
    candles = [Candle(open=1, high=2, low=0, close=1, volume=10, ts_epoch=0.0)]
    path = store.save_candles("NIFTY", candles)
    assert path.exists()
    loaded = store.load_candles("NIFTY")
    assert len(loaded) == 1
    assert loaded[0].close == 1.0
    assert loaded[0].volume == 10.0


def test_load_missing_symbol_returns_empty(tmp_path) -> None:
    store = _store(tmp_path)
    assert store.load_candles("NIFTY") == []
    assert store.candle_count("NIFTY") == 0


def test_save_deduplicates_by_ts(tmp_path) -> None:
    store = _store(tmp_path)
    candles = [
        Candle(open=1, high=2, low=0, close=1, volume=10, ts_epoch=0.0),
        Candle(open=1, high=2, low=0, close=2, volume=20, ts_epoch=0.0),
    ]
    store.save_candles("NIFTY", candles)
    loaded = store.load_candles("NIFTY")
    assert len(loaded) == 1
    assert loaded[0].close == 2.0


def test_save_and_load_oi_series(tmp_path) -> None:
    store = _store(tmp_path)
    series = [
        (0.0, 100_000.0, 95_000.0),
        (60.0, 120_000.0, 90_000.0),
        (120.0, 130_000.0, 85_000.0),
    ]
    path = store.save_oi_series("NIFTY", series)
    assert path.exists()
    loaded = store.load_oi_series("NIFTY")
    assert loaded == series


def test_load_missing_oi_series_returns_empty(tmp_path) -> None:
    store = _store(tmp_path)
    assert store.load_oi_series("NIFTY") == []


class TestCostModel:
    def test_long_entry_fill_slippage_up(self) -> None:
        cm = CostModel(slippage_points=1.0, cost_per_trade_points=0.5)
        assert cm.fill_price(24000, "LONG") == 24001.0
        assert cm.fill_price(24000, "SHORT") == 23999.0

    def test_net_pnl_subtracts_cost(self) -> None:
        cm = CostModel(slippage_points=0.0, cost_per_trade_points=2.0)
        assert cm.net_pnl_points(100, 110, "LONG") == pytest.approx(8.0)
        assert cm.net_pnl_points(100, 90, "SHORT") == pytest.approx(8.0)
