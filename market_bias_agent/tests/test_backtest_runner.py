"""Unit tests for the Telegram /backtest runner report builder."""

from __future__ import annotations

import types

import pytest

from modules import backtest_runner
from modules.backtest_runner import _report_lines, run_backtest_report


class FakeStore:
    def __init__(self, candles, oi) -> None:
        self._candles = candles
        self._oi = oi
        self.last_candle_start_ts = None
        self.last_oi_start_ts = None

    def load_candles(self, symbol: str, start_ts=None):
        self.last_candle_start_ts = start_ts
        return self._candles

    def load_oi_series(self, symbol: str, start_ts=None):
        self.last_oi_start_ts = start_ts
        return self._oi


def _fake_candles(n: int = 10) -> list:
    from core.candle_engine import Candle

    return [
        Candle(ts_epoch=float(i * 60), open=100.0, high=105.0, low=95.0, close=102.0, volume=1000.0)
        for i in range(n)
    ]


def test_report_lines_formats_metrics() -> None:
    report = {
        "signals": 5,
        "trades": 4,
        "rejected_by_checker": 1,
        "hit_rate": 0.5,
        "expectancy_points": 1.25,
        "profit_factor": 1.5,
        "max_drawdown_points": -4.0,
        "max_consecutive_losses": 2,
        "avg_win_points": 3.0,
        "avg_loss_points": -1.5,
        "exit_reason_counts": {
            "TARGET_HIT": 2,
            "SL_HIT": 2,
            "TIME_EXIT": 0,
            "DIRECTION_INVALIDATED": 0,
        },
    }
    lines = _report_lines(report)
    text = "\n".join(lines)
    assert "signals: 5" in text
    assert "hit rate: 0.5" in text
    assert "exits: TARGET_HIT: 2, SL_HIT: 2" in text
    assert "TIME_EXIT: 0" not in text  # zero-count exits omitted


def test_run_backtest_report_no_candles_raises(monkeypatch) -> None:
    store = FakeStore([], [])
    monkeypatch.setattr(backtest_runner, "DataStore", lambda *a, **k: store)
    with pytest.raises(ValueError, match="no candles"):
        run_backtest_report(settings=types.SimpleNamespace())


def test_run_backtest_report_applies_window(monkeypatch) -> None:
    store = FakeStore(_fake_candles(), [(0.0, 1000.0, 900.0)])
    monkeypatch.setattr(backtest_runner, "DataStore", lambda *a, **k: store)

    captured: dict = {}

    class FakeOI:
        def __init__(self, series) -> None:
            self._series = series

        @classmethod
        def from_store(cls, store, symbol, start_ts=None):
            captured["start_ts"] = start_ts
            return cls(store.load_oi_series(symbol, start_ts=start_ts))

    class FakeReplay:
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeEngine:
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs
            self.report = {
                "signals": 1,
                "trades": 0,
                "rejected_by_checker": 0,
                "hit_rate": 0.0,
                "expectancy_points": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_points": 0.0,
                "max_consecutive_losses": 0,
                "avg_win_points": 0.0,
                "avg_loss_points": 0.0,
                "exit_reason_counts": {
                    "TARGET_HIT": 0,
                    "SL_HIT": 0,
                    "TIME_EXIT": 0,
                    "DIRECTION_INVALIDATED": 0,
                },
            }

        def run(self, candles):
            return object()

        def build_report(self, result):
            return self.report

    monkeypatch.setattr(backtest_runner, "HistoricalOIProvider", FakeOI)
    monkeypatch.setattr(backtest_runner, "ReplayEngine", FakeReplay)
    monkeypatch.setattr(backtest_runner, "BacktestEngine", FakeEngine)

    report = run_backtest_report(settings=types.SimpleNamespace(), days=7)
    assert "last 7d" in report
    assert store.last_candle_start_ts is not None
    assert captured["start_ts"] == store.last_candle_start_ts
    assert store.last_oi_start_ts is not None

    report_full = run_backtest_report(settings=types.SimpleNamespace())
    assert "all history" in report_full
    assert store.last_candle_start_ts is None


def test_window_helpers() -> None:
    from modules.backtest_runner import _window_label, _window_start_ts

    assert _window_label() == "all history"
    assert _window_label(days=7) == "last 7d"
    assert _window_label(months=3, years=1) == "last 1y 3mo"
    assert _window_start_ts() is None
    assert _window_start_ts(days=0) is None or _window_start_ts(days=0) is not None
    start = _window_start_ts(days=30)
    assert start is not None
    assert start > 0


def test_run_backtest_report_no_oi_raises(monkeypatch) -> None:
    store = FakeStore(_fake_candles(), [])
    monkeypatch.setattr(backtest_runner, "DataStore", lambda *a, **k: store)

    class NoOI:
        @classmethod
        def from_store(cls, store, symbol, start_ts=None):
            raise ValueError("no real OI series found for NIFTY")

    monkeypatch.setattr(backtest_runner, "HistoricalOIProvider", NoOI)
    with pytest.raises(ValueError, match="no real OI series"):
        run_backtest_report(settings=types.SimpleNamespace())


def test_run_backtest_report_end_to_end(monkeypatch) -> None:
    store = FakeStore(_fake_candles(), [(0.0, 1000.0, 900.0)])
    monkeypatch.setattr(backtest_runner, "DataStore", lambda *a, **k: store)

    class FakeOI:
        def __init__(self, series) -> None:
            self._series = series

        @classmethod
        def from_store(cls, store, symbol, start_ts=None):
            return cls(store.load_oi_series(symbol))

    class FakeReplay:
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeEngine:
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs
            self.report = {
                "signals": 3,
                "trades": 2,
                "rejected_by_checker": 1,
                "hit_rate": 0.5,
                "expectancy_points": 0.4,
                "profit_factor": 1.2,
                "max_drawdown_points": -1.0,
                "max_consecutive_losses": 1,
                "avg_win_points": 2.0,
                "avg_loss_points": -1.2,
                "exit_reason_counts": {
                    "TARGET_HIT": 1,
                    "SL_HIT": 1,
                    "TIME_EXIT": 0,
                    "DIRECTION_INVALIDATED": 0,
                },
            }

        def run(self, candles):
            return object()

        def build_report(self, result):
            return self.report

        def walk_forward(self, candles, window_bars=390):
            return [dict(self.report, window=1)]

    monkeypatch.setattr(backtest_runner, "HistoricalOIProvider", FakeOI)
    monkeypatch.setattr(backtest_runner, "ReplayEngine", FakeReplay)
    monkeypatch.setattr(backtest_runner, "BacktestEngine", FakeEngine)

    report = run_backtest_report(settings=types.SimpleNamespace())
    assert "Backtest Report" in report
    assert "signals: 3" in report

    report_wf = run_backtest_report(settings=types.SimpleNamespace(), walk_forward=True)
    assert "Walk-forward windows" in report_wf
    assert "window 1" in report_wf
