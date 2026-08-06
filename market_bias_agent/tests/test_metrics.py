"""Unit tests for backtest/paper performance metrics."""

from __future__ import annotations

from utils.metrics import (
    avg_win_loss,
    equity_curve_from_pnl,
    max_consecutive_losses,
    max_drawdown,
)


def test_max_consecutive_losses() -> None:
    assert max_consecutive_losses(["WIN", "LOSS", "LOSS", "WIN", "LOSS"]) == 2
    assert max_consecutive_losses(["WIN", "WIN"]) == 0


def test_avg_win_loss() -> None:
    avg_win, avg_loss = avg_win_loss([10.0, -4.0, 6.0, -2.0])
    assert avg_win == 8.0
    assert avg_loss == -3.0


def test_avg_win_loss_empty() -> None:
    assert avg_win_loss([]) == (0.0, 0.0)


def test_equity_curve_and_drawdown() -> None:
    curve = equity_curve_from_pnl([10, -5, 3, -20, 30])
    assert curve == [10, 5, 8, -12, 18]
    assert max_drawdown(curve) == 22.0  # peak 10 -> trough -12
