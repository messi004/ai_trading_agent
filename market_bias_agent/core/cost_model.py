"""Slippage & cost model (Enhancement Phase 3).

Keeps the backtest honest: every simulated fill is worsened by slippage in
the adverse direction, and a per-trade cost is subtracted at close.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    slippage_points: float = 1.0  # adverse slippage per fill (points)
    cost_per_trade_points: float = 0.5  # brokerage + taxes equivalent in points

    def fill_price(self, alert_price: float, direction: str) -> float:
        """Adverse-slippage entry/exit fill for a long (+) or short (-) side."""
        sign = 1.0 if direction == "LONG" else -1.0
        return alert_price + sign * self.slippage_points

    def net_pnl_points(self, entry: float, exit_price: float, direction: str) -> float:
        """PnL in points after costs, for a round-trip on `direction`."""
        multiplier = 1.0 if direction == "LONG" else -1.0
        gross = (exit_price - entry) * multiplier
        return gross - self.cost_per_trade_points
