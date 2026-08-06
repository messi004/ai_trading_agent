"""Backtest engine with trade simulation + walk-forward (Enhancement Phase 3).

Simulates SL/Target execution bar-by-bar, applies the cost model, and
aggregates performance metrics. Walk-forward splits the series into
consecutive windows for train/validate style reporting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from config.constants import EXIT_REASONS
from config.settings import Settings
from core.candle_engine import Candle
from core.cost_model import CostModel
from core.logger import get_logger
from modules.checker_node import CheckerContext, CheckerNode
from modules.replay_engine import BacktestSignal, ReplayEngine
from utils.metrics import (
    avg_win_loss,
    equity_curve_from_pnl,
    expectancy,
    hit_rate,
    max_consecutive_losses,
    max_drawdown,
    profit_factor,
)

log = get_logger(__name__)

DEFAULT_MAX_HOLD_BARS = 15


@dataclass
class ClosedTrade:
    signal_ts: float
    direction: str
    trigger_type: str
    entry: float
    exit_price: float
    exit_reason: str
    pnl_points: float
    mfe: float
    mae: float
    bars_held: int


@dataclass
class BacktestResult:
    signals: list[BacktestSignal] = field(default_factory=list)
    trades: list[ClosedTrade] = field(default_factory=list)
    rejected_signals: list[BacktestSignal] = field(default_factory=list)

    @property
    def outcomes(self) -> list[str]:
        return [t.exit_reason for t in self.trades]

    @property
    def outcome_labels(self) -> list[str]:
        """Map each closed trade to WIN/LOSS/BE for hit-rate math."""
        labels = []
        for trade in self.trades:
            if trade.pnl_points > 0:
                labels.append("WIN")
            elif trade.pnl_points < 0:
                labels.append("LOSS")
            else:
                labels.append("BE")
        return labels

    @property
    def pnl_points(self) -> list[float]:
        return [t.pnl_points for t in self.trades]


class BacktestEngine:
    def __init__(
        self,
        settings: Settings,
        replay_engine: ReplayEngine,
        cost_model: CostModel | None = None,
        max_hold_bars: int = DEFAULT_MAX_HOLD_BARS,
        checker: CheckerNode | None = None,
    ) -> None:
        self._settings = settings
        self._replay = replay_engine
        self._cost = cost_model or CostModel()
        self._max_hold = max_hold_bars
        self._checker = checker

    def run(self, candles: Sequence[Candle]) -> BacktestResult:
        """Replay candles -> signals -> simulated trades."""
        signals = self._replay.run(list(candles))
        result = BacktestResult(signals=signals)
        for signal in signals:
            if self._checker is not None and not self._checker_approves(signal, list(candles)):
                result.rejected_signals.append(signal)
                continue
            trade = self._simulate_trade(signal, list(candles))
            if trade is not None:
                result.trades.append(trade)
                if self._checker is not None:
                    self._checker.record_exit_pnl(trade.pnl_points)
        return result

    def _checker_approves(self, signal: BacktestSignal, candles: Sequence[Candle]) -> bool:
        """Run the checker on a replay signal; feed Rule D with realized PnL."""
        from core.candle_engine import atr
        from core.signals import StructuredSignal, side_to_direction

        checker = self._checker
        if checker is None:
            return True

        structured = StructuredSignal(
            direction=side_to_direction(signal.direction),
            confidence=0.7,
            entry_zone=(signal.entry - 0.5, signal.entry + 0.5),
            sl=signal.sl,
            target=signal.target,
            rationale=f"{signal.trigger_type} {signal.divergence} at level",
            trap_type="NONE",
            ts_epoch=signal.ts_epoch,
            trigger_type=signal.trigger_type,
            strike=round(signal.entry),
            regime=signal.regime,
        )
        context = CheckerContext(metrics=signal.metrics, atr=atr(candles))
        verdict = checker.check(structured, context)
        return verdict.approved

    def _simulate_trade(
        self, signal: BacktestSignal, candles: Sequence[Candle]
    ) -> ClosedTrade | None:
        """Walk bars after the signal bar until SL/Target/time exit."""
        start = next((i for i, c in enumerate(candles) if c.ts_epoch >= signal.ts_epoch), None)
        if start is None:
            return None
        entry_fill = self._cost.fill_price(signal.entry, signal.direction)
        long = signal.direction == "LONG"
        target_price = entry_fill + signal.target if long else entry_fill - signal.target
        sl_price = entry_fill - signal.sl if long else entry_fill + signal.sl

        mfe = mae = 0.0
        for offset in range(1, self._max_hold + 1):
            idx = start + offset
            if idx >= len(candles):
                break
            bar = candles[idx]
            favorable = bar.high - entry_fill if long else entry_fill - bar.low
            adverse = entry_fill - bar.low if long else bar.high - entry_fill
            mfe = max(mfe, favorable)
            mae = max(mae, adverse)

            if long and bar.high >= target_price:
                return self._close(signal, entry_fill, target_price, "TARGET_HIT", mfe, mae, offset)
            if long and bar.low <= sl_price:
                return self._close(signal, entry_fill, sl_price, "SL_HIT", mfe, mae, offset)
            if not long and bar.low <= target_price:
                return self._close(signal, entry_fill, target_price, "TARGET_HIT", mfe, mae, offset)
            if not long and bar.high >= sl_price:
                return self._close(signal, entry_fill, sl_price, "SL_HIT", mfe, mae, offset)

        exit_bar = candles[min(start + self._max_hold, len(candles) - 1)]
        return self._close(
            signal,
            entry_fill,
            exit_bar.close,
            "TIME_EXIT",
            mfe,
            mae,
            min(self._max_hold, len(candles) - 1 - start),
        )

    def _close(
        self,
        signal: BacktestSignal,
        entry_fill: float,
        exit_fill: float,
        reason: str,
        mfe: float,
        mae: float,
        bars_held: int,
    ) -> ClosedTrade:
        exit_fill = self._cost.fill_price(exit_fill, signal.direction)
        return ClosedTrade(
            signal_ts=signal.ts_epoch,
            direction=signal.direction,
            trigger_type=signal.trigger_type,
            entry=entry_fill,
            exit_price=exit_fill,
            exit_reason=reason,
            pnl_points=self._cost.net_pnl_points(entry_fill, exit_fill, signal.direction),
            mfe=mfe,
            mae=mae,
            bars_held=bars_held,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def build_report(self, result: BacktestResult) -> dict:
        pnl = result.pnl_points
        gross_profit = sum(p for p in pnl if p > 0)
        gross_loss = sum(p for p in pnl if p < 0)
        return {
            "signals": len(result.signals),
            "trades": len(result.trades),
            "rejected_by_checker": len(result.rejected_signals),
            "hit_rate": hit_rate(result.outcome_labels),
            "expectancy_points": expectancy(pnl),
            "profit_factor": profit_factor(gross_profit, gross_loss),
            "max_drawdown_points": max_drawdown(equity_curve_from_pnl(pnl)),
            "max_consecutive_losses": max_consecutive_losses(result.outcome_labels),
            "avg_win_points": avg_win_loss(pnl)[0],
            "avg_loss_points": avg_win_loss(pnl)[1],
            "exit_reason_counts": {
                reason: result.outcomes.count(reason) for reason in EXIT_REASONS
            },
        }

    def walk_forward(
        self,
        candles: Sequence[Candle],
        window_bars: int = 390,
    ) -> list[dict]:
        """Split into consecutive windows and backtest each independently."""
        windows: list[dict] = []
        total = len(candles)
        index = 0
        window_no = 1
        while index < total:
            chunk = list(candles[index : index + window_bars])
            result = self.run(chunk)
            report = self.build_report(result)
            report["window"] = window_no
            report["start_ts"] = chunk[0].ts_epoch if chunk else None
            report["end_ts"] = chunk[-1].ts_epoch if chunk else None
            windows.append(report)
            index += window_bars
            window_no += 1
        return windows
