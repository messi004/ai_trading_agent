"""Performance metrics calculators (Enhancement Phase 3/8)."""

from __future__ import annotations

from collections.abc import Sequence


def hit_rate(outcomes: Sequence[str]) -> float:
    """Fraction of WIN outcomes (LOSS counts against, BE ignored)."""
    wins = sum(1 for o in outcomes if o == "WIN")
    losses = sum(1 for o in outcomes if o == "LOSS")
    total = wins + losses
    if total == 0:
        return 0.0
    return wins / total


def expectancy(pnl_points: Sequence[float]) -> float:
    """Mean PnL in points across all closed signals."""
    if not pnl_points:
        return 0.0
    return sum(pnl_points) / len(pnl_points)


def profit_factor(gross_profit: float, gross_loss: float) -> float:
    """Gross profit / abs(gross loss); inf if no losses."""
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / abs(gross_loss)


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Max peak-to-trough drawdown as a positive value."""
    peak = float("-inf")
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = peak - value
        if dd > max_dd:
            max_dd = dd
    return max_dd


def max_consecutive_losses(outcomes: Sequence[str]) -> int:
    """Longest run of consecutive LOSS outcomes."""
    best = current = 0
    for outcome in outcomes:
        if outcome == "LOSS":
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def avg_win_loss(pnl_points: Sequence[float]) -> tuple[float, float]:
    """(average winning PnL, average losing PnL) in points."""
    wins = [p for p in pnl_points if p > 0]
    losses = [p for p in pnl_points if p < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return avg_win, avg_loss


def equity_curve_from_pnl(pnl_points: Sequence[float], starting_equity: float = 0.0) -> list[float]:
    """Cumulative PnL curve (point-based)."""
    curve: list[float] = []
    running = starting_equity
    for pnl in pnl_points:
        running += pnl
        curve.append(running)
    return curve
