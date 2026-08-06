"""Paper trader — shadow order routing (Enhancement Phase 3).

Runs the live signal path in simulation: opens positions on signals and
resolves SL/Target/time exits from incoming prices. Tracks PnL and compares
the as-alerted price vs the as-executed (slippage-adjusted) fill.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from config.settings import Settings
from core.cost_model import CostModel
from core.logger import get_logger

log = get_logger(__name__)

DEFAULT_MAX_HOLD_SECONDS = 900.0


@dataclass
class PaperPosition:
    signal_id: str
    direction: str
    entry_alert: float
    entry_fill: float
    sl: float
    target: float
    opened_at: float
    status: str = "OPEN"  # OPEN | CLOSED
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl_points: float | None = None
    slippage_points: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


class PaperTrader:
    def __init__(
        self,
        settings: Settings,
        cost_model: CostModel | None = None,
        max_hold_seconds: float = DEFAULT_MAX_HOLD_SECONDS,
    ) -> None:
        self._settings = settings
        self._cost = cost_model or CostModel()
        self._max_hold = max_hold_seconds
        self._positions: dict[str, PaperPosition] = {}
        self._closed: list[PaperPosition] = []

    def submit_signal(self, signal: dict[str, Any]) -> str:
        """Open a paper position from a signal dict."""
        direction = str(signal.get("direction", "")).upper()
        if direction in ("BULLISH", "LONG"):
            direction = "LONG"
        elif direction in ("BEARISH", "SHORT"):
            direction = "SHORT"
        else:
            raise ValueError(f"direction must be LONG/SHORT, got {direction!r}")
        if "entry" in signal and signal.get("entry"):
            entry_alert = float(signal["entry"])
        else:
            zone = signal.get("entry_zone")
            if not zone:
                raise ValueError("signal needs an entry or entry_zone")
            entry_alert = float((float(zone[0]) + float(zone[1])) / 2.0)
        entry_fill = self._cost.fill_price(entry_alert, direction)
        position = PaperPosition(
            signal_id=str(signal.get("signal_id") or uuid.uuid4().hex),
            direction=direction,
            entry_alert=entry_alert,
            entry_fill=entry_fill,
            sl=float(signal.get("sl", 4.0)),
            target=float(signal.get("target", 6.0)),
            opened_at=float(signal.get("ts_epoch", 0.0)),
            slippage_points=self._cost.slippage_points,
        )
        self._positions[position.signal_id] = position
        return position.signal_id

    def update_price(self, price: float, ts_epoch: float) -> list[PaperPosition]:
        """Feed a price; resolve fills. Returns positions closed on this tick."""
        closed: list[PaperPosition] = []
        for position in list(self._positions.values()):
            if position.status != "OPEN":
                continue
            position.history.append({"ts_epoch": ts_epoch, "price": price})
            long = position.direction == "LONG"
            favorable = price - position.entry_fill if long else position.entry_fill - price
            adverse = position.entry_fill - price if long else price - position.entry_fill
            position.mfe = max(position.mfe, favorable)
            position.mae = max(position.mae, adverse)

            if long and price >= position.entry_fill + position.target:
                self._close(position, "TARGET_HIT", price)
                closed.append(position)
            elif long and price <= position.entry_fill - position.sl:
                self._close(position, "SL_HIT", price)
                closed.append(position)
            elif not long and price <= position.entry_fill - position.target:
                self._close(position, "TARGET_HIT", price)
                closed.append(position)
            elif not long and price >= position.entry_fill + position.sl:
                self._close(position, "SL_HIT", price)
                closed.append(position)
            elif ts_epoch - position.opened_at > self._max_hold:
                self._close(position, "TIME_EXIT", price)
                closed.append(position)
        return closed

    def open_positions(self) -> list[PaperPosition]:
        return [p for p in self._positions.values() if p.status == "OPEN"]

    def closed_positions(self) -> list[PaperPosition]:
        return list(self._closed)

    def _close(self, position: PaperPosition, reason: str, price: float) -> None:
        exit_fill = self._cost.fill_price(price, position.direction)
        position.status = "CLOSED"
        position.exit_reason = reason
        position.exit_price = exit_fill
        position.pnl_points = self._cost.net_pnl_points(
            position.entry_fill, exit_fill, position.direction
        )
        self._closed.append(position)
        log.info(
            "paper_trade_closed",
            extra={
                "signal_id": position.signal_id,
                "reason": reason,
                "pnl_points": round(position.pnl_points, 2),
                "mfe": round(position.mfe, 2),
                "mae": round(position.mae, 2),
            },
        )

    def report(self) -> dict:
        """Aggregate stats across closed positions."""
        closed = self._closed
        pnl = [p.pnl_points or 0.0 for p in closed]
        wins = sum(1 for p in closed if p.exit_reason == "TARGET_HIT")
        losses = sum(1 for p in closed if p.exit_reason == "SL_HIT")
        return {
            "positions_opened": len(closed) + len(self.open_positions()),
            "closed": len(closed),
            "open": len(self.open_positions()),
            "target_hits": wins,
            "sl_hits": losses,
            "time_exits": sum(1 for p in closed if p.exit_reason == "TIME_EXIT"),
            "total_pnl_points": round(sum(pnl), 2),
            "avg_pnl_points": round(sum(pnl) / len(pnl), 2) if pnl else 0.0,
            "hit_rate": round(wins / (wins + losses), 3) if (wins + losses) else 0.0,
        }
