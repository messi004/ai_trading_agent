"""Trap event records & payload schema (Enhancement Phase 5).

A historical trap event captures the market state at alert time plus the
outcome that followed. Stored as Qdrant payloads so similarity search can
combine vector distance with scalar filtering (expiry_week, session_date).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from config.constants import TRAP_OUTCOMES


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def compute_market_state(features: dict[str, Any]) -> str:
    """Human-readable market state string, e.g. 'PCR: 0.95, Spot: 24005, Call_OI_Vel: -85000'."""
    pcr = features.get("pcr", 0.0)
    spot = features.get("spot", 0.0)
    call_vel = features.get("call_oi_vel_1m", 0.0)
    return f"PCR: {pcr:.2f}, Spot: {spot:.0f}, Call_OI_Vel: {call_vel:,.0f}"


def compute_subsequent_move(price_before: float, price_after: float, minutes: int) -> str:
    """Describe the post-alert price move, e.g. '-45 points in 15 mins'."""
    points = price_after - price_before
    return f"{points:+.0f} points in {minutes} mins"


def parse_subsequent_move_points(subsequent_move: str) -> float | None:
    """Extract the signed points from '±N points in M mins'; None if unparsable."""
    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*points", subsequent_move)
    if not match:
        return None
    return float(match.group(1))


@dataclass
class TrapRecord:
    """One historical trap event, ready to index into Qdrant."""

    features: dict[str, Any]
    historical_outcome: str
    subsequent_move: str
    session_date: str  # YYYY-MM-DD
    expiry_week: int
    vector_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(default_factory=utc_now_iso)
    market_state: str = ""
    actual_outcome: str = ""  # Phase 8: learned outcome for live signals

    def __post_init__(self) -> None:
        if not self.market_state:
            self.market_state = compute_market_state(self.features)
        if self.historical_outcome not in TRAP_OUTCOMES:
            raise ValueError(
                "historical_outcome must be one of "
                f"{TRAP_OUTCOMES}, got {self.historical_outcome!r}"
            )
        if self.actual_outcome and self.actual_outcome not in TRAP_OUTCOMES:
            raise ValueError(
                "actual_outcome must be one of " f"{TRAP_OUTCOMES}, got {self.actual_outcome!r}"
            )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, raw: dict[str, Any]) -> TrapRecord:
        return cls(**{k: v for k, v in raw.items() if k in _FIELDS})


_FIELDS = {
    "vector_id",
    "timestamp",
    "market_state",
    "features",
    "historical_outcome",
    "subsequent_move",
    "session_date",
    "expiry_week",
    "actual_outcome",
}
