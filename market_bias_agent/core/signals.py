"""Structured signal schema (Enhancement Phase 4).

A strict, parseable signal produced by the Maker node and validated/enriched
by the Checker node. All fields mirror the PRD schema exactly; validation is
fail-fast so malformed output never reaches execution.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from config.constants import (
    MAKER_REQUIRED_FIELDS,
    SIGNAL_DIRECTIONS,
    SIGNAL_SIDES,
    TRAP_TYPES,
)


@dataclass
class StructuredSignal:
    """Canonical signal record.

    `direction` follows the PRD schema (BULLISH/BEARISH/NEUTRAL). The internal
    trading side (LONG/SHORT) is derived via `side()`.
    """

    direction: str
    confidence: float
    entry_zone: tuple[float, float]
    sl: float
    target: float
    rationale: str
    trap_type: str
    ts_epoch: float = 0.0
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    trigger_type: str = "SCALP"
    strike: float = 0.0
    regime: str = "ACTIVE"
    metadata: dict[str, Any] = field(default_factory=dict)

    def side(self) -> str:
        """Long/short side for execution ('' when direction is NEUTRAL)."""
        if self.direction == "BULLISH":
            return "LONG"
        if self.direction == "BEARISH":
            return "SHORT"
        return ""

    @property
    def entry(self) -> float:
        """Mid-point of the entry zone (backtest/paper-trader convenience)."""
        return (self.entry_zone[0] + self.entry_zone[1]) / 2.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> StructuredSignal:
        return cls(**{k: v for k, v in raw.items() if k in _FIELD_NAMES})


_FIELD_NAMES = {
    "direction",
    "confidence",
    "entry_zone",
    "sl",
    "target",
    "rationale",
    "trap_type",
    "ts_epoch",
    "signal_id",
    "trigger_type",
    "strike",
    "regime",
    "metadata",
}


def side_to_direction(side: str) -> str:
    """Map LONG/SHORT to the PRD BULLISH/BEARISH direction."""
    if side.upper() == "LONG":
        return "BULLISH"
    if side.upper() == "SHORT":
        return "BEARISH"
    return "NEUTRAL"


def validate_maker_signal(raw: dict[str, Any]) -> list[str]:
    """Validate a Maker output dict against the strict signal schema.

    Returns a list of error strings (empty == valid).
    """
    errors: list[str] = []
    for field_name in MAKER_REQUIRED_FIELDS:
        if field_name not in raw:
            errors.append(f"missing field: {field_name}")

    direction = raw.get("direction")
    if direction is not None and direction not in SIGNAL_DIRECTIONS:
        errors.append(f"direction must be one of {SIGNAL_DIRECTIONS}, got {direction!r}")

    confidence = raw.get("confidence")
    if confidence is not None and not isinstance(confidence, int | float):
        errors.append("confidence must be numeric")
    elif isinstance(confidence, int | float) and not 0.0 <= confidence <= 1.0:
        errors.append(f"confidence must be in [0,1], got {confidence}")

    entry_zone = raw.get("entry_zone")
    if entry_zone is not None:
        if not isinstance(entry_zone, list | tuple) or len(entry_zone) != 2:
            errors.append("entry_zone must be [low, high]")
        else:
            try:
                low, high = float(entry_zone[0]), float(entry_zone[1])
            except (TypeError, ValueError):
                errors.append("entry_zone values must be numeric")
            else:
                if low > high:
                    errors.append(f"entry_zone low {low} > high {high}")

    sl = raw.get("sl")
    if sl is not None:
        if not isinstance(sl, int | float) or sl <= 0:
            errors.append(f"sl must be a positive number, got {sl!r}")

    target = raw.get("target")
    if target is not None:
        if not isinstance(target, int | float) or target <= 0:
            errors.append(f"target must be a positive number, got {target!r}")

    trap_type = raw.get("trap_type")
    if trap_type is not None and trap_type not in TRAP_TYPES:
        errors.append(f"trap_type must be one of {TRAP_TYPES}, got {trap_type!r}")

    rationale = raw.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        errors.append("rationale must be a string")

    return errors


def direction_matches_schema(direction: str) -> bool:
    return direction in SIGNAL_DIRECTIONS


def valid_sides() -> tuple[str, ...]:
    return SIGNAL_SIDES
