"""Pre-market level calculations (PRD Module 6).

Pure, deterministic functions that compute next-day Support/Resistance and
Max-Pain pinning zones from the previous session's data. Keeping them I/O
free means the live premarket cron and the backtest/offline paths share the
exact same logic.

Formulas:
  * Classic pivot points (floor-trader style)
        P   = (H + L + C) / 3
        R1  = 2P - L        S1 = 2P - H
        R2  = P + (H - L)   S2 = P - (H - L)
  * Psychological round levels at `base` spacing around the pivot
  * Max Pain = strike with the highest combined Call+Put OI
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from config.constants import (
    MAX_PAIN_PINNING_TOLERANCE,
    OI_LEVEL_MIN_STRIKES,
    OI_WALL_MIN_RELATIVE_OI,
    OI_WALLS_PER_SIDE,
    S_R_LEVEL_ROUND_BASE,
)


class PreMarketLevelsError(ValueError):
    """Raised on invalid inputs for level calculations."""


@dataclass(frozen=True)
class SupportResistance:
    """Pivot point S/R fan plus psychological round levels."""

    pivot: float
    r1: float
    r2: float
    s1: float
    s2: float
    psych_resistance: float
    psych_support: float

    def ordered_levels(self) -> list[float]:
        """All levels sorted ascending (S2, S1, pivot, R1, R2)."""
        return sorted({self.s2, self.s1, self.pivot, self.r1, self.r2})

    def to_dict(self) -> dict:
        return {
            "pivot": round(self.pivot, 2),
            "r1": round(self.r1, 2),
            "r2": round(self.r2, 2),
            "s1": round(self.s1, 2),
            "s2": round(self.s2, 2),
            "psych_resistance": self.psych_resistance,
            "psych_support": self.psych_support,
        }


def compute_pivot_sr(
    previous_high: float,
    previous_low: float,
    previous_close: float,
    *,
    base: int = S_R_LEVEL_ROUND_BASE,
) -> SupportResistance:
    """Classic pivot points + nearest psychological round levels.

    Raises if the high/low/close are inconsistent (high < low).
    """
    if previous_high < previous_low:
        raise PreMarketLevelsError(
            f"high {previous_high} < low {previous_low} — invalid session range"
        )
    if previous_high <= 0 or previous_low <= 0 or previous_close <= 0:
        raise PreMarketLevelsError("high/low/close must all be positive")

    pivot = (previous_high + previous_low + previous_close) / 3.0
    r1 = 2.0 * pivot - previous_low
    s1 = 2.0 * pivot - previous_high
    r2 = pivot + (previous_high - previous_low)
    s2 = pivot - (previous_high - previous_low)

    def _round_level(price: float, direction: int) -> float:
        # nearest base-multiple in the chosen direction
        rounded = round(price / base) * base
        if direction > 0 and rounded <= price:
            rounded += base
        elif direction < 0 and rounded >= price:
            rounded -= base
        return float(rounded)

    psych_resistance = _round_level(pivot, direction=1)
    psych_support = _round_level(pivot, direction=-1)

    return SupportResistance(
        pivot=pivot,
        r1=r1,
        r2=r2,
        s1=s1,
        s2=s2,
        psych_resistance=psych_resistance,
        psych_support=psych_support,
    )


def combined_oi_by_strike(
    call_oi: dict[int, float],
    put_oi: dict[int, float],
) -> dict[int, float]:
    """Sum Call+Put OI per strike (union of both strike sets)."""
    strikes = set(call_oi) | set(put_oi)
    return {strike: call_oi.get(strike, 0.0) + put_oi.get(strike, 0.0) for strike in strikes}


def find_max_pain_strike(combined_oi: dict[int, float]) -> int | None:
    """Strike with the highest total OI (option-writer pinning)."""
    if not combined_oi:
        return None
    return max(combined_oi, key=lambda s: combined_oi[s])


def max_pain_zone(
    combined_oi: dict[int, float],
    tolerance: float = MAX_PAIN_PINNING_TOLERANCE,
) -> tuple[float, float] | None:
    """(low, high) band around the max-pain strike used by the live engine."""
    strike = find_max_pain_strike(combined_oi)
    if strike is None:
        return None
    return (strike - tolerance, strike + tolerance)


@dataclass(frozen=True)
class OIWallLevels:
    """Live intraday S/R walls derived from option-chain OI concentration.

    * ``resistance`` — strikes with the heaviest Call OI above spot (call walls)
    * ``support`` — strikes with the heaviest Put OI below spot (put walls)
    * ``max_pain`` — strike with the highest combined Call+Put OI

    Levels are strike prices (50-spaced for Nifty index options), ascending.
    """

    resistance: list[float]
    support: list[float]
    max_pain: float | None

    def ordered_levels(self) -> list[float]:
        """All walls + max-pain strike, sorted ascending."""
        levels = [*self.resistance, *self.support]
        if self.max_pain is not None:
            levels.append(self.max_pain)
        return sorted(set(levels))


def _top_walls(
    oi: dict[int, float],
    spot: float,
    *,
    direction: int,
    walls: int,
    min_relative: float,
) -> list[float]:
    """Top-N strongest OI walls on one side of spot.

    ``direction > 0`` looks above spot (Call walls -> resistance);
    ``direction < 0`` looks below spot (Put walls -> support). A wall must
    carry at least ``min_relative`` (fraction 0..1) of the max OI on that side
    to be considered significant, which filters out low-volume noise strikes.
    """
    if not oi:
        return []
    candidates = [
        (s, v) for s, v in oi.items() if v > 0 and (s > spot if direction > 0 else s < spot)
    ]
    if not candidates:
        return []
    max_oi = max(v for _, v in candidates)
    threshold = max_oi * min_relative
    strong = sorted((s for s, v in candidates if v >= threshold), key=lambda s: -oi[s])[:walls]
    return sorted(float(s) for s in strong)


def oi_wall_levels(
    call_oi: dict[int, float],
    put_oi: dict[int, float],
    spot: float,
    *,
    walls_per_side: int = OI_WALLS_PER_SIDE,
    min_relative: float = OI_WALL_MIN_RELATIVE_OI,
    min_strikes: int = OI_LEVEL_MIN_STRIKES,
) -> OIWallLevels:
    """Live S/R walls + max pain from per-strike OI, 50-spaced.

    Returns empty walls when fewer than ``min_strikes`` strikes carry OI
    (before market open / first minutes of a session the profile is unreliable).
    """
    call_oi = {s: v for s, v in call_oi.items() if v > 0}
    put_oi = {s: v for s, v in put_oi.items() if v > 0}
    combined = combined_oi_by_strike(call_oi, put_oi)
    if len(combined) < min_strikes:
        return OIWallLevels([], [], None)
    max_pain = find_max_pain_strike(combined)
    resistance = _top_walls(
        call_oi, spot, direction=1, walls=walls_per_side, min_relative=min_relative
    )
    support = _top_walls(
        put_oi, spot, direction=-1, walls=walls_per_side, min_relative=min_relative
    )
    return OIWallLevels(
        resistance=resistance,
        support=support,
        max_pain=float(max_pain) if max_pain is not None else None,
    )


def psychological_levels_around(
    price: float,
    span: int = 3,
    *,
    base: int = S_R_LEVEL_ROUND_BASE,
) -> list[float]:
    """Round levels ±`span` around `price` (e.g. 23900, 24000, 24100...)."""
    center = round(price / base) * base
    return [float(center + i * base) for i in range(-span, span + 1)]


def nearest_level(spot: float, levels: Iterable[float]) -> float | None:
    """Level closest to the spot price, or None when empty."""
    level_list = list(levels)
    if not level_list:
        return None
    return min(level_list, key=lambda level: abs(level - spot))


def session_bounds_from_ticks(ticks: Sequence[dict]) -> tuple[float, float, float] | None:
    """(high, low, close) of a session from Redis spot ticks.

    `ticks` is a list of dicts with a numeric "price". The last tick's price
    is used as the close; returns None when there are no usable ticks.
    """
    prices = [float(t["price"]) for t in ticks if t.get("price") is not None]
    if not prices:
        return None
    return (max(prices), min(prices), prices[-1])
