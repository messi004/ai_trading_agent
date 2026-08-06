"""Timezone utilities — single source of truth for IST.

The container runs with TZ=Asia/Kolkata, but code must NEVER rely on the
server-local clock. All schedule/expiry decisions use aware IST datetimes.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

UTC = timezone.utc


def now_ist() -> datetime:
    """Current time as an aware datetime in Asia/Kolkata."""
    return datetime.now(IST)


def to_ist(dt: datetime) -> datetime:
    """Convert any aware/naive datetime to IST. Naive input is treated as IST."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def iso_ist(dt: datetime | None = None) -> str:
    """ISO-8601 string of an IST datetime (aware), e.g. 2026-08-06T18:00:00+05:30."""
    return (dt or now_ist()).isoformat()


def parse_time_ist(hhmm: str) -> time:
    """Parse 'HH:MM' into an IST-aware time object."""
    hour, minute = hhmm.split(":")
    return time(hour=int(hour), minute=int(minute))


def market_status(
    now: datetime | None = None,
    open_ist: str = "09:15",
    close_ist: str = "15:30",
) -> str:
    """Return 'PRE', 'OPEN', or 'POST' for the given/current IST time.

    Weekends (Sat/Sun) and expiry-day early close are out of scope here;
    extend via a trading-calendar in Phase 3.
    """
    current = to_ist(now) if now is not None else now_ist()
    open_t, close_t = parse_time_ist(open_ist), parse_time_ist(close_ist)
    today = current.time()
    if today < open_t:
        return "PRE"
    if today > close_t:
        return "POST"
    return "OPEN"
