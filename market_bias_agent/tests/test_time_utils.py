"""Unit tests for IST time utilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from utils.time_utils import IST, market_status, now_ist, parse_time_ist, to_ist


def test_now_ist_is_aware() -> None:
    assert now_ist().tzinfo is not None
    assert now_ist().utcoffset() == timedelta(hours=5, minutes=30)


def test_to_ist_naive_treated_as_ist() -> None:
    naive = datetime(2026, 8, 6, 12, 0, 0)
    assert to_ist(naive).tzinfo == IST
    assert to_ist(naive).hour == 12


def test_to_ist_converts_utc() -> None:
    utc = datetime(2026, 8, 6, 5, 0, 0, tzinfo=timezone.utc)
    assert to_ist(utc).hour == 10  # UTC+5:30


def test_parse_time_ist() -> None:
    t = parse_time_ist("09:15")
    assert (t.hour, t.minute) == (9, 15)


def test_market_status_open() -> None:
    t = datetime(2026, 8, 6, 12, 0, 0, tzinfo=IST)
    assert market_status(t) == "OPEN"


def test_market_status_pre_and_post() -> None:
    pre = datetime(2026, 8, 6, 8, 0, 0, tzinfo=IST)
    post = datetime(2026, 8, 6, 16, 0, 0, tzinfo=IST)
    assert market_status(pre) == "PRE"
    assert market_status(post) == "POST"
