"""Unit tests for RedisManager buffer logic using fakeredis."""

from __future__ import annotations

import types

import fakeredis

from config.constants import OI_WINDOW_INTERVALS, SPOT_TICK_BUFFER_SIZE
from core.redis_manager import RedisManager


def _make_manager() -> tuple[RedisManager, fakeredis.FakeRedis]:
    settings = types.SimpleNamespace()
    mgr = RedisManager(settings)
    fake = fakeredis.FakeRedis(decode_responses=True)
    mgr.client = fake
    return mgr, fake


def test_push_and_read_spot_ticks() -> None:
    mgr, _ = _make_manager()
    for i in range(3):
        mgr.push_spot_tick({"price": 24000 + i, "ts_epoch": 1.0})
    ticks = mgr.get_spot_ticks()
    assert len(ticks) == 3
    assert ticks[-1]["price"] == 24002


def test_spot_tick_circular_buffer_limit() -> None:
    mgr, _ = _make_manager()
    for i in range(SPOT_TICK_BUFFER_SIZE + 50):
        mgr.push_spot_tick({"price": i, "ts_epoch": float(i)})
    ticks = mgr.get_spot_ticks()
    assert len(ticks) == SPOT_TICK_BUFFER_SIZE
    assert ticks[0]["price"] == 50


def test_oi_window_sliding() -> None:
    mgr, _ = _make_manager()
    for i in range(OI_WINDOW_INTERVALS + 10):
        mgr.push_call_oi(24000, float(i))
    window = mgr.get_call_oi_window(24000)
    assert len(window) == OI_WINDOW_INTERVALS
    assert window[-1] == OI_WINDOW_INTERVALS + 9


def test_last_tick_age_when_empty() -> None:
    mgr, _ = _make_manager()
    assert mgr.last_tick_age_seconds() is None
