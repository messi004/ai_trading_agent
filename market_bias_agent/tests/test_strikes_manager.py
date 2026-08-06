"""Unit tests for strikes manager + redis stream additions (Phase 1)."""

from __future__ import annotations

import types

import fakeredis

from core.redis_manager import RedisManager
from core.strikes_manager import StrikesManager


def _make_redis() -> RedisManager:
    settings = types.SimpleNamespace()
    mgr = RedisManager(settings)
    mgr.client = fakeredis.FakeRedis(decode_responses=True)
    return mgr


class FakeProvider:
    def __init__(self, expiry: str, strikes: list[int]):
        self.expiry = expiry
        self.strikes = strikes
        self.calls = 0

    def fetch_expiry_and_strikes(self, symbol: str) -> tuple[str, list[int]]:
        self.calls += 1
        return self.expiry, self.strikes


class TestRedisStream:
    def test_append_and_read_raw_ticks(self) -> None:
        r = _make_redis()
        r.append_raw_tick({"type": "spot", "price": 24000})
        r.append_raw_tick({"type": "spot", "price": 24005})
        ticks = r.read_raw_ticks()
        assert len(ticks) == 2
        assert ticks[0]["price"] == "24000"
        assert r.stream_length() == 2

    def test_strikes_set_roundtrip(self) -> None:
        r = _make_redis()
        r.set_strikes([24000, 24100, 24200])
        assert r.get_strikes() == [24000, 24100, 24200]
        r.set_strikes([])
        assert r.get_strikes() == []

    def test_active_expiry_roundtrip(self) -> None:
        r = _make_redis()
        assert r.get_active_expiry() is None
        r.set_active_expiry("2026-08-13")
        assert r.get_active_expiry() == "2026-08-13"


class TestStrikesManager:
    def test_sync_persists_strikes_and_expiry(self) -> None:
        r = _make_redis()
        mgr = StrikesManager(
            types.SimpleNamespace(nifty_symbol="NIFTY"),
            r,
            FakeProvider("2026-08-13", [24000, 24100]),
        )
        summary = mgr.sync()
        assert summary["expiry"] == "2026-08-13"
        assert r.get_strikes() == [24000, 24100]
        assert r.get_active_expiry() == "2026-08-13"
        assert not mgr.rollover_detected

    def test_expiry_rollover_detected_and_buffers_reset(self) -> None:
        r = _make_redis()
        provider = FakeProvider("2026-08-13", [24000, 24100])
        mgr = StrikesManager(types.SimpleNamespace(nifty_symbol="NIFTY"), r, provider)
        mgr.sync()
        r.push_call_oi(24000, 100.0)
        assert len(r.get_call_oi_window(24000)) == 1

        provider.expiry = "2026-09-03"  # rollover
        mgr.sync()
        assert mgr.rollover_detected
        assert r.get_call_oi_window(24000) == []
        assert r.get_active_expiry() == "2026-09-03"

    def test_sync_if_due_respects_interval(self) -> None:
        r = _make_redis()
        provider = FakeProvider("2026-08-13", [24000])
        mgr = StrikesManager(
            types.SimpleNamespace(nifty_symbol="NIFTY"), r, provider, sync_interval_seconds=100
        )
        assert mgr.sync_if_due(now=0.0) is True
        assert mgr.sync_if_due(now=50.0) is False
        assert mgr.sync_if_due(now=150.0) is True
        assert provider.calls == 2

    def test_nearest_strikes_window(self) -> None:
        r = _make_redis()
        strikes = list(range(23000, 25100, 100))
        r.set_strikes(strikes)
        mgr = StrikesManager(
            types.SimpleNamespace(nifty_symbol="NIFTY"), r, FakeProvider("2026-08-13", strikes)
        )
        near = mgr.nearest_strikes(spot=24050, range_around_atm=2)
        assert near == [23800, 23900, 24000, 24100, 24200]
