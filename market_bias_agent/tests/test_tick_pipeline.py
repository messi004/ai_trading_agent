"""Unit tests for tick pipeline (validate -> redis, Phase 1)."""

from __future__ import annotations

import time
import types

import fakeredis

from core.redis_manager import RedisManager
from core.tick_pipeline import TickPipeline
from core.tick_validator import TickValidator

NOW = time.time()


def _make() -> tuple[TickPipeline, RedisManager]:
    settings = types.SimpleNamespace()
    redis = RedisManager(settings)
    redis.client = fakeredis.FakeRedis(decode_responses=True)
    pipeline = TickPipeline(settings, redis, TickValidator())
    return pipeline, redis


def test_process_spot_tick_updates_buffers_and_stream() -> None:
    pipeline, redis = _make()
    tick = pipeline.process(
        {"type": "spot", "symbol": "NIFTY", "price": 24005.5, "volume": 10, "ts_epoch": NOW}
    )
    assert tick is not None
    assert tick.type == "spot"
    assert redis.get_spot_ticks(1)[0]["price"] == 24005.5
    assert redis.stream_length() == 1


def test_process_oi_tick_updates_window() -> None:
    pipeline, redis = _make()
    pipeline.process(
        {
            "type": "oi",
            "symbol": "NIFTY",
            "strike": 24000,
            "option_type": "CALL",
            "oi": 100_000,
            "ts_epoch": NOW,
        }
    )
    assert redis.get_call_oi_window(24000) == [100_000.0]
    pipeline.process(
        {
            "type": "oi",
            "symbol": "NIFTY",
            "strike": 24000,
            "option_type": "PUT",
            "oi": 90_000,
            "ts_epoch": NOW + 1,
        }
    )
    assert redis.get_put_oi_window(24000) == [90_000.0]


def test_process_drops_invalid_and_increments_stats() -> None:
    pipeline, redis = _make()
    assert (
        pipeline.process({"type": "spot", "symbol": "NIFTY", "price": "x", "ts_epoch": NOW}) is None
    )
    assert pipeline.stats.dropped_malformed == 1
    assert redis.stream_length() == 0
