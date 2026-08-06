"""Unit tests for WebSocket resilience (Phase 1)."""

from __future__ import annotations

import asyncio
import types

import pytest

from core.websocket_client import BreezeWebSocketClient, backoff_delay


class FakeTransport:
    def __init__(self, messages: list[dict] | None = None):
        self.messages = list(messages or [])
        self.subs: list[str] = []
        self.connect_calls = 0
        self.closed = False

    async def connect(self, subscriptions: list[str]) -> None:
        self.connect_calls += 1
        self.subs = subscriptions

    async def receive(self):
        if self.messages:
            return self.messages.pop(0)
        return None

    async def close(self) -> None:
        self.closed = True


def _client(transport, subs=None) -> BreezeWebSocketClient:
    settings = types.SimpleNamespace()
    return BreezeWebSocketClient(settings, transport, subscriptions_provider=lambda: subs or [])


def test_backoff_delay_bounds_and_cap() -> None:
    for attempt in range(8):
        d = backoff_delay(attempt)
        assert 0 <= d <= 60.0


@pytest.mark.asyncio
async def test_run_processes_ticks_and_subscribes() -> None:
    transport = FakeTransport(
        [
            {"type": "spot", "price": 1},
            {"type": "spot", "price": 2},
        ]
    )
    client = _client(transport, subs=["NIFTY", "STK24000"])
    received: list[dict] = []

    def handler(raw):
        received.append(raw)
        client.stop()

    client.set_tick_handler(handler)
    await client.run()
    assert received == [{"type": "spot", "price": 1}]
    assert transport.connect_calls == 1
    assert transport.subs == ["NIFTY", "STK24000"]
    assert not transport.closed  # stop happens before close


@pytest.mark.asyncio
async def test_run_reconnects_after_close() -> None:
    transport = FakeTransport()
    client = _client(transport)
    calls = 0

    async def task_runner():
        nonlocal calls
        await client.run()

    async def stopper():
        for _ in range(2000):
            await asyncio.sleep(0.01)
            if transport.connect_calls >= 2:
                client.stop()
                return

    await asyncio.wait_for(asyncio.gather(task_runner(), stopper()), timeout=10)
    assert transport.connect_calls >= 2
    assert client.reconnect_count >= 1


@pytest.mark.asyncio
async def test_watchdog_expired_detects_stale() -> None:
    import time

    transport = FakeTransport()
    client = _client(transport)
    client._connected = True
    client._last_tick_time = 0.0
    assert await client._watchdog_expired() is True

    client._last_tick_time = time.monotonic()
    assert await client._watchdog_expired() is False


@pytest.mark.asyncio
async def test_backfill_callback_invoked_on_connect() -> None:
    transport = FakeTransport()
    client = _client(transport)
    backfill_called = asyncio.Event()
    ticks = [{"type": "spot", "price": 1}]

    async def backfill():
        backfill_called.set()

    def handler(raw):
        client.stop()

    client.set_tick_handler(handler)
    client.set_backfill_callback(backfill)
    transport.messages = ticks
    await client.run()
    assert backfill_called.is_set()
