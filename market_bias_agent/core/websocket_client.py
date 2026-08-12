"""ICICI Direct Breeze WebSocket client (PRD Module 1 + Phase 1 resilience).

Phase 1 additions:
  * Exponential backoff with jitter on failure.
  * Tick watchdog: reconnect if no tick for N seconds.
  * Auto-resubscribe on every reconnect.
  * Optional REST backfill callback invoked after each (re)connect.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from config.constants import (
    TICK_WATCHDOG_IDLE_SECONDS,
    WS_BACKOFF_BASE_SECONDS,
    WS_BACKOFF_FACTOR,
    WS_BACKOFF_MAX_SECONDS,
)
from config.settings import Settings
from core.logger import get_logger
from utils.time_utils import market_status

log = get_logger(__name__)

TickHandler = Callable[[dict[str, Any]], Any]  # return value ignored
BackfillCallback = Callable[[], Awaitable[None]]
SubscriptionsProvider = Callable[[], list[str]]


@runtime_checkable
class WsTransport(Protocol):
    """Abstract socket transport so tests can inject a fake.

    A real implementation wraps Breeze's websocket: connect with the
    subscription list, receive parsed dicts, return None when closed.
    """

    async def connect(self, subscriptions: list[str]) -> None: ...

    async def receive(self) -> dict[str, Any] | None: ...

    async def close(self) -> None: ...


def backoff_delay(attempt: int) -> float:
    """Exponential backoff with full jitter: U(0, base * factor^attempt)."""
    cap = min(WS_BACKOFF_BASE_SECONDS * (WS_BACKOFF_FACTOR**attempt), WS_BACKOFF_MAX_SECONDS)
    return random.uniform(0, cap)


class BreezeWebSocketClient:
    def __init__(
        self,
        settings: Settings,
        transport: WsTransport,
        subscriptions_provider: SubscriptionsProvider,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._subscriptions = subscriptions_provider
        self._handler: TickHandler | None = None
        self._backfill: BackfillCallback | None = None
        self._status_callback: Any = None
        self._connected = False
        self._last_tick_time = 0.0
        self._reconnect_count = 0
        self._stop = False

    def set_tick_handler(self, handler: TickHandler) -> None:
        self._handler = handler

    def set_backfill_callback(self, callback: BackfillCallback) -> None:
        self._backfill = callback

    def set_status_callback(self, callback: Any) -> None:
        """Optional health hook: called on every connect state change (Phase 6)."""
        self._status_callback = callback

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    def stop(self) -> None:
        self._stop = True

    def _mark_tick(self) -> None:
        self._last_tick_time = time.monotonic()

    def _emit_status(self) -> None:
        if self._status_callback is not None:
            try:
                self._status_callback(self._connected, self._reconnect_count)
            except Exception as exc:  # noqa: BLE001 - health must never break ws
                log.warning("ws_status_callback_failed", extra={"error": str(exc)})

    async def _connect_and_subscribe(self) -> None:
        subs = self._subscriptions()
        log.info("ws_connecting", extra={"subscriptions": len(subs)})
        await self._transport.connect(subs)
        self._connected = True
        self._mark_tick()
        self._emit_status()
        log.info("ws_connected", extra={"subscriptions": len(subs)})
        if self._backfill is not None:
            try:
                await self._backfill()
            except Exception as exc:  # noqa: BLE001
                log.error("backfill_failed", extra={"error": str(exc)})

    async def _watchdog_expired(self, market: str | None = None) -> bool:
        """True if no tick received for the idle window while supposedly open.

        Market-aware like the health watchdog: outside OPEN hours a silent
        socket is expected (no ticks flow pre/post market), so we must not
        tear down a healthy connection just because the feed is quiet.
        `market` overrides the auto-detected PRE/OPEN/POST state (testability).
        """
        if not self._connected:
            return False
        if (market if market is not None else market_status()) != "OPEN":
            return False
        return (time.monotonic() - self._last_tick_time) > TICK_WATCHDOG_IDLE_SECONDS

    async def run(self) -> None:
        """Long-running loop with reconnect + watchdog. Call stop() to exit."""
        attempt = 0
        while not self._stop:
            try:
                await self._connect_and_subscribe()
                attempt = 0
                async for raw in self._receive_loop():
                    if self._handler is not None:
                        # Mark tick on receipt (not after processing) so a slow
                        # signal/LLM step never trips the 10s stale watchdog.
                        self._mark_tick()
                        # Off-loop: the signal engine calls the LLM synchronously;
                        # running it on the event loop blocks /health + Telegram.
                        await asyncio.to_thread(self._handler, raw)
                # stream ended without exception -> unexpected close
                raise ConnectionError("websocket stream closed")
            except asyncio.CancelledError:
                self._connected = False
                self._emit_status()
                raise
            except Exception as exc:  # noqa: BLE001
                self._connected = False
                self._emit_status()
                if self._stop:
                    return
                self._reconnect_count += 1
                delay = backoff_delay(attempt)
                attempt += 1
                log.error(
                    "ws_disconnected",
                    extra={"error": str(exc), "attempt": attempt, "retry_in": round(delay, 2)},
                )
                await asyncio.sleep(delay)

    async def _receive_loop(self) -> Any:
        """Yield parsed messages; return (break) when transport closes."""
        log.warning("debug_receive_loop_enter", extra={"connected": self._connected})
        while not self._stop:
            if await self._watchdog_expired():
                log.warning(
                    "ws_watchdog_stale",
                    extra={"idle_seconds": round(time.monotonic() - self._last_tick_time, 2)},
                )
                await self._transport.close()
                return
            try:
                raw = await asyncio.wait_for(
                    self._transport.receive(), timeout=TICK_WATCHDOG_IDLE_SECONDS
                )
            except asyncio.TimeoutError:
                log.warning("debug_receive_loop_timeout")
                continue
            if raw is None:
                log.warning("debug_receive_loop_none")
                return
            yield raw
