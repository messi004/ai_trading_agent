"""Live feed component factory (Phase 1).

Builds transport / snapshot / strikes providers. The stubs keep the app
bootable and testable without ICICI credentials; swap in the real Breeze
implementations (marked TODO) once credentials are configured.
"""

from __future__ import annotations

import asyncio
from typing import Any

from config.settings import Settings
from core.logger import get_logger
from core.strikes_manager import StrikesProvider
from core.websocket_client import WsTransport

log = get_logger(__name__)


class StubWsTransport:
    """No-op transport: logs and idles so the pipeline can run offline."""

    async def connect(self, subscriptions: list[str]) -> None:
        log.warning("stub_transport_connect", extra={"subscriptions": len(subscriptions)})

    async def receive(self) -> dict[str, Any] | None:
        await asyncio.sleep(3600)
        return None

    async def close(self) -> None:
        log.info("stub_transport_closed")


class StubSnapshotProvider:
    def fetch_recent_snapshot(self, symbol: str, strikes: list[int], lookback_seconds: int) -> dict:
        log.warning("stub_snapshot_fetch", extra={"symbol": symbol})
        return {"spot_ticks": [], "oi_samples": []}


class StubStrikesProvider:
    def fetch_expiry_and_strikes(self, symbol: str) -> tuple[str, list[int]]:
        log.warning("stub_strikes_fetch", extra={"symbol": symbol})
        return ("2000-01-01", [])


def build_transport(settings: Settings) -> WsTransport:
    # TODO(Phase 1): return a BreezeWebSocketTransport wrapping breeze_connect
    # when credentials are present; otherwise keep the stub.
    return StubWsTransport()


def build_snapshot_provider(settings: Settings):
    # TODO(Phase 1): real Breeze REST snapshot fetch.
    return StubSnapshotProvider()


def build_strikes_provider(settings: Settings) -> StrikesProvider:
    # TODO(Phase 1): real Breeze option chain REST fetch.
    return StubStrikesProvider()
