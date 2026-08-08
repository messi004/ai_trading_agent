"""Live feed component factory (Phase 1).

Builds transport / snapshot / strikes providers. The stubs keep the app
bootable and testable without ICICI credentials; swap in the real Breeze
implementations (marked TODO) once credentials are configured.
"""

from __future__ import annotations

import asyncio
from typing import Any

from config.settings import Settings
from core.breeze_session import BreezeSessionManager
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


def build_transport(settings: Settings, session: BreezeSessionManager | None = None) -> WsTransport:
    """Real Breeze transport when a session is configured, else the stub.

    The stub keeps the app bootable and testable offline; the real transport
    is only selected once ICICI session credentials are present.
    """
    from core.breeze_providers import BreezeStrikesProvider
    from core.breeze_transport import BreezeWsTransport

    session = session or BreezeSessionManager(settings)
    if not session.has_token and not session.has_credentials:
        return StubWsTransport()
    strikes_provider = BreezeStrikesProvider(settings, session)
    return BreezeWsTransport(settings, session, strikes_provider=strikes_provider)


def build_snapshot_provider(settings: Settings, session: BreezeSessionManager | None = None):
    """Real Breeze REST snapshot provider, else the offline stub."""
    from core.breeze_providers import BreezeSnapshotProvider

    session = session or BreezeSessionManager(settings)
    if not session.has_token and not session.has_credentials:
        return StubSnapshotProvider()
    return BreezeSnapshotProvider(settings, session)


def build_strikes_provider(
    settings: Settings, session: BreezeSessionManager | None = None
) -> StrikesProvider:
    """Real Breeze option-chain provider, else the offline stub."""
    from core.breeze_providers import BreezeStrikesProvider

    session = session or BreezeSessionManager(settings)
    if not session.has_token and not session.has_credentials:
        return StubStrikesProvider()
    return BreezeStrikesProvider(settings, session)
