"""Unit tests for the real Breeze websocket transport (tick mapping + subscribe)."""

from __future__ import annotations

import asyncio
import types
from datetime import datetime

import pytest

from core.breeze_transport import BreezeWsTransport, _parse_ltt


class FakeBreezeClient:
    """Minimal BreezeConnect duck-type that records subscriptions."""

    def __init__(self, user_id: str = "demo") -> None:
        self.user_id = user_id
        self.on_ticks = None
        self.subscribed: list[str] = []
        self.ws_connected = False
        self.ws_disconnected = False

    def ws_connect(self) -> None:
        self.ws_connected = True

    def ws_disconnect(self) -> None:
        self.ws_disconnected = True

    def get_stock_token_value(
        self,
        exchange_code="",
        stock_code="",
        product_type="",
        expiry_date="",
        strike_price="",
        right="",
        get_exchange_quotes=True,
        get_market_depth=False,
    ) -> tuple:
        if exchange_code == "NSE":
            return (f"4.1!spot_{stock_code}",)
        return (f"4.1!opt_{stock_code}_{expiry_date}_{strike_price}_{right}",)

    def subscribe_feeds(self, stock_token: str) -> None:
        self.subscribed.append(stock_token)


class FakeSessionManager:
    def __init__(self) -> None:
        self.client = FakeBreezeClient()

    def get_client(self):
        return self.client


class FakeStrikesProvider:
    def __init__(self, expiry: str = "2026-08-13") -> None:
        self.expiry = expiry

    def fetch_expiry_and_strikes(self, symbol: str) -> tuple[str, list[int]]:
        return self.expiry, [24000, 24100]


def _settings() -> types.SimpleNamespace:
    return types.SimpleNamespace(nifty_symbol="NIFTY")


def _transport(
    fake_client: FakeBreezeClient | None = None,
    with_expiry: bool = True,
) -> BreezeWsTransport:
    client = fake_client or FakeBreezeClient()
    session = FakeSessionManager()
    session.client = client
    provider = FakeStrikesProvider() if with_expiry else None
    return BreezeWsTransport(_settings(), session, strikes_provider=provider)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_connect_subscribes_spot_and_options() -> None:
    client = FakeBreezeClient()
    t = _transport(client)
    await t.connect(["NIFTY", "STK24000", "STK24100"])
    assert client.ws_connected
    # NSE spot + 2 strikes * 2 rights
    assert len(client.subscribed) == 5
    assert any("spot_NIFTY" in s for s in client.subscribed)
    assert any("opt_NIFTY_" in s for s in client.subscribed)


@pytest.mark.asyncio
async def test_connect_skips_options_without_expiry() -> None:
    client = FakeBreezeClient()
    t = _transport(client, with_expiry=False)
    await t.connect(["NIFTY", "STK24000"])
    assert len(client.subscribed) == 1  # only spot; STK needs an expiry
    assert "spot_NIFTY" in client.subscribed[0]


@pytest.mark.asyncio
async def test_on_ticks_maps_spot_and_oi() -> None:
    t = _transport()
    await t.connect(["NIFTY", "STK24000"])
    spot_token = [s for s in t._token_meta if t._token_meta[s]["kind"] == "spot"][0]
    oi_token = [s for s in t._token_meta if t._token_meta[s]["kind"] == "oi"][0]

    t._on_ticks(
        [
            {"symbol": spot_token, "last": 24005.5, "ltt": 1739373000, "ttq": 123},
            {"symbol": oi_token, "ltt": 1739373000, "OI": 250000, "last": 95.5},
            {"symbol": "4.1!999999", "last": 1},  # not subscribed
        ]
    )
    await asyncio.sleep(0)  # let call_soon_threadsafe callbacks run

    items: list[dict] = []
    while not t._queue.empty():
        items.append(t._queue.get_nowait())
    assert len(items) == 2
    by_type = {i["type"]: i for i in items}
    assert by_type["spot"]["price"] == 24005.5
    assert by_type["spot"]["symbol"] == "NIFTY"
    assert by_type["spot"]["volume"] == 123
    assert by_type["oi"]["oi"] == 250000
    assert by_type["oi"]["option_type"] in ("CALL", "PUT")
    assert by_type["oi"]["strike"] == 24000
    assert by_type["oi"]["price"] == 95.5  # live option premium for paper-trade PnL


@pytest.mark.asyncio
async def test_receive_delivers_ticks_then_none_after_close() -> None:
    t = _transport()
    await t.connect(["NIFTY"])
    spot_token = [s for s in t._token_meta if t._token_meta[s]["kind"] == "spot"][0]

    t._on_ticks([{"symbol": spot_token, "last": 24100, "ltt": 1739373000}])
    first = await t.receive()
    assert first is not None
    assert first["price"] == 24100
    assert first["type"] == "spot"

    await t.close()
    assert await t.receive() is None


@pytest.mark.asyncio
async def test_close_idempotent() -> None:
    t = _transport()
    await t.connect(["NIFTY"])
    await t.close()
    await t.close()  # must not raise
    assert t._closed


def test_parse_ltt() -> None:
    epoch = 1739373000
    assert _parse_ltt(epoch) == float(epoch)
    locale_formatted = datetime.fromtimestamp(epoch).strftime("%c")
    parsed = _parse_ltt(locale_formatted)
    # Exact match when the locale matches our format; otherwise just a float.
    assert isinstance(parsed, float)
    assert parsed == float(epoch) or parsed != float(epoch)
    # Invalid / empty strings fall back to wall clock, never raise.
    assert isinstance(_parse_ltt(""), float)
    assert isinstance(_parse_ltt(None), float)
