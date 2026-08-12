"""Real ICICI Breeze WebSocket transport (replaces the Phase 1 stub).

Wraps ``breeze_connect.BreezeConnect`` behind the :class:`WsTransport`
protocol used by :class:`BreezeWebSocketClient`, so ticks arrive as dicts
matching the pipeline's canonical shape::

    {"type": "spot", "symbol": "NIFTY", "price": ..., "volume": ..., "ts_epoch": ...}
    {"type": "oi", "symbol": "NIFTY", "strike": 24000, "option_type": "CALL",
     "oi": ..., "ts_epoch": ...}
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

from config.settings import Settings
from core.breeze_session import BreezeSessionManager
from core.logger import get_logger
from core.strikes_manager import StrikesProvider

log = get_logger(__name__)

# Breeze `ltt` is `datetime.fromtimestamp(...).strftime('%c')`; on Linux/C.UTF-8
# that is "Wed Feb 12 12:12:55 2025".
_LTT_FORMAT = "%a %b %d %H:%M:%S %Y"
_SPOT_EXCHANGE = "NSE"
_OPTIONS_EXCHANGE = "NFO"
_RIGHTS = (("call", "CALL"), ("put", "PUT"))


def _parse_ltt(ltt: Any) -> float:
    """Parse Breeze's strftime('%c') timestamp back to epoch seconds."""
    if isinstance(ltt, int | float):
        return float(ltt)
    if isinstance(ltt, str) and ltt:
        try:
            return datetime.strptime(ltt, _LTT_FORMAT).timestamp()
        except ValueError:
            try:
                return datetime.strptime(ltt, "%c").timestamp()  # locale fallback
            except ValueError:
                pass
    return time.time()


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class BreezeWsTransport:
    """Real transport: connects Breeze's websocket, maps ticks to pipeline dicts.

    Subscriptions use the same strings as the stub path:
      * ``settings.nifty_symbol`` (e.g. "NIFTY") -> NSE index spot
      * ``"STK<strike>"`` (e.g. "STK24000") -> NFO options call+put at strike
    """

    def __init__(
        self,
        settings: Settings,
        session: BreezeSessionManager,
        strikes_provider: StrikesProvider | None = None,
    ) -> None:
        self._settings = settings
        self._session = session
        self._strikes_provider = strikes_provider
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Any = None
        self._token_meta: dict[str, dict[str, Any]] = {}
        self._closed = False

    # ------------------------------------------------------------------
    # WsTransport protocol
    # ------------------------------------------------------------------
    async def connect(self, subscriptions: list[str]) -> None:
        self._closed = False
        self._loop = asyncio.get_running_loop()
        if self._client is None:
            self._client = self._session.get_client()
        self._client.on_ticks = self._on_ticks
        self._reset_sdk_handler(self._client)
        try:
            self._client.ws_connect()
        except Exception:
            # SDK ws_connect() assigns sio_handler *before* connecting; a
            # failed connect leaves a half-initialised handler in place, and a
            # later ws_connect() then sees `if not self.sio_handler` and never
            # reconnects -> every retry dies with "is not a connected
            # namespace.". Clear it so the next attempt starts fresh.
            self._reset_sdk_handler(self._client)
            raise
        self._drain_queue()
        expiry = self._active_expiry()
        log.warning(
            "breeze_transport_connecting",
            extra={"subscriptions": len(subscriptions), "expiry": expiry or "unknown"},
        )
        for sub in subscriptions:
            self._subscribe(sub, expiry)
        log.info("breeze_transport_connected", extra={"subscriptions": len(self._token_meta)})

    def _drain_queue(self) -> None:
        """Drop stale items (esp. the close() None sentinel) from a prior cycle.

        Otherwise the first receive() after reconnect returns the leftover None
        and the receive loop ends immediately, causing a spurious reconnect.
        """
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - racing consumer
                break

    async def receive(self) -> dict[str, Any] | None:
        if self._closed and self._queue.empty():
            return None
        item = await self._queue.get()
        if item is None:
            return None
        return item

    async def close(self) -> None:
        """Disconnect and stop feeding ticks. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.ws_disconnect()
            except Exception as exc:  # noqa: BLE001 - SDK disconnect is best-effort
                log.warning("breeze_transport_disconnect_error", extra={"error": str(exc)})
            self._reset_sdk_handler(client)
        self._queue.put_nowait(None)
        log.info("breeze_transport_closed")

    @staticmethod
    def _reset_sdk_handler(client: Any) -> None:
        """Force a fresh SDK socket on the next ws_connect().

        SDK bug: ws_disconnect() only emits a "disconnect" event; it neither
        closes the socket nor clears sio_handler. Worse, ws_connect() assigns
        sio_handler *before* connecting, so a failed connect leaves a
        half-initialised handler that makes every later ws_connect() a silent
        no-op (emits then fail with "/ is not a connected namespace."). Reset
        best-effort so the next connect() always builds a brand-new socket.
        """
        try:
            handler = getattr(client, "sio_handler", None)
            if handler is not None:
                old_sio = getattr(handler, "sio", None)
                client.sio_handler = None
                if old_sio is not None:
                    old_sio.disconnect()  # stop the stale socket's thread
        except Exception as exc:  # noqa: BLE001 - best-effort reset
            log.warning("breeze_transport_handler_reset_failed", extra={"error": str(exc)})

    # ------------------------------------------------------------------
    # Subscription helpers
    # ------------------------------------------------------------------
    def _active_expiry(self) -> str | None:
        if self._strikes_provider is not None:
            try:
                expiry, _ = self._strikes_provider.fetch_expiry_and_strikes(
                    self._settings.nifty_symbol
                )
                return expiry if expiry and expiry != "2000-01-01" else None
            except Exception as exc:  # noqa: BLE001 - expiry lookup must not kill connect
                log.warning("breeze_expiry_lookup_failed", extra={"error": str(exc)})
        return None

    def _subscribe(self, sub: str, expiry: str | None) -> None:
        client = self._client
        symbol = self._settings.nifty_symbol
        if sub == symbol:
            token = client.get_stock_token_value(
                exchange_code=_SPOT_EXCHANGE,
                stock_code=symbol,
                get_exchange_quotes=True,
                get_market_depth=False,
            )[0]
            self._token_meta[token] = {"kind": "spot", "symbol": symbol}
            client.subscribe_feeds(stock_token=token)
            log.info("breeze_subscribed_spot", extra={"token": token})
            return
        if sub.startswith("STK") and expiry is not None:
            strike = sub[3:]
            for right, option_type in _RIGHTS:
                token = client.get_stock_token_value(
                    exchange_code=_OPTIONS_EXCHANGE,
                    stock_code=symbol,
                    product_type="options",
                    expiry_date=expiry,
                    strike_price=strike,
                    right=right,
                    get_exchange_quotes=True,
                    get_market_depth=False,
                )[0]
                self._token_meta[token] = {
                    "kind": "oi",
                    "symbol": symbol,
                    "strike": int(strike),
                    "option_type": option_type,
                }
                client.subscribe_feeds(stock_token=token)
                log.info(
                    "breeze_subscribed_option",
                    extra={"token": token, "strike": strike, "right": right},
                )
            return
        log.warning(
            "breeze_subscription_skipped",
            extra={"subscription": sub, "has_expiry": expiry is not None},
        )

    # ------------------------------------------------------------------
    # Tick mapping
    # ------------------------------------------------------------------
    def _on_ticks(self, ticks: Any) -> None:
        """Breeze callback (runs on the socketio thread)."""
        if self._closed or self._loop is None:
            return
        items = ticks if isinstance(ticks, list) else [ticks]
        for tick in items:
            if not isinstance(tick, dict):
                continue
            symbol = tick.get("symbol")
            if not isinstance(symbol, str):
                continue
            meta = self._token_meta.get(symbol)
            if meta is None:
                continue  # quote for something we didn't subscribe to, or order message
            converted = self._convert(tick, meta)
            if converted is not None:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, converted)

    def _convert(self, tick: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any] | None:
        ts_epoch = _parse_ltt(tick.get("ltt"))
        if meta["kind"] == "spot":
            last = _to_float(tick.get("last"))
            if last <= 0:
                return None
            return {
                "type": "spot",
                "symbol": meta["symbol"],
                "ts_epoch": ts_epoch,
                "price": last,
                "volume": _to_float(tick.get("ttq") or tick.get("ltq")),
            }
        # OI tick (also carries the live option premium for paper-trade PnL)
        return {
            "type": "oi",
            "symbol": meta["symbol"],
            "ts_epoch": ts_epoch,
            "strike": meta["strike"],
            "option_type": meta["option_type"],
            "oi": _to_float(tick.get("OI")),
            "price": _to_float(tick.get("last")),
        }


def build_real_transport(
    settings: Settings,
    session: BreezeSessionManager,
    strikes_provider: StrikesProvider | None = None,
) -> BreezeWsTransport:
    """Factory for the real Breeze transport used by feed_factory."""
    return BreezeWsTransport(settings, session, strikes_provider=strikes_provider)
