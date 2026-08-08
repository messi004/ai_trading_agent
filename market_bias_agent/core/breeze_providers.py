"""Real Breeze REST data providers (strikes + snapshot) replacing the stubs.

Both wrap a :class:`BreezeSessionManager`-owned ``BreezeConnect`` client:

* :class:`BreezeStrikesProvider` derives the active expiry + strike chain
  from the client's NFO scrip table (``stock_script_dict_list[4]``), which
  ``generate_session`` already downloads. Contract keys look like
  ``OPT-NIFTY-2026-08-13-24000-CE``.
* :class:`BreezeSnapshotProvider` fetches recent bars / OI via the Breeze
  REST historical-data endpoint to warm Redis buffers after (re)connect.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

from config.constants import STRIKES_RANGE_AROUND_ATM
from config.settings import Settings
from core.breeze_session import BreezeSessionManager
from core.logger import get_logger

log = get_logger(__name__)

_NFO_INDEX = 4  # stock_script_dict_list slot for NFO contracts
_OPTION_PREFIX = "OPT-"
_CALL_SUFFIX = "-CE"
_PUT_SUFFIX = "-PE"
_LOOKBACK_DAYS = 2  # historical fetch window before now
# OPT-<SYMBOL>-<YYYY-MM-DD>-<STRIKE>-<CE|PE>  (symbol may itself contain dashes)
_OPTION_RE = re.compile(
    r"^OPT-(?P<symbol>.+)-(?P<expiry>\d{4}-\d{2}-\d{2})-(?P<strike>\d+)-(?P<right>CE|PE)$"
)


def _parse_contract(contract: str) -> tuple[str, str, int, str] | None:
    """Parse ``OPT-<SYMBOL>-<YYYY-MM-DD>-<STRIKE>-<CE|PE>``.

    Returns (symbol, expiry, strike, option_type) or None if not an option row.
    """
    match = _OPTION_RE.match(contract)
    if match is None:
        return None
    symbol, expiry, strike_text, right = match.groups()
    option_type = "CALL" if right == "CE" else ("PUT" if right == "PE" else "")
    if not option_type:
        return None
    try:
        strike = int(float(strike_text))
        datetime.strptime(expiry, "%Y-%m-%d")
    except ValueError:
        return None
    return symbol, expiry, strike, option_type


def _active_expiry(expiries: list[str], today: str | None = None) -> str:
    """Pick the nearest expiry on/after today; fall back to the latest one."""
    today = today or date.today().isoformat()
    future = sorted(e for e in expiries if e >= today)
    if future:
        return future[0]
    return max(expiries) if expiries else ""


class BreezeStrikesProvider:
    """Fetch active expiry + strike chain from the Breeze scrip table."""

    def __init__(self, settings: Settings, session: BreezeSessionManager) -> None:
        self._settings = settings
        self._session = session

    def _nfo_table(self) -> dict[str, Any]:
        client = self._session.get_client()
        tables = getattr(client, "stock_script_dict_list", None) or []
        if not tables or _NFO_INDEX >= len(tables):
            raise RuntimeError("Breeze stock script table not loaded")
        return tables[_NFO_INDEX] or {}

    def fetch_expiry_and_strikes(self, symbol: str) -> tuple[str, list[int]]:
        """Return (active_expiry 'YYYY-MM-DD', sorted strikes list)."""
        symbol = symbol.upper()
        by_expiry: dict[str, set[int]] = {}
        for contract in self._nfo_table():
            parsed = _parse_contract(contract)
            if parsed is None or parsed[0] != symbol:
                continue
            _sym, expiry, strike, _option_type = parsed
            by_expiry.setdefault(expiry, set()).add(strike)

        if not by_expiry:
            log.warning("breeze_strikes_none_found", extra={"symbol": symbol})
            return "", []

        expiry = _active_expiry(list(by_expiry))
        strikes = sorted(by_expiry.get(expiry, ()))
        log.info(
            "breeze_strikes_synced",
            extra={"symbol": symbol, "expiry": expiry, "strikes": len(strikes)},
        )
        return expiry, strikes


class BreezeSnapshotProvider:
    """Fetch recent NIFTY spot bars + per-strike OI to warm Redis buffers."""

    def __init__(self, settings: Settings, session: BreezeSessionManager) -> None:
        self._settings = settings
        self._session = session

    def _client(self) -> Any:
        return self._session.get_client()

    @staticmethod
    def _window(days: int = _LOOKBACK_DAYS) -> tuple[str, str]:
        """Return (from_date, to_date) covering the lookback window."""
        to = datetime.now()
        frm = to - timedelta(days=days)
        return frm.strftime("%Y-%m-%d"), to.strftime("%Y-%m-%d")

    def _spot_ticks(self, symbol: str) -> list[dict]:
        client = self._client()
        frm, to = self._window()
        raw = client.get_historical_data(
            interval="1minute",
            from_date=frm,
            to_date=to,
            stock_code=symbol,
            exchange_code="NSE",
        )
        rows = (raw or {}).get("Success") or []
        ticks = []
        for row in rows:
            close = _to_float(row.get("close"))
            if close <= 0:
                continue
            ts = _parse_iso(row.get("datetime"))
            if ts is None:
                continue
            ticks.append({"price": close, "ts_epoch": ts, "volume": _to_float(row.get("volume"))})
        return ticks

    def _oi_samples(self, symbol: str, strikes: list[int], expiry: str) -> list[dict]:
        if not strikes or not expiry:
            return []
        client = self._client()
        frm, to = self._window()
        samples = []
        for strike in strikes:
            for right, option_type in (("call", "CALL"), ("put", "PUT")):
                raw = client.get_historical_data(
                    interval="1minute",
                    from_date=frm,
                    to_date=to,
                    stock_code=symbol,
                    exchange_code="NFO",
                    product_type="options",
                    expiry_date=expiry,
                    option_type=right,
                    strike_price=str(strike),
                )
                rows = (raw or {}).get("Success") or []
                if not rows:
                    continue
                last = rows[-1]
                samples.append(
                    {
                        "strike": strike,
                        "option_type": option_type,
                        "oi": _to_float(last.get("open_interest")),
                        "ts_epoch": _parse_iso(last.get("datetime")) or 0.0,
                    }
                )
        return samples

    def fetch_recent_snapshot(self, symbol: str, strikes: list[int], lookback_seconds: int) -> dict:
        """Return {"spot_ticks": [...], "oi_samples": [...]} (oldest first).

        OI samples are capped to the strikes around ATM (default
        ``STRIKES_RANGE_AROUND_ATM`` each side) so the 2 calls-per-strike
        historical fetch stays bounded.
        """
        spot_ticks = self._spot_ticks(symbol)
        oi_samples: list[dict] = []
        if strikes and spot_ticks:
            expiry, chain = self._settings_expiry(symbol, strikes)
            window = self._atm_window(chain, spot_ticks[-1].get("price", 0.0))
            oi_samples = self._oi_samples(symbol, window, expiry)
        result = {"spot_ticks": spot_ticks, "oi_samples": oi_samples}
        log.info(
            "breeze_snapshot_fetched",
            extra={"symbol": symbol, "spot": len(spot_ticks), "oi": len(oi_samples)},
        )
        return result

    @staticmethod
    def _atm_window(
        strikes: list[int], spot: float, range_around_atm: int = STRIKES_RANGE_AROUND_ATM
    ) -> list[int]:
        """Strikes within `range_around_atm` of the spot's ATM strike."""
        if spot <= 0:
            return strikes
        atm = int(round(spot / 100.0) * 100)
        lo, hi = atm - range_around_atm * 100, atm + range_around_atm * 100
        return [s for s in strikes if lo <= s <= hi]

    def _settings_expiry(self, symbol: str, strikes: list[int]) -> tuple[str, list[int]]:
        provider = BreezeStrikesProvider(self._settings, self._session)
        try:
            expiry, full_chain = provider.fetch_expiry_and_strikes(symbol)
        except Exception as exc:  # noqa: BLE001 - fall back to configured strikes
            log.warning("breeze_snapshot_expiry_failed", extra={"error": str(exc)})
            return "", strikes
        if strikes:
            full_chain = sorted(set(full_chain) & set(strikes))
        return expiry, full_chain


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_iso(value: Any) -> float | None:
    if not value:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).replace("T", " ").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return None


def build_real_strikes_provider(
    settings: Settings, session: BreezeSessionManager
) -> BreezeStrikesProvider:
    return BreezeStrikesProvider(settings, session)


def build_real_snapshot_provider(
    settings: Settings, session: BreezeSessionManager
) -> BreezeSnapshotProvider:
    return BreezeSnapshotProvider(settings, session)
