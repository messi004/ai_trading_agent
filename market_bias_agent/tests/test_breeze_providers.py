"""Unit tests for the real Breeze REST providers (strikes + snapshot)."""

from __future__ import annotations

import types

import pytest

from core.breeze_providers import (
    BreezeSnapshotProvider,
    BreezeStrikesProvider,
    _active_expiry,
    _parse_contract,
)


class FakeBreezeClient:
    """Duck-typed BreezeConnect returning a canned NFO scrip table."""

    NFO_TABLE = {
        "OPT-NIFTY-2026-08-13-24000-CE": "4.1!1",
        "OPT-NIFTY-2026-08-13-24000-PE": "4.1!2",
        "OPT-NIFTY-2026-08-13-24100-CE": "4.1!3",
        "OPT-NIFTY-2026-08-13-24100-PE": "4.1!4",
        "OPT-NIFTY-2026-09-03-24200-CE": "4.1!5",
        "OPT-NIFTY-2026-09-03-24200-PE": "4.1!6",
        "FUT-NIFTY-2026-08-13": "4.1!7",
        "OPT-BANKNIFTY-2026-08-13-50000-CE": "4.1!8",
    }

    def __init__(self) -> None:
        self.stock_script_dict_list = [{}, {}, {}, {}, dict(self.NFO_TABLE)]
        self.historical_calls: list[dict] = []

    def get_historical_data(self, **kwargs) -> dict:
        self.historical_calls.append(kwargs)
        if kwargs.get("interval") == "1minute" and kwargs.get("exchange_code") == "NSE":
            rows = [
                {"datetime": "2026-08-09 10:00:00", "close": 24000, "volume": 100},
                {"datetime": "2026-08-09 10:01:00", "close": 24010, "volume": 120},
            ]
        else:
            rows = [{"datetime": "2026-08-09 10:01:00", "close": 24010, "open_interest": 250000}]
        return {"Success": rows}


class FakeSessionManager:
    def __init__(self) -> None:
        self.client = FakeBreezeClient()

    def get_client(self):
        return self.client


def _settings() -> types.SimpleNamespace:
    return types.SimpleNamespace(nifty_symbol="NIFTY")


def _strikes_provider(client: FakeBreezeClient | None = None):
    session = FakeSessionManager()
    session.client = client or FakeBreezeClient()
    return BreezeStrikesProvider(_settings(), session)  # type: ignore[arg-type]


def _snapshot_provider(client: FakeBreezeClient | None = None):
    session = FakeSessionManager()
    session.client = client or FakeBreezeClient()
    return BreezeSnapshotProvider(_settings(), session)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "contract,expected",
    [
        ("OPT-NIFTY-2026-08-13-24000-CE", ("NIFTY", "2026-08-13", 24000, "CALL")),
        ("OPT-NIFTY-2026-08-13-24000-PE", ("NIFTY", "2026-08-13", 24000, "PUT")),
        ("OPT-NIFTY-2026-09-03-24200-CE", ("NIFTY", "2026-09-03", 24200, "CALL")),
        ("FUT-NIFTY-2026-08-13", None),  # not an option
        ("OPT-NIFTY-2026-08-13-24000-XX", None),  # unknown right
        ("garbage", None),
    ],
)
def test_parse_contract(contract, expected) -> None:
    assert _parse_contract(contract) == expected


def test_active_expiry_picks_future() -> None:
    assert _active_expiry(["2026-09-03", "2026-08-13"], today="2026-08-09") == "2026-08-13"
    assert _active_expiry(["2026-09-03", "2026-08-13"], today="2026-08-20") == "2026-09-03"
    # all past -> latest
    assert _active_expiry(["2026-07-01", "2026-06-01"], today="2026-08-09") == "2026-07-01"
    assert _active_expiry([]) == ""


def test_strikes_provider_returns_active_expiry_and_chain() -> None:
    provider = _strikes_provider()
    expiry, strikes = provider.fetch_expiry_and_strikes("NIFTY")
    assert expiry == "2026-08-13"
    assert strikes == [24000, 24100]


def test_strikes_provider_excludes_other_symbols_and_futures() -> None:
    provider = _strikes_provider()
    _expiry, strikes = provider.fetch_expiry_and_strikes("BANKNIFTY")
    assert strikes == [50000]
    # futures-only symbol: no options -> empty chain
    provider = _strikes_provider(FakeBreezeClient())
    provider._session.client.stock_script_dict_list[4] = {"FUT-NIFTY-2026-08-13": "4.1!7"}
    _expiry, strikes = provider.fetch_expiry_and_strikes("NIFTY")
    assert strikes == []


def test_snapshot_provider_fetches_spot_and_oi() -> None:
    client = FakeBreezeClient()
    provider = _snapshot_provider(client)
    result = provider.fetch_recent_snapshot("NIFTY", [24000], lookback_seconds=600)
    spot = result["spot_ticks"]
    oi = result["oi_samples"]
    assert len(spot) == 2
    assert spot[0]["price"] == 24000
    assert spot[0]["ts_epoch"] > 0
    assert spot[-1]["price"] == 24010
    assert len(oi) == 2  # [24000] intersects [24000, 24100]: CALL + PUT
    assert {s["option_type"] for s in oi} == {"CALL", "PUT"}
    assert oi[0]["strike"] == 24000
    assert oi[0]["oi"] == 250000
    # historical API called for spot + 1 strike * 2 rights = 3 requests
    assert len(client.historical_calls) == 3


def test_snapshot_provider_handles_empty_strikes() -> None:
    provider = _snapshot_provider()
    result = provider.fetch_recent_snapshot("NIFTY", [], lookback_seconds=600)
    assert len(result["oi_samples"]) == 0
    assert len(result["spot_ticks"]) == 2


def test_atm_window_caps_chain() -> None:
    strikes = list(range(23000, 25100, 100))
    provider = _snapshot_provider()
    # Spot ~24010 -> ATM 24000; range 1 keeps [23900, 24100]
    window = provider._atm_window(strikes, spot=24010, range_around_atm=1)
    assert window == [23900, 24000, 24100]
    # No spot -> full chain untouched
    assert provider._atm_window(strikes, spot=0) == strikes


def test_snapshot_oi_capped_to_atm_window() -> None:
    client = FakeBreezeClient()
    provider = _snapshot_provider(client)
    # Full chain for NIFTY is [24000, 24100]; spot 24010 -> ATM 24000,
    # range_around_atm=20 keeps both, so both strikes are fetched (4 calls).
    result = provider.fetch_recent_snapshot("NIFTY", [24000, 24100], lookback_seconds=600)
    strikes_fetched = {s["strike"] for s in result["oi_samples"]}
    assert strikes_fetched == {24000, 24100}
    assert len(client.historical_calls) == 1 + 2 * 2  # spot + 2 strikes * 2 rights
