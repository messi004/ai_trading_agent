"""Unit tests for the real Breeze history provider used by ingest_history."""

from __future__ import annotations

import types

import pytest

from core.breeze_session import BreezeSessionError
from scripts.ingest_history import BreezeHistoryProvider


class FakeClient:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.called: list[dict] = []

    def get_historical_data(self, **kwargs) -> dict:
        self.called.append(kwargs)
        return {"Success": self._rows}


class FakeSession:
    def __init__(self, client: FakeClient | None, token: bool = True, creds: bool = False) -> None:
        self._client = client
        self._token = token
        self._creds = creds

    @property
    def has_token(self) -> bool:
        return self._token

    @property
    def has_credentials(self) -> bool:
        return self._creds

    def get_client(self):
        if self._client is None:
            raise BreezeSessionError("no client")
        return self._client


def _settings() -> types.SimpleNamespace:
    return types.SimpleNamespace(nifty_symbol="NIFTY")


def test_fetch_candles_maps_real_rows() -> None:
    client = FakeClient(
        [
            {
                "datetime": "2026-01-05 09:15:00",
                "open": 24000.0,
                "high": 24005.0,
                "low": 23998.0,
                "close": 24003.0,
                "volume": 1500,
            },
            {
                "datetime": "2026-01-05 09:16:00",
                "open": 24003.0,
                "high": 24010.0,
                "low": 24002.0,
                "close": 24009.0,
                "volume": 900,
            },
        ]
    )
    provider = BreezeHistoryProvider(_settings(), FakeSession(client))  # type: ignore[arg-type]
    candles = provider.fetch_candles("NIFTY", "2026-01-05", "2026-01-05")
    assert len(candles) == 2
    assert candles[0].open == 24000.0
    assert candles[0].close == 24003.0
    assert candles[0].high == 24005.0
    assert candles[1].ts_epoch > candles[0].ts_epoch
    assert client.called[0]["exchange_code"] == "NSE"
    assert client.called[0]["interval"] == "1minute"


def test_fetch_skips_malformed_rows() -> None:
    client = FakeClient(
        [
            {"datetime": "2026-01-05 09:15:00", "close": 0},  # zero close -> dropped
            {"datetime": "bad-date", "close": 100.0},  # unparsable ts -> dropped
            {"datetime": "2026-01-05 09:16:00", "open": 1, "high": 2, "low": 1, "close": 1.5},
        ]
    )
    provider = BreezeHistoryProvider(_settings(), FakeSession(client))  # type: ignore[arg-type]
    candles = provider.fetch_candles("NIFTY", "2026-01-05", "2026-01-05")
    assert len(candles) == 1


def test_raises_without_live_session() -> None:
    provider = BreezeHistoryProvider(
        _settings(), FakeSession(None, token=False, creds=False)  # type: ignore[arg-type]
    )
    with pytest.raises(BreezeSessionError):
        provider.fetch_candles("NIFTY", "2026-01-05", "2026-01-05")


def test_fetch_oi_series_aggregates_real_rows() -> None:
    """OI series sums real per-strike open_interest across strikes per minute."""

    def make_client() -> FakeClient:
        rows = [
            {"datetime": "2026-01-05 09:15:00", "open_interest": 250000},
            {"datetime": "2026-01-05 09:16:00", "open_interest": 260000},
        ]
        return FakeClient(rows)

    client = make_client()
    provider = BreezeHistoryProvider(_settings(), FakeSession(client))  # type: ignore[arg-type]
    series = provider.fetch_oi_series(
        "NIFTY", [24000, 24100], "2026-01-08", "2026-01-05", "2026-01-05"
    )
    assert series, "expected OI rows"
    # two strikes x (call+put) = 4 history calls, each returning 2 rows
    assert len(client.called) == 4
    assert all(c["exchange_code"] == "NFO" for c in client.called)
    assert client.called[0]["option_type"] == "call"
    # same timestamps summed across the 4 calls -> each ts appears once
    timestamps = {row[0] for row in series}
    assert len(timestamps) == 2
    # totals reflect all four (call/put x 2 strikes) contributions
    expected = 250000 * 4
    assert series[0][1] + series[0][2] == expected
