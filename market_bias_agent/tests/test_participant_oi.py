"""Unit tests for the real participant-OI provider + structural bias."""

from __future__ import annotations

import httpx
import pytest

from core.participant_oi import (
    NiftyTraderParticipantOIProvider,
    ParticipantOIError,
    ParticipantPosition,
    compute_structural_bias,
)


def _position(
    client_type: str,
    *,
    fut_long: float = 0.0,
    fut_short: float = 0.0,
    call_long: float = 0.0,
    call_short: float = 0.0,
    put_long: float = 0.0,
    put_short: float = 0.0,
) -> ParticipantPosition:
    return ParticipantPosition(
        client_type=client_type,
        future_index_long=fut_long,
        future_index_short=fut_short,
        option_index_call_long=call_long,
        option_index_call_short=call_short,
        option_index_put_long=put_long,
        option_index_put_short=put_short,
        date="2026-08-06",
        nifty50=24636.0,
    )


class _MockTransport:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=self._payload)


def test_fetch_latest_parses_real_rows() -> None:
    payload = {
        "result": 1,
        "resultMessage": "Success",
        "resultData": {
            "data": [
                {
                    "client_type": "Client",
                    "future_index_long": 228874.0,
                    "future_index_short": 62043.0,
                    "option_index_call_long": 2916605.0,
                    "option_index_call_short": 2897584.0,
                    "option_index_put_long": 3534672.0,
                    "option_index_put_short": 4141090.0,
                    "date": "2026-08-07",
                    "nifty50": 24636.0,
                },
                {
                    "client_type": "FII",
                    "future_index_long": 32009.0,
                    "future_index_short": 287122.0,
                    "option_index_call_long": 507535.0,
                    "option_index_call_short": 683709.0,
                    "option_index_put_long": 1075158.0,
                    "option_index_put_short": 620999.0,
                    "date": "2026-08-07",
                    "nifty50": 24636.0,
                },
                {"client_type": "TOTAL", "future_index_long": 1, "future_index_short": 1},
            ]
        },
    }
    client = httpx.Client(transport=_MockTransport(payload))
    provider = NiftyTraderParticipantOIProvider(client=client)
    positions = provider.fetch_latest()
    assert len(positions) == 2  # TOTAL filtered out
    fii = next(p for p in positions if p.client_type == "FII")
    assert fii.future_index_net == pytest.approx(32009.0 - 287122.0)


def test_fetch_latest_raises_on_no_rows() -> None:
    client = httpx.Client(transport=_MockTransport({"result": 0, "resultData": None}))
    provider = NiftyTraderParticipantOIProvider(client=client)
    with pytest.raises(ParticipantOIError):
        provider.fetch_latest()


def test_fetch_latest_raises_on_http_error() -> None:
    class _ErrTransport:
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

    provider = NiftyTraderParticipantOIProvider(client=httpx.Client(transport=_ErrTransport()))
    with pytest.raises(ParticipantOIError):
        provider.fetch_latest()


def test_structural_bias_bearish_when_fii_short_and_call_writing() -> None:
    positions = [
        _position("FII", fut_short=250000.0, call_short=600000.0, put_short=400000.0),
        _position("PRO", call_short=800000.0, put_short=600000.0),
        _position("CLIENT", fut_long=200000.0),
    ]
    result = compute_structural_bias(positions)
    assert result["bias"] == "BEARISH"
    assert any("Retail heavy long" in s for s in result["signals"])


def test_structural_bias_bullish_when_fii_put_writing() -> None:
    positions = [
        _position("FII", fut_long=120000.0, call_short=200000.0, put_short=800000.0),
        _position("CLIENT", fut_short=150000.0),
    ]
    result = compute_structural_bias(positions)
    assert result["bias"] == "BULLISH"


def test_structural_bias_neutral_without_positioning() -> None:
    positions = [_position("FII", call_short=100.0, put_short=100.0)]
    result = compute_structural_bias(positions)
    assert result["bias"] == "NEUTRAL"
