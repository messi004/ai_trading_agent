"""Participant-wise Open Interest provider + structural bias (PRD Module 6).

Fetches the real daily NSE participant-wise OI report (FII / DII / Pro /
Client) for index futures and index options through the NiftyTrader public
web API. Computes the next-day structural bias exactly as the PRD describes
(e.g. "FIIs written Calls + Retailers Heavy Long = Sell on Rise").

The provider raises when the upstream API is unreachable or returns no data,
so EOD analysis never silently reports on fabricated numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from core.logger import get_logger

log = get_logger(__name__)

_NIFTYTRADER_BASE = "https://webapi.niftytrader.in/webapi"
_PARTICIPANT_OI_PATH = "Resource/participant-wise-oi-chart-data"
_REFERER = "https://www.niftytrader.in/participant-wise-oi"
_TIMEOUT_SECONDS = 20.0


class ParticipantOIError(RuntimeError):
    """Raised when the upstream participant-OI source is unavailable."""


@dataclass(frozen=True)
class ParticipantPosition:
    """Long/short open-interest (contracts) for one participant cohort."""

    client_type: str
    future_index_long: float
    future_index_short: float
    option_index_call_long: float
    option_index_call_short: float
    option_index_put_long: float
    option_index_put_short: float
    date: str = ""
    nifty50: float = 0.0

    @property
    def future_index_net(self) -> float:
        return self.future_index_long - self.future_index_short

    @property
    def call_short(self) -> float:
        """Written (sold) calls — supply overhead for the index."""
        return self.option_index_call_short

    @property
    def put_short(self) -> float:
        """Written (sold) puts — demand/floor for the index."""
        return self.option_index_put_short

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_type": self.client_type,
            "date": self.date,
            "nifty50": self.nifty50,
            "future_index_long": self.future_index_long,
            "future_index_short": self.future_index_short,
            "future_index_net": self.future_index_net,
            "option_index_call_long": self.option_index_call_long,
            "option_index_call_short": self.option_index_call_short,
            "option_index_put_long": self.option_index_put_long,
            "option_index_put_short": self.option_index_put_short,
        }


class NiftyTraderParticipantOIProvider:
    """Real participant-wise OI via the NiftyTrader public web API."""

    def __init__(
        self, client: httpx.Client | None = None, base_url: str = _NIFTYTRADER_BASE
    ) -> None:
        self._base_url = base_url
        self._client = client

    def _get(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=_TIMEOUT_SECONDS)
        return self._client

    def fetch_latest(self) -> list[ParticipantPosition]:
        """Fetch the latest EOD participant-OI snapshot (all cohorts).

        Raises :class:`ParticipantOIError` when the API is unreachable or the
        response contains no usable rows — never returns fabricated data.
        """
        url = f"{self._base_url}/{_PARTICIPANT_OI_PATH}"
        try:
            resp = self._get().get(url, headers={"Referer": _REFERER, "Origin": _REFERER})
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as exc:
            raise ParticipantOIError(f"participant-OI API unreachable: {exc}") from exc
        except ValueError as exc:
            raise ParticipantOIError(f"participant-OI API returned invalid JSON: {exc}") from exc

        rows = ((payload or {}).get("resultData") or {}).get("data") or []
        if not rows:
            raise ParticipantOIError("participant-OI API returned no rows")

        positions: list[ParticipantPosition] = []
        for row in rows:
            client_type = str(row.get("client_type") or "").strip()
            if not client_type or client_type.upper() == "TOTAL":
                continue
            positions.append(
                ParticipantPosition(
                    client_type=client_type,
                    future_index_long=_to_float(row.get("future_index_long")),
                    future_index_short=_to_float(row.get("future_index_short")),
                    option_index_call_long=_to_float(row.get("option_index_call_long")),
                    option_index_call_short=_to_float(row.get("option_index_call_short")),
                    option_index_put_long=_to_float(row.get("option_index_put_long")),
                    option_index_put_short=_to_float(row.get("option_index_put_short")),
                    date=str(row.get("date") or ""),
                    nifty50=_to_float(row.get("nifty50")),
                )
            )
        if not positions:
            raise ParticipantOIError("participant-OI API returned no participant cohorts")
        log.info(
            "participant_oi_fetched",
            extra={"cohorts": [p.client_type for p in positions], "count": len(positions)},
        )
        return positions


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _cohort(positions: list[ParticipantPosition], name: str) -> ParticipantPosition | None:
    return next((p for p in positions if p.client_type.upper() == name.upper()), None)


def compute_structural_bias(positions: list[ParticipantPosition]) -> dict[str, Any]:
    """Derive the next-day structural bias from participant positioning.

    Heuristics (PRD Module 6):
      * FII net index-futures < 0  -> institutional selling pressure.
      * FII/Pro writing calls heavily -> resistance overhead.
      * Retail (Client) heavy long against FII/Pro short -> "sell on rise".
      * FII/Pro writing puts heavily -> floor/support beneath the market.

    Returns a dict with per-cohort positioning and an overall bias string.
    """
    fii = _cohort(positions, "FII")
    pro = _cohort(positions, "PRO")
    client = _cohort(positions, "CLIENT")

    rows: dict[str, Any] = {}
    for cohort in (fii, pro, client):
        if cohort is not None:
            rows[cohort.client_type.upper()] = cohort.to_dict()

    fii_fut_net = fii.future_index_net if fii else 0.0
    fii_call_short = fii.call_short if fii else 0.0
    fii_put_short = fii.put_short if fii else 0.0
    pro_call_short = pro.call_short if pro else 0.0
    pro_put_short = pro.put_short if pro else 0.0
    client_fut_net = client.future_index_net if client else 0.0

    signals: list[str] = []
    if fii_fut_net < 0:
        signals.append("FII net short index futures")
    elif fii_fut_net > 0:
        signals.append("FII net long index futures")

    if fii is not None and fii_call_short > fii_put_short:
        signals.append("FII written more calls than puts (overhead resistance)")
    elif fii is not None and fii_put_short > fii_call_short:
        signals.append("FII written more puts than calls (floor support)")

    if pro is not None and pro_call_short > pro_put_short:
        signals.append("Pro writing calls (resistance)")
    elif pro is not None and pro_put_short > pro_call_short:
        signals.append("Pro writing puts (support)")

    bearish = fii_fut_net < 0 or (fii_call_short > fii_put_short)
    if client_fut_net > 0 and bearish:
        signals.append("Retail heavy long vs institutional short (Sell on Rise)")

    if bearish:
        bias = "BEARISH"
    elif fii_fut_net > 0 or (fii is not None and fii_put_short > fii_call_short):
        bias = "BULLISH"
    else:
        bias = "NEUTRAL"

    return {
        "bias": bias,
        "signals": signals,
        "participants": rows,
        "fii_futures_net": fii_fut_net,
    }
