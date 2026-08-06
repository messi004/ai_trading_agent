"""Unit tests for tick validation (Phase 1)."""

from __future__ import annotations

from core.tick_validator import Tick, TickError, TickValidator, normalize_tick

NOW = 1_000_000.0


def _raw(**overrides) -> dict:
    base = {"type": "spot", "symbol": "NIFTY", "price": 24005.5, "volume": 100, "ts_epoch": NOW}
    base.update(overrides)
    return base


def test_normalize_spot_tick() -> None:
    t = normalize_tick(_raw())
    assert t.type == "spot"
    assert t.price == 24005.5
    assert t.volume == 100
    assert t.ts_epoch == NOW


def test_normalize_oi_tick() -> None:
    t = normalize_tick(
        {
            "type": "oi",
            "symbol": "NIFTY",
            "strike": 24000,
            "option_type": "CALL",
            "oi": 123_456,
            "ts_epoch": NOW,
        }
    )
    assert t.option_type == "CALL"
    assert t.oi == 123_456


def test_normalize_rejects_bad_type() -> None:
    try:
        normalize_tick(_raw(type="order"))
    except TickError:
        pass
    else:
        raise AssertionError("expected TickError")


def test_normalize_rejects_non_numeric_price() -> None:
    try:
        normalize_tick(_raw(price="abc"))
    except TickError:
        pass
    else:
        raise AssertionError("expected TickError")


def test_validator_accepts_fresh_tick() -> None:
    v = TickValidator()
    tick = v.validate(_raw(), now=NOW)
    assert tick is not None
    assert v.stats.accepted == 1


def test_validator_drops_stale_tick() -> None:
    v = TickValidator()
    assert v.validate(_raw(ts_epoch=NOW - 100), now=NOW) is None
    assert v.stats.dropped_stale == 1


def test_validator_drops_duplicate() -> None:
    v = TickValidator()
    assert v.validate(_raw(), now=NOW) is not None
    assert v.validate(_raw(), now=NOW + 1) is None
    assert v.stats.dropped_duplicate == 1


def test_validator_drops_out_of_order() -> None:
    v = TickValidator()
    assert v.validate(_raw(), now=NOW) is not None
    assert v.validate(_raw(ts_epoch=NOW + 100), now=NOW + 100) is None
    assert v.stats.dropped_out_of_order == 1


def test_validator_drops_malformed() -> None:
    v = TickValidator()
    assert v.validate(_raw(price="x"), now=NOW) is None
    assert v.stats.dropped_malformed == 1


def test_signature_ignores_ts() -> None:
    a = Tick(type="spot", symbol="NIFTY", ts_epoch=1, price=10.0)
    b = Tick(type="spot", symbol="NIFTY", ts_epoch=2, price=10.0)
    assert a.signature == b.signature
