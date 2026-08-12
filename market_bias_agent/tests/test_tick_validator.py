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
            "price": 95.5,
            "ts_epoch": NOW,
        }
    )
    assert t.option_type == "CALL"
    assert t.oi == 123_456
    assert t.price == 95.5  # live option premium


def test_normalize_oi_tick_defaults_premium_to_zero() -> None:
    t = normalize_tick(
        {
            "type": "oi",
            "symbol": "NIFTY",
            "strike": 24000,
            "option_type": "PUT",
            "oi": 123_456,
            "ts_epoch": NOW,
        }
    )
    assert t.price == 0.0


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


def _oi_raw(**overrides) -> dict:
    base = {
        "type": "oi",
        "symbol": "NIFTY",
        "strike": 24000,
        "option_type": "CALL",
        "oi": 123_456,
        "price": 95.5,
        "ts_epoch": NOW,
    }
    base.update(overrides)
    return base


def test_oi_tick_accepts_stale_trade_but_fresh_oi() -> None:
    """OI `ltt` is the last TRADE time; an illiquid strike trades rarely yet
    pushes current OI regularly. A minutes-old trade timestamp must not drop it."""
    v = TickValidator()
    tick = v.validate(_oi_raw(ts_epoch=NOW - 600), now=NOW)
    assert tick is not None
    assert v.stats.accepted == 1


def test_oi_tick_drops_truly_stale_oi() -> None:
    """Beyond the OI window the data is genuinely stale and must be dropped."""
    v = TickValidator()
    assert v.validate(_oi_raw(ts_epoch=NOW - 7200), now=NOW) is None
    assert v.stats.dropped_stale == 1


def test_oi_tick_accepts_sporadic_forward_gap() -> None:
    """Illiquid strikes push OI a few minutes apart; a forward gap up to the
    OI skew window is normal, not a clock jump."""
    v = TickValidator()
    assert v.validate(_oi_raw(), now=NOW) is not None
    assert v.validate(_oi_raw(ts_epoch=NOW + 120, oi=200_000), now=NOW + 120) is not None
    assert v.stats.dropped_out_of_order == 0
    assert v.stats.accepted == 2


def test_oi_ticks_different_strikes_do_not_compare() -> None:
    """All OI ticks share symbol "NIFTY"; the strike/option_type must separate
    them, or one illiquid strike's jumpy `ltt` rejects another's healthy data."""
    v = TickValidator()
    assert v.validate(_oi_raw(strike=24000, ts_epoch=NOW - 600), now=NOW) is not None
    assert (
        v.validate(_oi_raw(strike=24050, ts_epoch=NOW + 240, oi=200_000), now=NOW + 240)
        is not None
    )
    assert v.stats.dropped_out_of_order == 0
    assert v.stats.accepted == 2


def test_spot_keeps_strict_checks() -> None:
    """Spot prices only change on trade, so strict staleness/skew remain."""
    v = TickValidator()
    assert v.validate(_raw(ts_epoch=NOW - 100), now=NOW) is None
    assert v.stats.dropped_stale == 1
    assert v.validate(_raw(), now=NOW) is not None
    assert v.validate(_raw(ts_epoch=NOW + 100, price=24100), now=NOW + 100) is None
    assert v.stats.dropped_out_of_order == 1


def test_signature_ignores_ts() -> None:
    a = Tick(type="spot", symbol="NIFTY", ts_epoch=1, price=10.0)
    b = Tick(type="spot", symbol="NIFTY", ts_epoch=2, price=10.0)
    assert a.signature == b.signature
