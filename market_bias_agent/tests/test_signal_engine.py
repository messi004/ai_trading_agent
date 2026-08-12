"""End-to-end tests for the live signal engine (trigger -> maker -> checker -> paper -> post)."""

from __future__ import annotations

import json
import time

import fakeredis
import pytest

from config.settings import Settings
from core.redis_manager import RedisManager
from core.signal_store import SignalLogStore
from core.tick_pipeline import TickPipeline
from core.tick_validator import TickValidator
from graph.workflow import SignalWorkflow
from memory.memory_service import MemoryService
from memory.vector_store import MemoryVectorStore
from modules.checker_node import CheckerNode
from modules.maker_node import MakerNode
from modules.paper_trader import PaperTrader
from modules.post_analysis import PostAnalysisEngine
from modules.signal_engine import SignalEngine

VALID = {
    "direction": "BULLISH",
    "confidence": 0.8,
    "entry_zone": [23500.0, 23520.0],
    "sl": 4.0,
    "target": 6.0,
    "rationale": "call velocity + pcr healthy",
    "trap_type": "NONE",
}


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send_alert(self, text: str, photo: bytes = b"") -> bool:
        self.sent.append(text)
        return True

    def send_ops(self, text: str) -> bool:
        self.sent.append(text)
        return True


@pytest.fixture()
def env(tmp_path):
    settings = Settings(qdrant_collection="test_traps")  # empty GEMINI key -> offline embedders
    redis_mgr = RedisManager(settings)
    redis_mgr.client = fakeredis.FakeRedis(decode_responses=True)

    maker = MakerNode(settings, llm_call=lambda _p: json.dumps(VALID))
    checker = CheckerNode(settings)
    workflow = SignalWorkflow(settings, maker=maker, checker=checker)

    store = SignalLogStore(str(tmp_path / "signals.db"))
    memory = MemoryService(
        settings,
        store=MemoryVectorStore(collection="test_traps"),
    )
    post = PostAnalysisEngine(settings, store=store, memory=memory)
    trader = PaperTrader(settings)
    telegram = FakeTelegram()
    engine = SignalEngine(
        settings,
        workflow,
        redis_mgr,
        post_analysis=post,
        paper_trader=trader,
        telegram=telegram,
        memory=memory,
        is_market_open=lambda: True,
        feature_throttle_seconds=0.0,
    )
    yield redis_mgr, engine, post, trader, telegram, store
    store.close()


def _prime_oi(redis_mgr: RedisManager, call: float, put: float, strikes: list[int]) -> None:
    redis_mgr.set_strikes(strikes)
    for s in strikes:
        redis_mgr.push_call_oi(s, call)
        redis_mgr.push_put_oi(s, put)


def _spot_tick(price: float, ts: float) -> dict:
    return {"type": "spot", "symbol": "NIFTY", "ts_epoch": ts, "price": price, "volume": 100.0}


def _oi_tick(strike: int, otype: str, oi: float, ts: float, price: float = 0.0) -> dict:
    return {
        "type": "oi",
        "symbol": "NIFTY",
        "ts_epoch": ts,
        "price": price,
        "volume": 0.0,
        "strike": strike,
        "option_type": otype,
        "oi": oi,
    }


def test_engine_approves_and_opens_paper_position(env) -> None:
    redis_mgr, engine, post, trader, telegram, store = env
    pipeline = TickPipeline(engine._settings, redis_mgr, TickValidator(), signal_engine=engine)
    # 2 strikes x 1.25M call + 1.3M put each
    _prime_oi(redis_mgr, 1_250_000, 1_300_000, [23450, 23500])
    now = time.time()
    pipeline.process(_spot_tick(23500.0, now))

    # A large 1m call velocity spike (> 40k) across strikes triggers SCALP.
    pipeline.process(_oi_tick(23450, "CALL", 1_350_000, now))
    pipeline.process(_oi_tick(23500, "CALL", 1_350_000, now))

    assert post.signal_count() >= 1
    assert len(trader.open_positions()) >= 1
    assert telegram.sent  # an alert was sent
    assert store.get_by_status("MONITORING")  # lifecycle moved to monitoring


def test_engine_resolves_target_exit_and_writes_back(env, tmp_path) -> None:
    redis_mgr, engine, post, trader, telegram, store = env
    pipeline = TickPipeline(engine._settings, redis_mgr, TickValidator(), signal_engine=engine)
    _prime_oi(redis_mgr, 1_250_000, 1_300_000, [23500])
    now = time.time()
    pipeline.process(_spot_tick(23500.0, now))
    pipeline.process(_oi_tick(23500, "CALL", 1_400_000, now))

    assert post.signal_count() == 1
    pos = trader.open_positions()[0]
    entry = pos.entry_fill
    # Long: price reaching entry + target = TARGET_HIT (real-time tick)
    pipeline.process(_spot_tick(entry + 10.0, time.time()))

    assert not trader.open_positions()
    closed = trader.closed_positions()
    assert closed and closed[0].exit_reason == "TARGET_HIT"
    assert closed[0].pnl_points > 0
    # ANALYZED -> CLOSED after write-back
    assert any(row["status"] == "CLOSED" for row in store.get_closed())


def test_engine_does_not_open_on_rejected(env) -> None:
    redis_mgr, engine, post, trader, telegram, store = env
    settings = engine._settings
    engine._workflow = SignalWorkflow(
        settings,
        maker=MakerNode(settings, llm_call=lambda _p: json.dumps(VALID)),
        checker=CheckerNode(settings),
    )
    # Force Rule B rejection: BULLISH with PCR < 0.75 (put OI very low).
    _prime_oi(redis_mgr, 1_000_000, 100_000, [23500])
    now = time.time()
    pipeline = TickPipeline(settings, redis_mgr, TickValidator(), signal_engine=engine)
    pipeline.process(_spot_tick(23500.0, now))
    pipeline.process(_oi_tick(23500, "CALL", 1_200_000, now))

    assert post.signal_count() == 0
    assert not trader.open_positions()
    assert not telegram.sent


def test_pipeline_invokes_engine(env) -> None:
    redis_mgr, engine, post, trader, telegram, store = env
    pipeline = TickPipeline(engine._settings, redis_mgr, TickValidator(), signal_engine=engine)
    now = time.time()
    _prime_oi(redis_mgr, 1_250_000, 1_300_000, [23500])
    pipeline.process(_spot_tick(23500.0, now))
    pipeline.process(_oi_tick(23500, "CALL", 1_400_000, now))
    assert post.signal_count() == 1


def test_engine_books_premium_pnl_on_oi_ticks(env) -> None:
    redis_mgr, engine, post, trader, telegram, store = env
    pipeline = TickPipeline(engine._settings, redis_mgr, TickValidator(), signal_engine=engine)
    _prime_oi(redis_mgr, 1_250_000, 1_300_000, [23500])
    now = time.time()

    # Feed live premium BEFORE the signal so entry premium is captured.
    pipeline.process(_oi_tick(23500, "CALL", 1_300_000, now, price=90.0))
    pipeline.process(_spot_tick(23500.0, now))
    pipeline.process(_oi_tick(23500, "CALL", 1_400_000, now, price=95.0))

    pos = trader.open_positions()[0]
    assert pos.strike == 23500
    assert pos.option_type == "CALL"
    assert pos.entry_premium == 95.0

    # Premium rises with the underlying -> target exit books premium profit.
    pipeline.process(_oi_tick(23500, "CALL", 1_400_000, time.time(), price=105.0))
    pipeline.process(_spot_tick(pos.entry_fill + 10.0, time.time()))

    assert not trader.open_positions()
    closed = trader.closed_positions()
    assert closed and closed[0].exit_reason == "TARGET_HIT"
    assert closed[0].pnl_premium_points == pytest.approx(10.0)


def test_structural_bias_feature_from_redis(env) -> None:
    from utils.time_utils import now_ist

    redis_mgr, engine, *_ = env
    redis_mgr.set_eod_bias(
        {
            "bias": "BEARISH",
            "signals": ["FII net short index futures", "FII written more calls than puts"],
            "nifty50": 24636.0,
            "session_date": now_ist().date().isoformat(),
            "computed_at_ist": now_ist().isoformat(),
        }
    )
    features = engine.feature_snapshot()
    assert features["structural_bias"] == "BEARISH"
    assert "FII net short index futures" in features["institutional_signals"]


def test_structural_bias_neutral_when_stale(env) -> None:
    from datetime import timedelta

    from utils.time_utils import now_ist

    redis_mgr, engine, *_ = env
    redis_mgr.set_eod_bias(
        {
            "bias": "BULLISH",
            "signals": ["FII net long index futures"],
            "session_date": (now_ist() - timedelta(days=1)).date().isoformat(),
            "computed_at_ist": now_ist().isoformat(),
        }
    )
    features = engine.feature_snapshot()
    assert features["structural_bias"] == "NEUTRAL"
    assert features["institutional_signals"] == []


def test_structural_bias_neutral_when_missing(env) -> None:
    redis_mgr, engine, *_ = env
    assert redis_mgr.get_eod_bias() is None
    features = engine.feature_snapshot()
    assert features["structural_bias"] == "NEUTRAL"
    assert features["institutional_signals"] == []


def test_premarket_context_features_structured(env) -> None:
    redis_mgr, engine, *_ = env
    redis_mgr.set_pre_market_levels(
        {
            "session_date": "2026-08-12",
            "prev_high": 24200.0,
            "prev_low": 24000.0,
            "prev_close": 24100.0,
            "pivot": 24100.0,
            "r1": 24200.0,
            "r2": 24300.0,
            "s1": 24000.0,
            "s2": 23900.0,
            "psych_resistance": 24200.0,
            "psych_support": 24000.0,
            "levels": [23900.0, 24000.0, 24100.0, 24200.0, 24300.0],
            "max_pain": {"strike": 24100.0, "zone": [24088.0, 24112.0]},
        }
    )
    features = engine.feature_snapshot()
    assert features["premarket_pivot"] == 24100.0
    assert features["premarket_s1"] == 24000.0
    assert features["premarket_r1"] == 24200.0
    assert features["premarket_max_pain"] == 24100.0
    assert features["premarket_max_pain_zone"] == [24088.0, 24112.0]


def test_premarket_context_empty_when_missing(env) -> None:
    redis_mgr, engine, *_ = env
    assert redis_mgr.get_pre_market_levels() is None
    features = engine.feature_snapshot()
    assert "premarket_pivot" not in features
    assert "premarket_max_pain" not in features


def test_premarket_context_includes_oi_walls(env) -> None:
    redis_mgr, engine, *_ = env
    redis_mgr.set_pre_market_levels(
        {
            "pivot": 24100.0,
            "r1": 24200.0,
            "s1": 24000.0,
            "max_pain": {"strike": 24100.0, "zone": [24088.0, 24112.0]},
            "oi_walls": {
                "resistance": [24200.0, 24250.0],
                "support": [24000.0, 23950.0],
                "max_pain": 24100.0,
            },
        }
    )
    features = engine.feature_snapshot()
    assert features["premarket_oi_resistance"] == [24200.0, 24250.0]
    assert features["premarket_oi_support"] == [24000.0, 23950.0]
    assert features["premarket_oi_max_pain"] == 24100.0

    levels = engine._premarket_level_list()
    assert 24200.0 in levels
    assert 23950.0 in levels
    assert 24100.0 in levels  # both max-pain + walls max pain deduped into the list
