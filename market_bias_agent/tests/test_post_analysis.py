"""Phase 8 tests: signal lifecycle store, post-trade analysis, memory write-back."""

import pytest

from config.settings import Settings
from core.signal_store import SignalLogStore, SignalTransitionError
from core.signals import StructuredSignal
from memory.embeddings import FeatureEmbedder
from memory.memory_service import MemoryService
from memory.vector_store import MemoryVectorStore
from modules.post_analysis import (
    PostAnalysisEngine,
    TradeSummary,
    outcome_from_pnl,
)


def make_signal(**overrides) -> StructuredSignal:
    base = dict(
        direction="BULLISH",
        confidence=0.8,
        entry_zone=(23999, 24002),
        sl=4.0,
        target=6.0,
        rationale="test",
        trap_type="BULL_TRAP",
        ts_epoch=1_700_000_000.0,
        trigger_type="SCALP",
        strike=24000,
    )
    base.update(overrides)
    return StructuredSignal(**base)


def make_store(tmp_path) -> SignalLogStore:
    return SignalLogStore(str(tmp_path / "signals.db"))


def make_engine(tmp_path) -> PostAnalysisEngine:
    memory = MemoryService(Settings(), store=MemoryVectorStore(), embedder=FeatureEmbedder())
    store = make_store(tmp_path)
    engine = PostAnalysisEngine(Settings(), store=store, memory=memory)
    return engine


def walk_lifecycle(engine: PostAnalysisEngine, signal_id: str) -> None:
    engine.approve(signal_id)
    engine.monitor(signal_id, entry_fill=24001.0)
    engine.record_exit(
        signal_id,
        exit_price=24010.0,
        exit_reason="TARGET_HIT",
        pnl_points=9.0,
        mfe=10.0,
        mae=1.0,
        ts=1_700_001_000.0,
    )


class TestOutcomeMapping:
    def test_outcome_from_pnl(self) -> None:
        assert outcome_from_pnl(5.0) == "WIN"
        assert outcome_from_pnl(-2.0) == "LOSS"
        assert outcome_from_pnl(0.0) == "BE"


class TestSignalStore:
    def test_register_and_get(self, tmp_path) -> None:
        store = make_store(tmp_path)
        signal = make_signal()
        store.register_signal(signal)
        row = store.get_signal(signal.signal_id)
        assert row is not None
        assert row["status"] == "SIGNAL_GENERATED"
        assert row["direction"] == "BULLISH"
        assert row["trap_type"] == "BULL_TRAP"
        assert store.count() == 1

    def test_lifecycle_transitions(self, tmp_path) -> None:
        store = make_store(tmp_path)
        signal = make_signal()
        store.register_signal(signal)
        store.mark_approved(signal.signal_id)
        store.mark_monitoring(signal.signal_id, entry_fill=24001.0)
        store.mark_exited(
            signal.signal_id,
            exit_price=24010.0,
            exit_reason="TARGET_HIT",
            pnl_points=9.0,
            mfe=10.0,
            mae=1.0,
        )
        store.mark_closed(signal.signal_id)
        row = store.get_signal(signal.signal_id)
        assert row["status"] == "CLOSED"
        assert row["outcome"] == "WIN"

    def test_illegal_transition_rejected(self, tmp_path) -> None:
        store = make_store(tmp_path)
        signal = make_signal()
        store.register_signal(signal)
        with pytest.raises(SignalTransitionError):
            store.mark_exited(
                signal.signal_id,
                exit_price=24010.0,
                exit_reason="TARGET_HIT",
                pnl_points=9.0,
                mfe=10.0,
                mae=1.0,
            )
        with pytest.raises(SignalTransitionError):
            store.mark_closed(signal.signal_id)

    def test_query_by_status(self, tmp_path) -> None:
        store = make_store(tmp_path)
        store.register_signal(make_signal())
        assert len(store.get_by_status("SIGNAL_GENERATED")) == 1
        assert store.get_analysed() == []
        assert store.get_closed() == []


class TestPostAnalysisEngine:
    def test_full_flow_with_writeback(self, tmp_path) -> None:
        engine = make_engine(tmp_path)
        signal = make_signal()
        engine.register_signal(signal)
        walk_lifecycle(engine, signal.signal_id)
        assert engine._store.get_signal(signal.signal_id)["status"] == "ANALYZED"
        assert engine._store.get_signal(signal.signal_id)["outcome"] == "WIN"

        vector_id = engine.close_and_write_back(
            signal.signal_id, features={"spot": 24001, "pcr": 0.95, "volatility": "ACTIVE"}
        )
        assert vector_id is not None
        assert engine._store.get_signal(signal.signal_id)["status"] == "CLOSED"
        assert engine._memory.store.count() == 1

    def test_writeback_requires_analysed(self, tmp_path) -> None:
        engine = make_engine(tmp_path)
        signal = make_signal()
        engine.register_signal(signal)
        assert engine.close_and_write_back(signal.signal_id) is None

    def test_trade_summary_text(self, tmp_path) -> None:
        engine = make_engine(tmp_path)
        signal = make_signal()
        engine.register_signal(signal)
        walk_lifecycle(engine, signal.signal_id)
        duration = (1_700_001_000.0 - signal.ts_epoch) / 60.0
        summary = TradeSummary(
            signal_id=signal.signal_id,
            exit_reason="TARGET_HIT",
            pnl_points=9.0,
            mfe=10.0,
            mae=1.0,
            duration_minutes=duration,
        )
        text = summary.to_text()
        assert "TARGET HIT" in text
        assert "+9.0 pts" in text
        assert "MFE 10.0" in text

    def test_weekly_report_by_trigger(self, tmp_path) -> None:
        engine = make_engine(tmp_path)
        for i in range(3):
            signal = make_signal()
            signal.signal_id = f"sig{i}"
            engine.register_signal(signal)
            walk_lifecycle(engine, signal.signal_id)
            engine.close_and_write_back(signal.signal_id)
        report = engine.weekly_report()
        assert "SCALP" in report
        assert report["SCALP"]["signals"] == 3
        assert report["SCALP"]["hit_rate"] == 1.0
        assert report["SCALP"]["expectancy_points"] > 0

    def test_bias_correction_suggestion(self, tmp_path) -> None:
        engine = make_engine(tmp_path)
        for i in range(6):
            signal = make_signal(trap_type="BULL_TRAP")
            signal.signal_id = f"trap{i}"
            engine.register_signal(signal)
            engine.approve(signal.signal_id)
            engine.monitor(signal.signal_id, entry_fill=24001.0)
            engine.record_exit(
                signal.signal_id,
                exit_price=23990.0,
                exit_reason="SL_HIT",
                pnl_points=-11.0,
                mfe=1.0,
                mae=12.0,
                ts=1_700_001_000.0,
            )
            engine.close_and_write_back(signal.signal_id)
        suggestions = engine.bias_correction_suggestions()
        assert any("BULL_TRAP" in s for s in suggestions)

    def test_no_suggestion_with_few_samples(self, tmp_path) -> None:
        engine = make_engine(tmp_path)
        assert engine.bias_correction_suggestions() == []

    def test_invalid_exit_reason(self, tmp_path) -> None:
        engine = make_engine(tmp_path)
        signal = make_signal()
        engine.register_signal(signal)
        engine.approve(signal.signal_id)
        engine.monitor(signal.signal_id, entry_fill=24001.0)
        with pytest.raises(ValueError):
            engine.record_exit(
                signal.signal_id,
                exit_price=23990.0,
                exit_reason="BOGUS",
                pnl_points=-11.0,
                mfe=1.0,
                mae=12.0,
            )
