"""Phase 5 memory tests: trap records, embeddings, stores, service, EOD wiring."""

import json

import pytest

from config.constants import QDRANT_COLLECTION_DIM
from config.settings import Settings
from core.participant_oi import ParticipantOIError
from memory.embeddings import FeatureEmbedder, GeminiEmbedder, get_embedder
from memory.memory_service import MemoryService, SimilarSituation, build_memory_service
from memory.trap_records import (
    TrapRecord,
    compute_market_state,
    compute_subsequent_move,
    parse_subsequent_move_points,
)
from memory.vector_store import MemoryVectorStore, cosine_similarity
from modules.eod_engine import EODEngine

FEATURES = {
    "pcr": 0.95,
    "spot": 24005,
    "call_oi_vel_1m": -85000,
    "put_oi_vel_1m": 30000,
    "velocity_5m": 150000,
    "volatility": "HIGH_VOL",
}


def make_record(
    seed_features: dict | None = None, outcome: str = "TARGET_HIT", **overrides
) -> TrapRecord:
    kwargs = dict(
        features=seed_features or dict(FEATURES),
        historical_outcome=outcome,
        subsequent_move="+45 points in 15 mins",
        session_date="2026-08-05",
        expiry_week=3,
    )
    kwargs.update(overrides)
    return TrapRecord(**kwargs)


def make_service() -> MemoryService:
    settings = Settings()
    return MemoryService(settings, store=MemoryVectorStore(), embedder=FeatureEmbedder())


class TestTrapRecords:
    def test_market_state_string(self) -> None:
        assert compute_market_state(FEATURES) == "PCR: 0.95, Spot: 24005, Call_OI_Vel: -85,000"

    def test_subsequent_move(self) -> None:
        assert compute_subsequent_move(24000, 23955, 15) == "-45 points in 15 mins"
        assert parse_subsequent_move_points("-45 points in 15 mins") == -45.0
        assert parse_subsequent_move_points("+6.5 points in 2 mins") == 6.5
        assert parse_subsequent_move_points("no data") is None

    def test_record_payload_roundtrip(self) -> None:
        record = make_record()
        payload = record.to_payload()
        assert payload["market_state"] == compute_market_state(FEATURES)
        assert TrapRecord.from_payload(payload) == record

    def test_invalid_outcome_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_record(outcome="NOT_A_TRAP_OUTCOME")


class TestEmbeddings:
    def test_feature_embedder_dim_and_deterministic(self) -> None:
        embedder = FeatureEmbedder()
        vec = embedder.embed(make_record().to_payload())
        assert len(vec) == QDRANT_COLLECTION_DIM
        assert embedder.embed(make_record().to_payload()) == vec

    def test_feature_embedder_regime_one_hot(self) -> None:
        embedder = FeatureEmbedder()
        high = embedder.embed(make_record().to_payload())
        calm_rec = make_record(seed_features=dict(FEATURES, volatility="CALM"))
        calm = embedder.embed(calm_rec.to_payload())
        # regime sits at indices 5:8
        assert high[5:8] == [0.0, 0.0, 1.0]
        assert calm[5:8] == [1.0, 0.0, 0.0]

    def test_gemini_embedder_requires_key(self) -> None:
        with pytest.raises(ValueError):
            GeminiEmbedder(api_key="")

    def test_get_embedder_falls_back_offline(self) -> None:
        settings = Settings(gemini_api_key="")
        assert isinstance(get_embedder(settings), FeatureEmbedder)

    def test_get_embedder_defaults_to_feature_even_with_key(self) -> None:
        # GEMINI key present but backend unset -> FeatureEmbedder so Qdrant dim (8) matches.
        settings = Settings(gemini_api_key="g")
        embedder = get_embedder(settings)
        assert isinstance(embedder, FeatureEmbedder)
        assert embedder.dim == QDRANT_COLLECTION_DIM

    def test_get_embedder_gemini_backend_explicit(self) -> None:
        # Explicit EMBEDDING_BACKEND=gemini uses GeminiEmbedder when a key exists.
        settings = Settings(gemini_api_key="g", embedding_backend="gemini")
        embedder = get_embedder(settings)
        assert isinstance(embedder, GeminiEmbedder)
        assert embedder.dim == 3072

    def test_get_embedder_gemini_backend_without_key_falls_back(self) -> None:
        settings = Settings(gemini_api_key="", embedding_backend="gemini")
        assert isinstance(get_embedder(settings), FeatureEmbedder)

    def test_cosine_similarity(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


class TestMemoryStore:
    def test_upsert_search_count(self) -> None:
        store = MemoryVectorStore()
        embedder = FeatureEmbedder()
        store.upsert(
            [make_record(), make_record(outcome="SL_HIT", subsequent_move="-30 points in 5 mins")],
            embedder,
        )
        assert store.count() == 2
        hits = store.search(embedder.embed(make_record().to_payload()))
        assert len(hits) == 2
        assert hits[0].score >= hits[1].score

    def test_search_filter_by_expiry_week(self) -> None:
        store = MemoryVectorStore()
        embedder = FeatureEmbedder()
        store.upsert(
            [
                make_record(expiry_week=3),
                make_record(expiry_week=4, outcome="SL_HIT"),
            ],
            embedder,
        )
        hits = store.search(
            embedder.embed(make_record().to_payload()),
            filter_payload={"expiry_week": 3},
        )
        assert len(hits) == 1
        assert hits[0].payload["expiry_week"] == 3

    def test_delete_by_filter(self) -> None:
        store = MemoryVectorStore()
        embedder = FeatureEmbedder()
        store.upsert(
            [
                make_record(session_date="2026-08-05"),
                make_record(session_date="2026-08-06", outcome="SL_HIT"),
            ],
            embedder,
        )
        removed = store.delete({"session_date": "2026-08-05"})
        assert removed == 1
        assert store.count() == 1

    def test_export_snapshot(self, tmp_path) -> None:
        store = MemoryVectorStore()
        store.upsert([make_record()], FeatureEmbedder())
        path = store.export_snapshot(str(tmp_path / "snap.json"))
        data = json.loads(open(path).read())
        assert len(data) == 1
        assert data[0]["payload"]["historical_outcome"] == "TARGET_HIT"


class TestMemoryService:
    def test_index_and_query_similar(self) -> None:
        service = make_service()
        service.index_trap_event(
            features=FEATURES,
            outcome="TARGET_HIT",
            subsequent_move="+45 points in 15 mins",
            session_date="2026-08-05",
            expiry_week=3,
        )
        service.index_trap_event(
            features=dict(FEATURES, pcr=1.1),
            outcome="SL_HIT",
            subsequent_move="-20 points in 4 mins",
            session_date="2026-08-04",
            expiry_week=3,
        )
        summary = service.query_similar(FEATURES, expiry_week=3)
        assert isinstance(summary, SimilarSituation)
        assert summary.count == 2
        assert summary.win_rate == pytest.approx(0.5)
        assert summary.avg_move_points == pytest.approx(12.5)
        assert summary.matched_expiry_week is True

    def test_query_boost_same_regime(self) -> None:
        service = make_service()
        service.index_records(
            [
                make_record(features=dict(FEATURES), outcome="TARGET_HIT"),
                make_record(features=dict(FEATURES, volatility="CALM"), outcome="SL_HIT"),
            ]
        )
        plain = service.query_similar(FEATURES, expiry_week=3)
        boosted = service.query_similar(FEATURES, expiry_week=3, regime="HIGH_VOL")
        # boosting moves the HIGH_VOL hit above the CALM one
        assert boosted.hits[0].payload["features"]["volatility"] == "HIGH_VOL"
        assert boosted.hits[0].score > plain.hits[0].score

    def test_compact_stale(self) -> None:
        service = make_service()
        service.index_records(
            [
                make_record(session_date="2026-08-05", outcome="TARGET_HIT"),
                make_record(session_date="2026-08-06", outcome="SL_HIT"),
            ]
        )
        removed = service.store.delete({"session_date": "2026-08-05"})
        assert removed == 1
        assert service.store.count() == 1

    def test_export_snapshot(self, tmp_path) -> None:
        service = make_service()
        service.index_records([make_record()])
        path = service.export_snapshot(str(tmp_path / "backup.json"))
        assert path.endswith("backup.json")

    def test_build_memory_service_forced(self) -> None:
        service = build_memory_service(Settings(), force_memory=True)
        assert isinstance(service.store, MemoryVectorStore)


class TestEODEngine:
    @staticmethod
    def _stub_participant():
        class _Stub:
            def fetch_latest(self):
                raise ParticipantOIError("offline test")

        return _Stub()

    def test_index_day_traps(self) -> None:
        engine = EODEngine(Settings(), memory=make_service())
        events = [
            {
                "features": dict(FEATURES),
                "outcome": "TARGET_HIT",
                "subsequent_move": "+45 points in 15 mins",
                "session_date": "2026-08-05",
                "expiry_week": 3,
            }
        ]
        ids = engine.index_day_traps(events)
        assert len(ids) == 1
        assert engine._memory.store.count() == 1

    def test_index_empty_events(self) -> None:
        engine = EODEngine(Settings(), memory=make_service())
        assert engine.index_day_traps([]) == []

    def test_weekly_compact(self) -> None:
        engine = EODEngine(Settings(), memory=make_service())
        engine.index_day_traps(
            [
                {
                    "features": dict(FEATURES),
                    "outcome": "SL_HIT",
                    "subsequent_move": "-10 points in 2 mins",
                    "session_date": "2026-08-05",
                    "expiry_week": 3,
                }
            ]
        )
        assert engine.weekly_compact() == 0  # recent data survives (cutoff is 45 days ago)

    def test_run_does_not_crash_empty(self) -> None:
        engine = EODEngine(
            Settings(), memory=make_service(), participant_provider=self._stub_participant()
        )
        result = engine.run()
        assert result["participant_report"]["status"] == "unavailable"
        assert "traps_indexed" in result

    def test_run_persists_bias_to_redis(self) -> None:
        import fakeredis

        from core.redis_manager import RedisManager

        redis_mgr = RedisManager(Settings())
        redis_mgr.client = fakeredis.FakeRedis(decode_responses=True)

        class _OkParticipant:
            def fetch_latest(self):
                from core.participant_oi import ParticipantPosition

                return [
                    ParticipantPosition(
                        client_type="FII",
                        future_index_long=32009.0,
                        future_index_short=287122.0,
                        option_index_call_long=0.0,
                        option_index_call_short=683709.0,
                        option_index_put_long=0.0,
                        option_index_put_short=620999.0,
                        date="2026-08-11",
                        nifty50=24636.0,
                    )
                ]

        engine = EODEngine(
            Settings(),
            memory=make_service(),
            participant_provider=_OkParticipant(),
            redis=redis_mgr,
        )
        result = engine.run()
        assert result["participant_report"]["status"] == "ok"
        bias = redis_mgr.get_eod_bias()
        assert bias is not None
        assert bias["bias"] == "BEARISH"
        assert "computed_at_ist" in bias
        assert "signals" in bias

    def test_detailed_report_contains_positioning_and_signals(self) -> None:
        class _OkParticipant:
            def fetch_latest(self):
                from core.participant_oi import ParticipantPosition

                return [
                    ParticipantPosition(
                        client_type="FII",
                        future_index_long=32009.0,
                        future_index_short=287122.0,
                        option_index_call_long=0.0,
                        option_index_call_short=683709.0,
                        option_index_put_long=0.0,
                        option_index_put_short=620999.0,
                        date="2026-08-11",
                        nifty50=24636.0,
                    ),
                    ParticipantPosition(
                        client_type="CLIENT",
                        future_index_long=228874.0,
                        future_index_short=62043.0,
                        option_index_call_long=0.0,
                        option_index_call_short=2897584.0,
                        option_index_put_long=0.0,
                        option_index_put_short=4141090.0,
                        date="2026-08-11",
                        nifty50=24636.0,
                    ),
                ]

        engine = EODEngine(Settings(), memory=make_service(), participant_provider=_OkParticipant())
        report = engine._participant_report()
        text = engine._report_text(report, ["a", "b"])
        assert "Structural bias: <b>BEARISH</b>" in text
        assert "📊 Positioning" in text
        assert "FII" in text
        assert "CLIENT" in text
        assert "futures net" in text
        assert "🎯 Signals" in text
        assert "Traps indexed to memory: 2" in text
