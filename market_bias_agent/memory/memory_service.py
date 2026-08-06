"""Memory service — indexing, similarity query, lifecycle (Enhancement Phase 5).

Implements the PRD query strategy: top-K similar historical traps within a
scalar-filtered band (expiry week), with a score boost when market regime /
PCR match the current conditions, plus a win-rate summary for Telegram alerts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from config.constants import (
    OUTCOME_LOSS,
    OUTCOME_WIN,
    QDRANT_MAX_AGE_DAYS,
    QDRANT_SIMILAR_BOOST,
    QDRANT_SIMILAR_LIMIT,
)
from config.settings import Settings
from core.logger import get_logger
from memory.embeddings import Embedder, get_embedder
from memory.trap_records import TrapRecord, parse_subsequent_move_points
from memory.vector_store import (
    MemoryVectorStore,
    QdrantVectorStore,
    SearchHit,
    VectorStore,
    get_vector_store,
)

log = get_logger(__name__)


@dataclass
class SimilarSituation:
    """Top-K similar situations + aggregate outcome summary."""

    hits: list[SearchHit]
    count: int = 0
    win_rate: float = 0.0
    avg_move_points: float = 0.0
    matched_expiry_week: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "similar_count": self.count,
            "win_rate": self.win_rate,
            "avg_move_points": self.avg_move_points,
            "matched_expiry_week": self.matched_expiry_week,
            "hits": [
                {
                    "vector_id": h.vector_id,
                    "score": round(h.score, 4),
                    "outcome": h.payload.get("historical_outcome"),
                }
                for h in self.hits
            ],
        }


def _boost_apply(
    hits: list[SearchHit], boost_conditions: list[tuple[str, Any]], boost: float
) -> list[SearchHit]:
    """Add `boost` to the score for EACH matching condition (regime + band)."""
    from memory.vector_store import _get_nested

    for hit in hits:
        matches = sum(
            1 for key, value in boost_conditions if _get_nested(hit.payload, key) == value
        )
        if matches:
            hit.score += boost * matches
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


class MemoryService:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._settings = settings
        self._store = store or get_vector_store(settings)
        self._embedder = embedder or get_embedder(settings)
        self._store.ensure_collection()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_trap_event(
        self,
        *,
        features: dict[str, Any],
        outcome: str,
        subsequent_move: str,
        session_date: str,
        expiry_week: int,
    ) -> str:
        """Build + store one trap event. Returns the vector_id."""
        record = TrapRecord(
            features=features,
            historical_outcome=outcome,
            subsequent_move=subsequent_move,
            session_date=session_date,
            expiry_week=expiry_week,
        )
        ids = self.index_records([record])
        return ids[0]

    def index_records(self, records: list[TrapRecord]) -> list[str]:
        if not records:
            return []
        ids = self._store.upsert(records, self._embedder)
        log.info(
            "memory_indexed_traps",
            extra={"count": len(ids), "collection": getattr(self._store, "_collection", "")},
        )
        return ids

    # ------------------------------------------------------------------
    # Query strategy (PRD 6.2)
    # ------------------------------------------------------------------
    def query_similar(
        self,
        features: dict[str, Any],
        *,
        expiry_week: int,
        limit: int = QDRANT_SIMILAR_LIMIT,
        regime: str | None = None,
    ) -> SimilarSituation:
        """Top-K similar traps in the same expiry-week band, regime-boosted."""
        query_vector = self._embedder.embed({"features": features})
        hits = self._store.search(
            query_vector,
            limit=max(limit, 1),
            filter_payload={"expiry_week": expiry_week},
        )
        boost_conditions: list[tuple[str, Any]] = [("expiry_week", expiry_week)]
        if regime is not None:
            boost_conditions.append(("features.volatility", regime))
        hits = _boost_apply(hits, boost_conditions, QDRANT_SIMILAR_BOOST)
        return self._summarize(hits, matched_expiry_week=bool(hits))

    def _summarize(self, hits: list[SearchHit], matched_expiry_week: bool) -> SimilarSituation:
        outcomes = [h.payload.get("historical_outcome") for h in hits]
        wins = sum(1 for o in outcomes if o == OUTCOME_WIN)
        losses = sum(1 for o in outcomes if o == OUTCOME_LOSS)
        moves: list[float] = []
        for raw in (h.payload.get("subsequent_move") for h in hits):
            parsed = parse_subsequent_move_points(str(raw)) if raw is not None else None
            if parsed is not None:
                moves.append(parsed)
        win_rate = wins / (wins + losses) if (wins + losses) else 0.0
        return SimilarSituation(
            hits=hits,
            count=len(hits),
            win_rate=win_rate,
            avg_move_points=sum(moves) / len(moves) if moves else 0.0,
            matched_expiry_week=matched_expiry_week,
        )

    # ------------------------------------------------------------------
    # Data lifecycle (PRD 6.3)
    # ------------------------------------------------------------------
    def compact_stale(self, max_age_days: int = QDRANT_MAX_AGE_DAYS) -> int:
        """Remove vectors older than `max_age_days` (weekly compaction job)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).date().isoformat()
        removed = self._store.delete({"session_date": cutoff})
        log.info("memory_compaction", extra={"cutoff": cutoff, "removed": removed})
        return removed

    def export_snapshot(self, path: str) -> str:
        """Back up the full collection to a JSON file."""
        target = self._store.export_snapshot(path)
        log.info("memory_snapshot", extra={"path": target})
        return target

    @property
    def store(self) -> VectorStore:
        return self._store


def build_memory_service(settings: Settings, *, force_memory: bool = False) -> MemoryService:
    """Construct a service; when forced offline, always uses the memory store."""
    store: VectorStore
    if force_memory:
        store = MemoryVectorStore(collection=settings.qdrant_collection)
    else:
        store = QdrantVectorStore(settings)
    return MemoryService(settings, store=store, embedder=get_embedder(settings))
