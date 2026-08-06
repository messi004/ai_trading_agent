"""Qdrant vector memory (PRD Module 3 / Enhancement Phase 5).

`MemoryVectorStore` is a dependency-free in-memory cosine store used for
tests, offline boot and backtests. `QdrantVectorStore` talks to a real
Qdrant instance via the qdrant-client. `get_vector_store` picks the real
store when Qdrant is reachable and falls back to memory otherwise, so the
agent boots even with no vector DB present.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from config.constants import (
    QDRANT_BATCH_SIZE,
    QDRANT_COLLECTION_DIM,
    QDRANT_HNSW_EF_CONSTRUCT,
    QDRANT_HNSW_M,
    QDRANT_HNSW_MAX_INDEXING_THREADS,
    QDRANT_PAYLOAD_INDEX_FIELDS,
)
from config.settings import Settings
from core.logger import get_logger
from memory.embeddings import Embedder
from memory.trap_records import TrapRecord

log = get_logger(__name__)


@dataclass
class SearchHit:
    vector_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    def ensure_collection(self, dim: int = QDRANT_COLLECTION_DIM) -> None: ...

    def upsert(self, records: list[TrapRecord], embedder: Embedder) -> list[str]: ...

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[SearchHit]: ...

    def delete(self, filter_payload: dict[str, Any]) -> int: ...

    def count(self) -> int: ...

    def export_snapshot(self, path: str) -> str: ...

    def close(self) -> None: ...


def _get_nested(payload: dict[str, Any], dotted_key: str) -> Any:
    """Resolve 'a.b.c' against a (possibly nested) payload dict."""
    value: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _payload_matches(payload: dict[str, Any], conditions: dict[str, Any]) -> bool:
    return all(_get_nested(payload, k) == v for k, v in conditions.items())


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MemoryVectorStore:
    """In-memory cosine store (tests / offline / backtests)."""

    def __init__(self, collection: str = "nifty_historical_traps") -> None:
        self._collection = collection
        self._vectors: dict[str, list[float]] = {}
        self._payloads: dict[str, dict[str, Any]] = {}

    def ensure_collection(self, dim: int = QDRANT_COLLECTION_DIM) -> None:
        pass  # in-memory: nothing to create

    def upsert(self, records: list[TrapRecord], embedder: Embedder) -> list[str]:
        ids: list[str] = []
        for record in records:
            vector = embedder.embed(record.to_payload())
            self._vectors[record.vector_id] = vector
            self._payloads[record.vector_id] = record.to_payload()
            ids.append(record.vector_id)
        return ids

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        scored: list[SearchHit] = []
        for vector_id, vector in self._vectors.items():
            payload = self._payloads[vector_id]
            if filter_payload is not None and not _payload_matches(payload, filter_payload):
                continue
            scored.append(
                SearchHit(
                    vector_id=vector_id,
                    score=cosine_similarity(query_vector, vector),
                    payload=payload,
                )
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:limit]

    def delete(self, filter_payload: dict[str, Any]) -> int:
        to_delete = [
            vid
            for vid, payload in self._payloads.items()
            if _payload_matches(payload, filter_payload)
        ]
        for vid in to_delete:
            self._vectors.pop(vid, None)
            self._payloads.pop(vid, None)
        return len(to_delete)

    def count(self) -> int:
        return len(self._vectors)

    def export_snapshot(self, path: str) -> str:
        data = [
            {"vector_id": vid, "vector": vec, "payload": self._payloads[vid]}
            for vid, vec in self._vectors.items()
        ]
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2))
        return str(target)

    def close(self) -> None:
        pass


class QdrantVectorStore:
    """Real Qdrant-backed store via qdrant-client."""

    def __init__(
        self,
        settings: Settings,
        *,
        collection: str | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._settings = settings
        self._collection = collection or settings.qdrant_collection
        self._timeout = int(timeout_seconds)
        self._client: Any = None

    @property
    def _qdrant(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                host=self._settings.qdrant_host,
                port=self._settings.qdrant_port,
                api_key=self._settings.qdrant_api_key or None,
                timeout=self._timeout,
            )
        return self._client

    def ensure_collection(self, dim: int = QDRANT_COLLECTION_DIM) -> None:
        from qdrant_client.http import models as qm

        collections = [c.name for c in self._qdrant.get_collections().collections]
        if self._collection in collections:
            self._ensure_payload_indexes()
            return
        self._qdrant.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            hnsw_config=qm.HnswConfigDiff(
                m=QDRANT_HNSW_M,
                ef_construct=QDRANT_HNSW_EF_CONSTRUCT,
                max_indexing_threads=QDRANT_HNSW_MAX_INDEXING_THREADS,
            ),
        )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """Scalar indexes over filter fields so filtered search stays fast."""
        from qdrant_client.http import models as qm

        try:
            schema = self._qdrant.get_collection(self._collection).payload_schema
            existing = set(schema.keys())
        except Exception:  # noqa: BLE001 - schema read may not exist yet
            existing = set()
        for index_field in QDRANT_PAYLOAD_INDEX_FIELDS:
            if index_field not in existing:
                try:
                    self._qdrant.create_payload_index(
                        collection_name=self._collection,
                        field_name=index_field,
                        field_schema=qm.PayloadSchemaType.KEYWORD,
                    )
                except Exception as exc:  # noqa: BLE001 - index creation is best-effort
                    log.debug("payload_index_skip", extra={"field": index_field, "error": str(exc)})

    def upsert(self, records: list[TrapRecord], embedder: Embedder) -> list[str]:
        from qdrant_client.http import models as qm

        self.ensure_collection()
        ids: list[str] = []
        for start in range(0, len(records), QDRANT_BATCH_SIZE):
            batch = records[start : start + QDRANT_BATCH_SIZE]
            points = []
            for record in batch:
                points.append(
                    qm.PointStruct(
                        id=str(uuid.UUID(record.vector_id)),
                        vector=embedder.embed(record.to_payload()),
                        payload=record.to_payload(),
                    )
                )
            self._qdrant.upsert(collection_name=self._collection, points=points)
            ids.extend(r.vector_id for r in batch)
        return ids

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        self.ensure_collection()
        points = self._qdrant.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=self._build_filter(filter_payload),
        )
        return [
            SearchHit(
                vector_id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in points
        ]

    def delete(self, filter_payload: dict[str, Any]) -> int:
        from qdrant_client.http import models as qm

        self.ensure_collection()
        flt = self._build_filter(filter_payload)
        matched = int(self._qdrant.count(collection_name=self._collection, count_filter=flt).count)
        self._qdrant.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(filter=flt),
        )
        return matched

    def count(self) -> int:
        try:
            return int(self._qdrant.count(collection_name=self._collection).count)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            return 0

    def export_snapshot(self, path: str) -> str:
        self.ensure_collection()
        records: list[dict[str, Any]] = []
        scroll_id = None
        while True:
            result = self._qdrant.scroll(
                collection_name=self._collection,
                limit=QDRANT_BATCH_SIZE,
                scroll_id=scroll_id,
                with_payload=True,
                with_vectors=True,
            )
            records.extend(
                {
                    "vector_id": str(point.id),
                    "vector": list(point.vector),
                    "payload": dict(point.payload or {}),
                }
                for point in result[0]
            )
            scroll_id = result[1]
            if scroll_id is None:
                break
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records, indent=2))
        return str(target)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    def _build_filter(self, filter_payload: dict[str, Any] | None) -> Any:
        from qdrant_client.http import models as qm

        if not filter_payload:
            return None
        return qm.Filter(
            must=[
                qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
                for k, v in filter_payload.items()
            ]
        )


def get_vector_store(settings: Settings) -> VectorStore:
    """Real Qdrant when reachable, in-memory otherwise (graceful offline boot)."""
    store = QdrantVectorStore(settings)
    try:
        store.ensure_collection()
        log.info("vector_store_qdrant_connected", extra={"collection": store._collection})
        return store
    except Exception:  # noqa: BLE001 - no Qdrant available
        log.warning(
            "vector_store_qdrant_unavailable_fallback_memory",
            extra={"host": settings.qdrant_host},
        )
        try:
            store.close()
        except Exception:  # noqa: BLE001
            pass
        return MemoryVectorStore(collection=settings.qdrant_collection)
