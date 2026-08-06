"""Embedder abstraction for trap events (Enhancement Phase 5).

`GeminiEmbedder` uses Google Gemini's OpenAI-compatible endpoint
(`text-embedding-004`) over the `market_state` string. `FeatureEmbedder`
produces a deterministic numeric embedding from raw features so the pipeline
is fully testable offline and in backtests without spending tokens.
"""

from __future__ import annotations

from typing import Any, Protocol

from config.constants import (
    GEMINI_BASE_URL,
    OI_VELOCITY_SCALE,
    QDRANT_COLLECTION_DIM,
    SPOT_REFERENCE,
    SPOT_SCALE,
)
from config.settings import Settings
from core.logger import get_logger

log = get_logger(__name__)

_REGIMES = ("CALM", "ACTIVE", "HIGH_VOL")


class Embedder(Protocol):
    """Turns a trap payload into an embedding vector."""

    def embed(self, payload: dict[str, Any]) -> list[float]: ...


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(value, bound))


class FeatureEmbedder:
    """Deterministic numeric embedding of the PRD `features` dict.

    Vector layout (dim 8):
      [0] pcr normalized          (pcr / 3.0, clipped)
      [1] spot pinning            (spot - SPOT_REFERENCE) / SPOT_SCALE, clipped
      [2] call_oi_vel_1m          / OI_VELOCITY_SCALE, clipped
      [3] put_oi_vel_1m           / OI_VELOCITY_SCALE, clipped
      [4] velocity_5m             / OI_VELOCITY_SCALE, clipped
      [5:8] volatility one-hot    CALM / ACTIVE / HIGH_VOL
    """

    def __init__(self, dim: int = QDRANT_COLLECTION_DIM) -> None:
        self._dim = dim

    def embed(self, payload: dict[str, Any]) -> list[float]:
        features = payload.get("features") or {}
        regime = str(features.get("volatility", "ACTIVE")).upper()
        if regime not in _REGIMES:
            regime = "ACTIVE"
        one_hot = [1.0 if regime == r else 0.0 for r in _REGIMES]
        vector = [
            _clip(float(features.get("pcr", 1.0)), 3.0) / 3.0,
            _clip((float(features.get("spot", SPOT_REFERENCE)) - SPOT_REFERENCE) / SPOT_SCALE, 1.0),
            _clip(float(features.get("call_oi_vel_1m", 0.0)), OI_VELOCITY_SCALE)
            / OI_VELOCITY_SCALE,
            _clip(float(features.get("put_oi_vel_1m", 0.0)), OI_VELOCITY_SCALE) / OI_VELOCITY_SCALE,
            _clip(float(features.get("velocity_5m", 0.0)), OI_VELOCITY_SCALE) / OI_VELOCITY_SCALE,
            *one_hot,
        ]
        if len(vector) > self._dim:
            vector = vector[: self._dim]
        while len(vector) < self._dim:
            vector.append(0.0)
        return vector


class GeminiEmbedder:
    """text-embedding-004 via Gemini's OpenAI-compatible endpoint (live path)."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001") -> None:
        if not api_key:
            raise ValueError("GeminiEmbedder requires GEMINI_API_KEY")
        self._api_key = api_key
        self._model = model
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    def embed(self, payload: dict[str, Any]) -> list[float]:
        text = payload.get("market_state") or ""
        if not text:
            raise ValueError("GeminiEmbedder needs a non-empty market_state payload field")
        response = self._client.embeddings.create(model=self._model, input=[text])
        return [float(v) for v in response.data[0].embedding]


def get_embedder(settings: Settings) -> Embedder:
    """Prefer the real model when a key exists, else deterministic offline mode."""
    if settings.gemini_api_key:
        try:
            return GeminiEmbedder(settings.gemini_api_key, settings.embedding_model)
        except Exception:  # noqa: BLE001 - fall back to offline embedding
            log.warning("gemini_embedder_unavailable_fallback_to_feature")
    return FeatureEmbedder()
