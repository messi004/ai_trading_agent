"""Matplotlib OI bar chart generator (PRD Module 5 / Enhancement Phase 7).

Renders the Call vs Put OI bar chart as PNG bytes using a headless Agg
backend, and reuses already-rendered figures via an LRU cache so repeated
Telegram alerts don't re-plot the same strikes (cost saving).
"""

from __future__ import annotations

import hashlib
import io
from typing import Any
from urllib.parse import urlencode

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from config.constants import LLM_CACHE_SIZE  # noqa: E402


def _chart_key(call_oi: dict, put_oi: dict, spot: float | None) -> str:
    """Stable hash of the inputs so near-identical requests reuse the figure."""
    payload = urlencode(
        {
            "call": sorted(call_oi.items()),
            "put": sorted(put_oi.items()),
            "spot": spot or 0,
        }
    )
    return hashlib.sha1(payload.encode()).hexdigest()


class ChartCache:
    """Bounded LRU cache of pre-rendered PNG bytes."""

    def __init__(self, max_size: int = LLM_CACHE_SIZE) -> None:
        self._max_size = max_size
        self._store: dict[str, bytes] = {}
        self._hits = 0
        self._misses = 0

    def get_or_render(self, key: str, renderer: Any) -> bytes:
        cached = self._store.get(key)
        if cached is not None:
            self._hits += 1
            self._store.pop(key)
            self._store[key] = cached  # move to MRU
            return cached
        self._misses += 1
        rendered = renderer()
        self._store[key] = rendered
        while len(self._store) > self._max_size:
            self._store.pop(next(iter(self._store)))  # evict LRU
        return rendered

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0


def generate_oi_chart(
    call_oi: dict,
    put_oi: dict,
    spot: float | None = None,
    *,
    cache: ChartCache | None = None,
) -> bytes:
    """Render Call vs Put OI per strike. Uses the shared cache when given."""
    key = _chart_key(call_oi, put_oi, spot)

    def render() -> bytes:
        strikes = sorted(set(call_oi) | set(put_oi))
        if not strikes:
            return _empty_png()
        x = range(len(strikes))
        calls = [call_oi.get(s, 0) for s in strikes]
        puts = [put_oi.get(s, 0) for s in strikes]
        fig, ax = plt.subplots(figsize=(9, 5), dpi=90)
        ax.bar(x, calls, width=0.4, label="Call OI", color="#d9534f", align="edge")
        ax.bar([i + 0.4 for i in x], puts, width=0.4, label="Put OI", color="#5cb85c", align="edge")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(s) for s in strikes], rotation=45, ha="right")
        ax.set_ylabel("Open Interest (contracts)")
        if spot is not None:
            ax.axvline(
                min(strikes, key=lambda s: abs(s - spot)) + 0.2, color="black", ls="--", lw=1
            )
            ax.set_title(f"Nifty {spot:,.0f} — Call vs Put OI")
        else:
            ax.set_title("Nifty — Call vs Put OI")
        ax.legend()
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        return buffer.getvalue()

    if cache is not None:
        return cache.get_or_render(key, render)
    return render()


def _empty_png() -> bytes:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=90)
    ax.text(0.5, 0.5, "No OI data", ha="center", va="center", fontsize=14)
    ax.axis("off")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()
