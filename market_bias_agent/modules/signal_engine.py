"""Live signal engine — wiring the PRD pipeline end-to-end.

Ties the pieces together on the real tick path:
  trigger (math_engine) → Maker (Gemini LLM) → Checker (rules A-G) →
  approved signal persisted in the signal store → paper-trader shadow fill →
  price feed resolves SL/Target/time exits → post-analysis records outcome,
  writes the actual outcome back to Qdrant memory, and feeds PnL into the
  Checker's Rule-D daily-loss circuit.

Runs synchronously so it slots into the existing websocket tick handler;
cooldowns keep LLM spend bounded (duplicate strikes are gated before the
model is ever called).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from config.constants import (
    DUPLICATE_ALERT_COOLDOWN_SECONDS,
    SPOT_TICK_BUFFER_SIZE,
)
from config.settings import Settings
from core.candle_engine import atr, build_candles_from_ticks, classify_volatility_regime
from core.features import compute_volume_delta
from core.logger import get_logger
from core.math_engine import (
    compute_oi_metrics,
    compute_pcr,
    evaluate_triggers_with_regime,
    nearest_round_level,
)
from core.redis_manager import RedisManager
from core.signals import StructuredSignal
from graph.workflow import SignalWorkflow
from modules.paper_trader import PaperPosition, PaperTrader
from modules.post_analysis import PostAnalysisEngine
from utils.telegram_bot import TelegramBot
from utils.time_utils import market_status

log = get_logger(__name__)

MAX_OI_HISTORY_SECONDS = 320.0
FEATURE_THROTTLE_SECONDS = 1.0


def _expiry_week(ts_epoch: float) -> int:
    """1..2 per month as a coarse expiry-week band for memory filtering."""
    day = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isocalendar()
    return (day[1] + 1) // 2


class SignalEngine:
    def __init__(
        self,
        settings: Settings,
        workflow: SignalWorkflow,
        redis: RedisManager,
        *,
        post_analysis: PostAnalysisEngine | None = None,
        paper_trader: PaperTrader | None = None,
        telegram: TelegramBot | None = None,
        memory: Any | None = None,
        health: Any | None = None,
        is_market_open: Callable[[], bool] | None = None,
        feature_throttle_seconds: float = FEATURE_THROTTLE_SECONDS,
    ) -> None:
        self._settings = settings
        self._workflow = workflow
        self._redis = redis
        self._post = post_analysis or PostAnalysisEngine(settings)
        self._trader = paper_trader or PaperTrader(settings)
        self._telegram = telegram or TelegramBot(settings)
        self._memory = memory
        self._health = health
        self._is_market_open = is_market_open or (lambda: market_status() == "OPEN")
        self._feature_throttle = feature_throttle_seconds

        # Time-sampled total-OI history for 1m/5m velocity computation.
        self._oi_history: deque[tuple[float, float, float]] = deque()  # (ts, call, put)
        self._last_feature_ts = 0.0
        self._spot = 0.0
        self._atr = 0.0
        self._regime = "ACTIVE"
        self._last_signal_by_strike: dict[float, float] = {}
        self._last_signal_ts = 0.0

    # ------------------------------------------------------------------
    # Tick entry point (called by the pipeline for every validated tick)
    # ------------------------------------------------------------------
    def on_tick(self, tick: dict[str, Any]) -> None:
        now = time.time()
        if tick.get("type") == "spot":
            self._spot = float(tick.get("price", self._spot))
            self._handle_spot(tick)
        self._maybe_evaluate(now)

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------
    def _build_features(self, now: float) -> dict[str, Any]:
        total_call, total_put = self._total_oi()
        self._record_oi_snapshot(now, total_call, total_put)

        call_60, call_300 = self._oi_total_at(now, 60), self._oi_total_at(now, 300)
        put_60, put_300 = self._oi_put_at(now, 60), self._oi_put_at(now, 300)
        pcr = compute_pcr(total_call, total_put) if total_call > 0 else 1.0

        spot_ticks = self._safe_spot_ticks()
        candles = build_candles_from_ticks(spot_ticks) if spot_ticks else []
        self._atr = atr(candles) if candles else self._atr
        self._regime = classify_volatility_regime(candles) if candles else self._regime
        volume_delta = compute_volume_delta(spot_ticks[-50:]) if spot_ticks else None

        near_level = self._near_level(self._spot)
        return {
            "spot": self._spot,
            "pcr": pcr,
            "total_call_oi": total_call,
            "total_put_oi": total_put,
            "call_oi_vel_1m": total_call - call_60,
            "put_oi_vel_1m": total_put - put_60,
            "call_oi_vel_5m": total_call - call_300,
            "put_oi_vel_5m": total_put - put_300,
            "atr": self._atr,
            "strike": near_level or self._spot,
            "near_level": near_level,
            "regime": self._regime,
            "volatility": self._regime,
            "volume_delta_1m": volume_delta.delta if volume_delta else 0.0,
            "volume_delta_bias": volume_delta.bias if volume_delta else "NEUTRAL",
        }

    def _total_oi(self) -> tuple[float, float]:
        total_call = 0.0
        total_put = 0.0
        try:
            strikes = self._redis.get_strikes()
            for strike in strikes:
                call_window = self._redis.get_call_oi_window(strike)
                put_window = self._redis.get_put_oi_window(strike)
                if call_window:
                    total_call += call_window[-1]
                if put_window:
                    total_put += put_window[-1]
        except Exception as exc:  # noqa: BLE001 - never break the tick path
            log.warning("signal_oi_read_failed", extra={"error": str(exc)})
        return total_call, total_put

    def _record_oi_snapshot(self, now: float, total_call: float, total_put: float) -> None:
        last = self._oi_history[-1] if self._oi_history else None
        if last is None or last[1] != total_call or last[2] != total_put:
            self._oi_history.append((now, total_call, total_put))
        cutoff = now - MAX_OI_HISTORY_SECONDS
        while self._oi_history and self._oi_history[0][0] < cutoff:
            self._oi_history.popleft()

    def _oi_total_at(self, now: float, window_seconds: float) -> float:
        """Call total OI at (now - window), or the oldest sample if none that old yet."""
        best = self._oi_lookup(now, window_seconds)
        return best[1] if best else 0.0

    def _oi_put_at(self, now: float, window_seconds: float) -> float:
        """Put total OI at (now - window), or the oldest sample if none that old yet."""
        best = self._oi_lookup(now, window_seconds)
        return best[2] if best else 0.0

    def _oi_lookup(self, now: float, window_seconds: float) -> tuple[float, float, float] | None:
        target = now - window_seconds
        best: tuple[float, float, float] | None = None
        for entry in self._oi_history:
            if entry[0] <= target:
                best = entry
            else:
                break
        if best is None and self._oi_history:
            best = self._oi_history[0]
        return best

    def _safe_spot_ticks(self) -> list[dict[str, Any]]:
        try:
            return self._redis.get_spot_ticks(SPOT_TICK_BUFFER_SIZE)
        except Exception as exc:  # noqa: BLE001
            log.warning("signal_spot_read_failed", extra={"error": str(exc)})
            return []

    def _near_level(self, spot: float) -> float | None:
        try:
            levels = self._redis.get_pre_market_levels()
            if isinstance(levels, dict):
                for key in ("support_resistance", "levels"):
                    bucket = levels.get(key)
                    if isinstance(bucket, list | tuple) and bucket:
                        return min(bucket, key=lambda lv: abs(float(lv) - spot))
                for level_key in ("r1", "s1", "r2", "s2", "pivot", "max_pain"):
                    if level_key in levels:
                        return float(levels[level_key])
        except Exception as exc:  # noqa: BLE001
            log.warning("signal_level_read_failed", extra={"error": str(exc)})
        return nearest_round_level(spot)

    # ------------------------------------------------------------------
    # Trigger evaluation (throttled)
    # ------------------------------------------------------------------
    def _maybe_evaluate(self, now: float) -> None:
        if now - self._last_feature_ts < self._feature_throttle:
            return
        if not self._is_market_open():
            return
        self._last_feature_ts = now
        if self._spot <= 0:
            return

        features = self._build_features(now)
        metrics = compute_oi_metrics(
            total_call_oi=features["total_call_oi"],
            total_put_oi=features["total_put_oi"],
            call_oi_60s_ago=features["total_call_oi"] - features["call_oi_vel_1m"],
            call_oi_300s_ago=features["total_call_oi"] - features["call_oi_vel_5m"],
            put_oi_60s_ago=features["total_put_oi"] - features["put_oi_vel_1m"],
            put_oi_300s_ago=features["total_put_oi"] - features["put_oi_vel_5m"],
        )
        trigger = evaluate_triggers_with_regime(
            metrics,
            self._spot,
            features["regime"],
            self._settings.trigger,
        )
        if not trigger.triggered:
            return
        features["trigger_type"] = trigger.trigger_type
        self._produce_signal(features, now)

    # ------------------------------------------------------------------
    # Signal production (Maker -> Checker -> persist -> paper fill -> alert)
    # ------------------------------------------------------------------
    def _produce_signal(self, features: dict[str, Any], now: float) -> None:
        strike = float(features.get("strike", self._spot))
        last = self._last_signal_by_strike.get(strike)
        if last is not None and now - last < DUPLICATE_ALERT_COOLDOWN_SECONDS:
            return
        if self._health is not None:
            self._health.record_trigger()

        memory_context = self._memory_context(features, now)
        decision = self._workflow.run_sync(features, memory_context)

        if self._health is not None:
            self._health.record_llm_tokens(self._workflow.maker().budget().spent)
        approved = decision.get("status") == "APPROVED"
        if self._health is not None:
            self._health.record_verdict(approved)

        if not approved:
            log.info(
                "signal_rejected",
                extra={
                    "trigger": features.get("trigger_type"),
                    "rules": decision.get("rejected_rules"),
                },
            )
            return

        signal_dict = decision.get("signal") or {}
        signal = StructuredSignal.from_dict(signal_dict)
        if not signal.ts_epoch:
            signal.ts_epoch = now

        self._last_signal_by_strike[strike] = now
        self._last_signal_ts = now

        self._post.register_signal(signal)
        self._post.approve(signal.signal_id)

        entry = signal.entry
        self._trader.submit_signal(signal.to_dict())
        self._post.monitor(signal.signal_id, entry)
        log.info(
            "signal_approved",
            extra={
                "signal_id": signal.signal_id,
                "direction": signal.direction,
                "trigger": signal.trigger_type,
                "entry": round(entry, 2),
                "sl": signal.sl,
                "target": signal.target,
            },
        )
        self._telegram.send_alert(self._alert_text(signal, decision, features))

    def _memory_context(self, features: dict[str, Any], now: float) -> dict[str, Any] | None:
        if self._memory is None:
            return None
        try:
            similar = self._memory.query_similar(
                features,
                expiry_week=_expiry_week(now),
                regime=features.get("regime"),
            )
            return similar.to_dict()
        except Exception as exc:  # noqa: BLE001 - memory must never block a signal
            log.warning("signal_memory_query_failed", extra={"error": str(exc)})
            return None

    def _alert_text(
        self, signal: StructuredSignal, decision: dict[str, Any], features: dict[str, Any]
    ) -> str:
        similar = decision.get("similar") if isinstance(decision.get("similar"), dict) else None
        similar_line = ""
        if similar:
            similar_line = (
                f"\nSimilar situations: {similar.get('similar_count', 0)} | "
                f"win-rate {similar.get('win_rate', 0.0):.0%}"
            )
        entry_low, entry_high = signal.entry_zone
        return (
            "🧭 <b>SIGNAL APPROVED</b>\n"
            f"Direction: <b>{signal.direction}</b> (conf {signal.confidence:.2f})\n"
            f"Trigger: {signal.trigger_type} | Trap: {signal.trap_type}\n"
            f"Entry zone: {entry_low:,.1f} – {entry_high:,.1f} | "
            f"SL {signal.sl} | Target {signal.target}\n"
            f"{similar_line}\n"
            f"<i>{signal.rationale}</i>"
        )

    # ------------------------------------------------------------------
    # Paper position monitoring
    # ------------------------------------------------------------------
    def _handle_spot(self, tick: dict[str, Any]) -> None:
        price = float(tick["price"])
        ts = float(tick.get("ts_epoch", time.time()))
        try:
            closed = self._trader.update_price(price, ts)
        except Exception as exc:  # noqa: BLE001
            log.warning("paper_update_failed", extra={"error": str(exc)})
            return
        for position in closed:
            self._close_position(position)

    def _close_position(self, position: PaperPosition) -> None:
        try:
            summary = self._post.record_exit(
                position.signal_id,
                exit_price=float(position.exit_price or 0.0),
                exit_reason=str(position.exit_reason or "TIME_EXIT"),
                pnl_points=float(position.pnl_points or 0.0),
                mfe=float(position.mfe),
                mae=float(position.mae),
                ts=time.time(),
            )
            self._post.close_and_write_back(position.signal_id)
            self._workflow.checker().record_exit_pnl(float(position.pnl_points or 0.0))
            self._telegram.send_alert(summary.to_text())
        except Exception as exc:  # noqa: BLE001 - post-trade bookkeeping must not crash ticks
            log.warning(
                "post_trade_failed",
                extra={"signal_id": position.signal_id, "error": str(exc)},
            )

    # ------------------------------------------------------------------
    # Introspection for /status and tests
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "paper_trader": self._trader.report(),
            "weekly_report": self._post.weekly_report(),
            "bias_correction": self._post.bias_correction_suggestions(),
            "signals_logged": self._post.signal_count(),
            "last_signal_ts": self._last_signal_ts or None,
        }
