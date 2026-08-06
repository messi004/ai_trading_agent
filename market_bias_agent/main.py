"""Application entry point.

Boots config + logging, exposes a /health endpoint, and (Phase 1+) starts
the websocket pipeline and cron schedulers.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore[import-untyped]
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]
from fastapi import FastAPI

from config.constants import (
    EOD_CRON_HOUR_IST,
    EOD_CRON_MINUTE_IST,
    PREMARKET_CRON_HOUR_IST,
    PREMARKET_CRON_MINUTE_IST,
)
from config.settings import ConfigError, get_settings
from core.feed_factory import build_strikes_provider, build_transport
from core.health import HealthRegistry
from core.logger import get_logger, setup_logging
from core.redis_manager import RedisManager
from core.strikes_manager import StrikesManager
from core.tick_pipeline import TickPipeline
from core.tick_validator import TickValidator
from core.websocket_client import BreezeWebSocketClient
from graph.workflow import SignalWorkflow
from memory.memory_service import MemoryService, build_memory_service
from modules.checker_node import CheckerNode
from modules.maker_node import MakerNode
from modules.monitoring import MonitoringService, status_page
from modules.paper_trader import PaperTrader
from modules.post_analysis import PostAnalysisEngine
from modules.premarket_engine import PreMarketEngine
from modules.signal_engine import SignalEngine
from utils.telegram_bot import TelegramBot

log = get_logger(__name__)

settings = get_settings()
redis_mgr = RedisManager(settings)
health = HealthRegistry(settings)
premarket = PreMarketEngine(settings, redis_mgr)
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

maker_node = MakerNode(settings)
checker_node = CheckerNode(settings)
workflow = SignalWorkflow(settings, maker=maker_node, checker=checker_node)

telegram = TelegramBot(settings)
monitoring = MonitoringService(
    settings, health, telegram=telegram, checker=checker_node, audit_writer=redis_mgr.push_audit
)

memory_service: MemoryService | None = None
try:
    memory_service = build_memory_service(settings)
except Exception as exc:  # noqa: BLE001 - Qdrant may be down at boot; memory stays offline
    log.warning("memory_boot_fallback", extra={"error": str(exc)})
    memory_service = build_memory_service(settings, force_memory=True)

post_analysis = PostAnalysisEngine(settings, memory=memory_service)
paper_trader = PaperTrader(settings)

signal_engine = SignalEngine(
    settings,
    workflow,
    redis_mgr,
    post_analysis=post_analysis,
    paper_trader=paper_trader,
    telegram=telegram,
    memory=memory_service,
    health=health,
)


def _run_eod() -> None:
    try:
        from modules.eod_engine import EODEngine

        EODEngine(settings).run()
        health.record_cron_success("eod")
    except Exception as exc:  # noqa: BLE001
        log.error("eod_cron_failed", extra={"error": str(exc)})


def _run_premarket() -> None:
    try:
        premarket.run()
        health.record_cron_success("premarket")
    except Exception as exc:  # noqa: BLE001
        log.error("premarket_cron_failed", extra={"error": str(exc)})


def _start_cron() -> None:
    scheduler.add_job(
        _run_premarket,
        CronTrigger(hour=PREMARKET_CRON_HOUR_IST, minute=PREMARKET_CRON_MINUTE_IST),
        id="premarket",
        replace_existing=True,
    )
    scheduler.add_job(
        _run_eod,
        CronTrigger(hour=EOD_CRON_HOUR_IST, minute=EOD_CRON_MINUTE_IST),
        id="eod",
        replace_existing=True,
    )
    scheduler.start()


async def _start_pipeline() -> None:
    """Phase 1: strikes sync + resilient websocket feed -> tick pipeline."""
    validator = TickValidator()
    pipeline = TickPipeline(
        settings, redis_mgr, validator, health=health, signal_engine=signal_engine
    )

    strikes_mgr = StrikesManager(settings, redis_mgr, build_strikes_provider(settings))
    try:
        strikes_mgr.sync_if_due()
    except Exception as exc:  # noqa: BLE001
        log.error("strikes_initial_sync_failed", extra={"error": str(exc)})

    transport = build_transport(settings)
    ws_client = BreezeWebSocketClient(
        settings,
        transport,
        subscriptions_provider=lambda: [settings.nifty_symbol]
        + [f"STK{s}" for s in redis_mgr.get_strikes()],
    )
    ws_client.set_tick_handler(pipeline.process)
    ws_client.set_status_callback(health.set_ws)
    health.set_ws(False, 0)
    try:
        await ws_client.run()
    finally:
        health.set_ws(False, ws_client.reconnect_count)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level, settings.log_json)
    log.info(
        "app_start",
        extra={"app_env": settings.app_env, "profile": settings.threshold_profile},
    )
    try:
        redis_mgr.connect()
        log.info(
            "redis_connected",
            extra={"host": settings.redis_host, "port": settings.redis_port},
        )
    except Exception as exc:  # noqa: BLE001
        log.error("redis_connect_failed", extra={"error": str(exc)})
    workflow.build()
    pipeline_task = asyncio.create_task(_start_pipeline())
    _start_cron()
    try:
        yield
    finally:
        pipeline_task.cancel()
        scheduler.shutdown(wait=False)
        redis_mgr.close()


app = FastAPI(title="Dual-Engine AI Trading Agent", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health_check() -> dict:
    return health.status(
        redis_connected=redis_mgr.client is not None,
        profile=settings.threshold_profile,
    )


@app.get("/status")
def status() -> dict:
    """JSON status page: health + recent audit decisions (Phase 6)."""
    audit_recent = []
    redis_memory = None
    redis_keys = 0
    try:
        if redis_mgr.client is not None:
            audit_recent = redis_mgr.get_audit(count=10)
            redis_memory = redis_mgr.memory_usage_bytes()
            redis_keys = redis_mgr.dbsize()
    except Exception as exc:  # noqa: BLE001
        log.warning("status_audit_read_failed", extra={"error": str(exc)})
    page = status_page(
        health,
        redis_connected=redis_mgr.client is not None,
        profile=settings.threshold_profile,
        audit_recent=audit_recent,
    )
    page["redis_memory_bytes"] = redis_memory
    page["redis_keys"] = redis_keys
    page["premarket_levels"] = premarket.last_result or _premarket_from_redis()
    page["signal_engine"] = signal_engine.stats()
    return page


def _premarket_from_redis() -> dict | None:
    try:
        if redis_mgr.client is not None:
            return redis_mgr.get_pre_market_levels()
    except Exception as exc:  # noqa: BLE001
        log.warning("premarket_read_failed", extra={"error": str(exc)})
    return None


@app.get("/ops/watchdog")
def ops_watchdog() -> dict:
    """Manually trigger the stale-tick watchdog (or via cron)."""
    alerted = monitoring.watchdog_check()
    return {"alerted": alerted, "last_tick_age_seconds": health.last_tick_age_seconds()}


@app.get("/ops/daily-report")
def ops_daily_report() -> dict:
    """Build the daily ops report (Telegram text preview)."""
    report = monitoring.build_daily_report()
    stats = signal_engine.stats()
    report += (
        "\n"
        f"Paper: {stats['paper_trader']['positions_opened']} trades | "
        f"closed {stats['paper_trader']['closed']} | "
        f"PnL {stats['paper_trader']['total_pnl_points']:+.1f} pts"
    )
    if stats["bias_correction"]:
        report += "\nBias corrections:\n" + "\n".join(f"  - {s}" for s in stats["bias_correction"])
    return {"report": report, "sent": False}


@app.get("/test-signal")
async def test_signal() -> dict:
    """Run the full maker->checker workflow on a synthetic tick snapshot."""
    spot = 0.0
    levels = premarket.last_result or _premarket_from_redis()
    if isinstance(levels, dict):
        spot = float(levels.get("spot") or spot)
    features = {
        "spot": spot,
        "pcr": 1.05,
        "total_call_oi": 2500000.0,
        "total_put_oi": 2625000.0,
        "call_oi_vel_1m": 12500.0,
        "put_oi_vel_1m": -4300.0,
        "call_oi_vel_5m": 61200.0,
        "put_oi_vel_5m": -9800.0,
        "atr": 40.0,
        "strike": spot or 23500.0,
        "near_level": 23500.0,
        "trigger_type": "SCALP",
        "volatility": "ACTIVE",
        "volume_delta_1m": 14500.0,
    }
    decision = await workflow.invoke(features)
    decision["llm_budget_remaining"] = workflow.maker().budget().remaining
    decision["llm_cached"] = bool(workflow.maker().cache().size)
    return decision


def main() -> None:
    try:
        get_settings()
    except ConfigError as exc:
        setup_logging("CRITICAL", True)
        log.critical("config_error", extra={"error": str(exc)})
        raise SystemExit(1) from exc

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.health_port, log_level="warning")


if __name__ == "__main__":
    main()
