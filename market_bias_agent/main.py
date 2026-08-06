"""Application entry point.

Boots config + logging, exposes a /health endpoint, and (Phase 1+) starts
the websocket pipeline and cron schedulers.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config.settings import ConfigError, get_settings
from core.feed_factory import build_strikes_provider, build_transport
from core.health import HealthRegistry
from core.logger import get_logger, setup_logging
from core.redis_manager import RedisManager
from core.strikes_manager import StrikesManager
from core.tick_pipeline import TickPipeline
from core.tick_validator import TickValidator
from core.websocket_client import BreezeWebSocketClient
from modules.monitoring import MonitoringService, status_page

log = get_logger(__name__)

settings = get_settings()
redis_mgr = RedisManager(settings)
health = HealthRegistry(settings)
monitoring = MonitoringService(settings, health, audit_writer=redis_mgr.push_audit)


async def _start_pipeline() -> None:
    """Phase 1: strikes sync + resilient websocket feed -> tick pipeline."""
    validator = TickValidator()
    pipeline = TickPipeline(settings, redis_mgr, validator, health=health)

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
    pipeline_task = asyncio.create_task(_start_pipeline())
    try:
        yield
    finally:
        pipeline_task.cancel()
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
    return page


@app.get("/ops/watchdog")
def ops_watchdog() -> dict:
    """Manually trigger the stale-tick watchdog (or via cron)."""
    alerted = monitoring.watchdog_check()
    return {"alerted": alerted, "last_tick_age_seconds": health.last_tick_age_seconds()}


@app.get("/ops/daily-report")
def ops_daily_report() -> dict:
    """Build the daily ops report (Telegram text preview)."""
    return {"report": monitoring.build_daily_report(), "sent": False}


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
