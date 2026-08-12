"""Application entry point.

Boots config + logging, exposes a /health endpoint, and (Phase 1+) starts
the websocket pipeline and cron schedulers.
"""

from __future__ import annotations

import asyncio
import types
from contextlib import asynccontextmanager
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore[import-untyped]
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]
from fastapi import FastAPI, HTTPException

from config.constants import (
    EOD_CRON_HOUR_IST,
    EOD_CRON_MINUTE_IST,
    PREMARKET_CRON_HOUR_IST,
    PREMARKET_CRON_MINUTE_IST,
    PREMARKET_REFRESH_CRON_HOUR_IST,
    PREMARKET_REFRESH_INTERVAL_MINUTES,
    SESSION_REFRESH_CRON_HOUR_IST,
    SESSION_REFRESH_CRON_MINUTE_IST,
)
from config.settings import ConfigError, get_settings
from core.backfill import BackfillService
from core.breeze_session import BreezeSessionManager
from core.feed_factory import (
    StubWsTransport,
    build_snapshot_provider,
    build_strikes_provider,
    build_transport,
)
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
from utils.telegram_listener import build_telegram_listener
from utils.time_utils import market_status

log = get_logger(__name__)

settings = get_settings()
redis_mgr = RedisManager(settings)
health = HealthRegistry(settings)
premarket = PreMarketEngine(settings, redis_mgr)
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
session_mgr = BreezeSessionManager(settings, redis_mgr)

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

pipeline: TickPipeline | None = None
pipeline_task: asyncio.Task | None = None
_ws_client: BreezeWebSocketClient | None = None
_pipeline_lock = asyncio.Lock()


def _run_eod() -> None:
    try:
        from modules.eod_engine import EODEngine

        EODEngine(settings, memory=memory_service, telegram=telegram).run()
        health.record_cron_success("eod")
        monitoring.send_daily_report()
    except Exception as exc:  # noqa: BLE001
        log.error("eod_cron_failed", extra={"error": str(exc)})


def _run_premarket() -> None:
    try:
        premarket.run()
        health.record_cron_success("premarket")
        telegram.send_ops(premarket.report_text())
    except Exception as exc:  # noqa: BLE001
        log.error("premarket_cron_failed", extra={"error": str(exc)})


def _run_premarket_refresh() -> None:
    """Intraday refresh of OI-profile levels (walls + max pain) during the session."""
    if market_status() != "OPEN":
        return
    try:
        premarket.refresh_intraday()
        health.record_cron_success("premarket_refresh")
    except Exception as exc:  # noqa: BLE001
        log.error("premarket_refresh_failed", extra={"error": str(exc)})


def _run_backtest_report(**kwargs: Any) -> str:
    """Backtest worker for the Telegram /backtest command (runs in a thread)."""
    from modules.backtest_runner import run_backtest_report

    return run_backtest_report(**kwargs)


def _premarket_report() -> str:
    """On-demand premarket report for Telegram commands/inline buttons."""
    premarket.run()
    return premarket.report_text()


def _daily_report() -> str:
    """On-demand daily ops report for Telegram commands/inline buttons."""
    return monitoring.build_daily_report()


def _run_session_refresh() -> None:
    try:
        if session_mgr.maybe_refresh():
            log.info("session_refreshed_via_cron")
    except Exception as exc:  # noqa: BLE001
        log.error("session_refresh_cron_failed", extra={"error": str(exc)})


def _start_cron() -> None:
    scheduler.add_job(
        _run_premarket,
        CronTrigger(hour=PREMARKET_CRON_HOUR_IST, minute=PREMARKET_CRON_MINUTE_IST),
        id="premarket",
        replace_existing=True,
    )
    scheduler.add_job(
        _run_premarket_refresh,
        CronTrigger(
            hour=f"{PREMARKET_REFRESH_CRON_HOUR_IST}-15",
            minute=f"*/{PREMARKET_REFRESH_INTERVAL_MINUTES}",
        ),
        id="premarket-refresh",
        replace_existing=True,
    )
    scheduler.add_job(
        _run_eod,
        CronTrigger(hour=EOD_CRON_HOUR_IST, minute=EOD_CRON_MINUTE_IST),
        id="eod",
        replace_existing=True,
    )
    scheduler.add_job(
        _run_session_refresh,
        CronTrigger(hour=SESSION_REFRESH_CRON_HOUR_IST, minute=SESSION_REFRESH_CRON_MINUTE_IST),
        id="session-refresh",
        replace_existing=True,
    )
    scheduler.start()


async def _start_pipeline() -> None:
    """Phase 1: strikes sync + resilient websocket feed -> tick pipeline."""
    global pipeline, _ws_client
    validator = TickValidator()
    pipeline = TickPipeline(
        settings, redis_mgr, validator, health=health, signal_engine=signal_engine
    )

    strikes_mgr = StrikesManager(settings, redis_mgr, build_strikes_provider(settings, session_mgr))
    try:
        strikes_mgr.sync_if_due()
    except Exception as exc:  # noqa: BLE001
        log.error("strikes_initial_sync_failed", extra={"error": str(exc)})

    transport = build_transport(settings, session=session_mgr)
    ws_client = BreezeWebSocketClient(
        settings,
        transport,
        subscriptions_provider=lambda: [settings.nifty_symbol]
        + [f"STK{s}" for s in redis_mgr.get_strikes()],
    )
    _ws_client = ws_client
    ws_client.set_tick_handler(pipeline.process)
    ws_client.set_status_callback(health.set_ws)
    if not isinstance(transport, StubWsTransport):
        backfill = BackfillService(
            settings, redis_mgr, build_snapshot_provider(settings, session_mgr)
        )

        async def _run_backfill() -> None:
            await backfill.run()

        ws_client.set_backfill_callback(_run_backfill)
    health.set_ws(False, 0)
    try:
        await ws_client.run()
    finally:
        health.set_ws(False, ws_client.reconnect_count)
        if _ws_client is ws_client:
            _ws_client = None


async def _restart_pipeline() -> None:
    """Stop the current pipeline (e.g. stub) and rebuild it with the real feed.

    Called after a fresh ICICI session token is pushed via Telegram so the
    transport upgrades from the offline stub to live Breeze.
    """
    global pipeline_task
    async with _pipeline_lock:
        old_task = pipeline_task
        if old_task is not None and not old_task.done():
            old_task.cancel()
            try:
                await old_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 - teardown must not block restart
                log.warning("pipeline_teardown_error", extra={"error": str(exc)})
        pipeline_task = asyncio.create_task(_start_pipeline())
        log.info("pipeline_restarted", extra={"has_token": session_mgr.has_token})


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
    global pipeline_task
    pipeline_task = asyncio.create_task(_start_pipeline())

    def _on_session_updated() -> None:
        try:
            asyncio.create_task(_restart_pipeline())
        except RuntimeError:  # event loop not running (rare teardown)
            log.warning("pipeline_restart_schedule_failed")

    listener = build_telegram_listener(
        settings,
        session_mgr,
        notify=telegram.send_ops,
        on_session_updated=_on_session_updated,
        backtest_runner=_run_backtest_report,
        premarket_report=_premarket_report,
        daily_report=_daily_report,
    )
    listener_task = asyncio.create_task(listener.run())

    async def _debug_task_dump() -> None:
        while True:
            await asyncio.sleep(30)
            try:
                for t in asyncio.all_tasks():
                    if t is asyncio.current_task():
                        continue
                    frames = []
                    try:
                        for fr in t.get_stack():
                            code = getattr(fr, "f_code", None) or getattr(fr, "code", None)
                            if code is None:
                                continue
                            frames.append(f"{code.co_filename}:{fr.f_lineno}:{code.co_name}")
                    except Exception:  # noqa: BLE001 - debug only
                        pass
                    coro_frames: list[str] = []
                    cr = getattr(t, "get_coro", lambda: None)()
                    while cr is not None:
                        gfr: Any = getattr(cr, "cr_frame", None) or getattr(cr, "ag_frame", None)
                        if gfr is not None:
                            coro_frames.append(
                                f"{gfr.f_code.co_filename}:{gfr.f_lineno}:{gfr.f_code.co_name}"
                            )
                        nxt = getattr(cr, "cr_await", None)
                        if nxt is None:
                            nxt = getattr(cr, "ag_await", None)
                        if (
                            nxt is not None
                            and not isinstance(nxt, types.CoroutineType)
                            and not isinstance(nxt, types.AsyncGeneratorType)
                        ):
                            nxt = None
                        cr = nxt
                    log.warning(
                        "debug_task_dump",
                        extra={
                            "task": t.get_name(),
                            "state": t._state,
                            "frames": frames,
                            "coro_chain": coro_frames,
                        },
                    )
            except Exception as exc:  # noqa: BLE001 - debug only
                log.warning("debug_task_dump_failed", extra={"error": str(exc)})

    debug_task = asyncio.create_task(_debug_task_dump())
    _start_cron()
    try:
        yield
    finally:
        if pipeline_task is not None:
            pipeline_task.cancel()
        listener_task.cancel()
        debug_task.cancel()
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
    page["validation_stats"] = vars(pipeline.stats) if pipeline is not None else {}
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
    sent = monitoring.send_text(report)
    return {"report": report, "sent": sent}


@app.post("/ops/ingest-tick")
def ops_ingest_tick(payload: dict) -> dict:
    """Push tick(s) through the live pipeline (validate -> redis -> signal engine).

    Accepts a single tick dict or {"ticks": [tick, ...]}. Used for offline
    replay/tests when the market feed is closed.
    """
    ticks = payload.get("ticks", payload)
    if not isinstance(ticks, list):
        raise HTTPException(status_code=400, detail="expected a tick dict or {'ticks': [...]}")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not started yet")
    accepted = dropped = errors = 0
    for raw in ticks:
        if not isinstance(raw, dict):
            errors += 1
            continue
        try:
            tick = pipeline.process(raw)
        except Exception as exc:  # noqa: BLE001 - never let replay break ingestion
            errors += 1
            log.error("ingest_tick_failed", extra={"error": str(exc)})
            continue
        if tick is None:
            dropped += 1
        else:
            accepted += 1
    return {
        "accepted": accepted,
        "dropped": dropped,
        "errors": errors,
        "validation_stats": vars(pipeline.stats),
        "signal_engine": signal_engine.stats(),
    }


@app.get("/test-signal")
async def test_signal() -> dict:
    """Run the maker->checker workflow on the current live feature snapshot.

    Reads the real feature vector from the live pipeline state (spot, PCR, OI
    velocity, ATR, regime) instead of any synthetic/sample inputs, so it is a
    faithful diagnostic of what the engine would decide right now.
    """
    features = signal_engine.feature_snapshot()
    if not features.get("spot"):
        return {"error": "no live data yet — pipeline still warming up"}
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
