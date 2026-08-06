"""Phase 6 tests: health registry, telegram, monitoring, audit trail."""

import fakeredis
import pytest

from config.settings import Settings
from core.health import HealthRegistry
from core.redis_manager import RedisManager
from modules.checker_node import CheckerNode
from modules.monitoring import MonitoringService, status_page
from tests.test_guardrails import make_signal
from utils.telegram_bot import TelegramBot


class FakeTelegram:
    def __init__(self) -> None:
        self.ops_messages: list[str] = []
        self.alert_messages: list[str] = []
        self.sent_ops = 0
        self.sent_alert = 0

    def send_ops(self, text: str) -> bool:
        self.ops_messages.append(text)
        self.sent_ops += 1
        return True

    def send_alert(self, text: str, photo: bytes = b"") -> bool:
        self.alert_messages.append(text)
        self.sent_alert += 1
        return True


def make_health(**kwargs) -> HealthRegistry:
    return HealthRegistry(Settings(), **kwargs)


def make_monitoring(**kwargs) -> tuple[MonitoringService, FakeTelegram]:
    telegram = FakeTelegram()
    service = MonitoringService(Settings(), make_health(), telegram=telegram, **kwargs)
    return service, telegram


class TestHealthRegistry:
    def test_tick_tracking(self) -> None:
        health = make_health()
        assert health.last_tick_age_seconds() is None
        health.record_tick()
        assert health.ticks_processed == 1
        assert health.last_tick_age_seconds() is not None

    def test_verdict_counters(self) -> None:
        health = make_health()
        health.record_verdict(True)
        health.record_verdict(False)
        assert health.approved == 1
        assert health.rejected == 1

    def test_buffer_fill_clamped(self) -> None:
        health = make_health()
        health.set_buffer_fill(150)
        assert health.buffer_fill_pct == 100.0
        health.set_buffer_fill(-5)
        assert health.buffer_fill_pct == 0.0

    def test_status_shape(self) -> None:
        health = make_health()
        health.record_tick()
        health.set_ws(True, 2)
        health.record_cron_success("eod")
        status = health.status(redis_connected=True, profile="MODERATE")
        for key in (
            "status",
            "market",
            "ws_connected",
            "reconnect_count",
            "last_tick_age_seconds",
            "ticks_processed",
            "last_cron_success",
            "profile",
        ):
            assert key in status
        assert status["ws_connected"] is True

    def test_daily_summary(self) -> None:
        health = make_health()
        health.record_trigger()
        health.record_verdict(True)
        health.record_verdict(False)
        health.record_llm_tokens(1200)
        summary = health.daily_summary()
        assert summary["triggers"] == 1
        assert summary["approved"] == 1
        assert summary["rejected"] == 1
        assert summary["rejection_rate"] == pytest.approx(1.0)
        assert summary["llm_tokens_spent"] == 1200

    def test_watchdog_not_expired_when_closed(self) -> None:
        health = make_health()
        health.record_tick()
        assert health.watchdog_expired(idle_seconds=0.01, market="POST") is False

    def test_watchdog_expired_when_open_and_stale(self) -> None:
        health = make_health()
        health.last_tick_ts = 0.0  # simulate old tick
        health.ticks_processed = 1
        assert health.watchdog_expired(idle_seconds=1.0, market="OPEN") is True

    def test_watchdog_not_expired_when_fresh(self) -> None:
        health = make_health()
        health.record_tick()
        assert health.watchdog_expired(idle_seconds=3600.0, market="OPEN") is False


class TestTelegramBot:
    def test_no_token_degrades_gracefully(self) -> None:
        bot = TelegramBot(Settings(telegram_bot_token=""))
        assert bot.send_text("123", "hello") is False
        assert bot.send_ops("hello") is False
        assert bot.send_alert("hi") is False
        bot.close()

    def test_no_chat_id_fails(self) -> None:
        bot = TelegramBot(Settings(telegram_bot_token="fake"))
        assert bot.send_text(None, "hello") is False
        bot.close()


class TestMonitoring:
    def test_watchdog_silent_outside_market(self) -> None:
        service, telegram = make_monitoring()
        health = service._health
        health.last_tick_ts = 0.0
        health.ticks_processed = 1
        alerted = service.watchdog_check(idle_seconds=0.0)
        assert alerted is False  # market not OPEN at test time -> no alert
        assert telegram.sent_ops == 0

    def test_watchdog_alert_with_open_market(self, monkeypatch) -> None:
        service, telegram = make_monitoring()
        health = service._health
        monkeypatch.setattr(
            health, "watchdog_expired", lambda idle_seconds=300.0, market=None: True
        )
        alerted = service.watchdog_check(idle_seconds=1.0)
        assert alerted is True
        assert telegram.sent_ops == 1
        assert "OPS ALERT" in telegram.ops_messages[0]

    def test_watchdog_silent_when_healthy(self, monkeypatch) -> None:
        service, telegram = make_monitoring()
        monkeypatch.setattr(
            service._health, "watchdog_expired", lambda idle_seconds=300.0, market=None: False
        )
        assert service.watchdog_check() is False
        assert telegram.sent_ops == 0

    def test_daily_report_includes_rejections_by_rule(self) -> None:
        checker = CheckerNode(settings=None)
        checker.check(make_signal(sl=8.0))  # rule A rejection
        checker.check(make_signal(sl=8.0))  # rule A rejection again
        service, telegram = make_monitoring(checker=checker)
        report = service.build_daily_report()
        assert "Daily Ops Report" in report
        assert "Rejections by rule" in report
        assert "A: 2" in report

    def test_send_daily_report(self) -> None:
        service, telegram = make_monitoring()
        assert service.send_daily_report() is True
        assert telegram.sent_ops == 1

    def test_status_page(self) -> None:
        health = make_health()
        page = status_page(
            health, redis_connected=True, profile="CONSERVATIVE", audit_recent=[{"approved": True}]
        )
        assert page["status"] == "ok"
        assert page["profile"] == "CONSERVATIVE"
        assert page["audit_recent"] == [{"approved": True}]


class TestAuditTrail:
    def test_redis_audit_roundtrip(self) -> None:
        mgr = RedisManager(Settings())
        mgr.client = fakeredis.FakeRedis(decode_responses=True)
        mgr.push_audit({"signal_id": "abc", "approved": True})
        mgr.push_audit({"signal_id": "def", "approved": False})
        assert mgr.audit_length() == 2
        recent = mgr.get_audit(count=2)
        assert recent[0]["signal_id"] == "def"  # newest first

    def test_audit_capped(self) -> None:
        mgr = RedisManager(Settings())
        mgr.client = fakeredis.FakeRedis(decode_responses=True)
        for i in range(15):
            mgr.push_audit({"signal_id": str(i)}, maxlen=10)
        assert mgr.audit_length() == 10

    def test_checker_writes_to_audit_writer(self) -> None:
        written: list[dict] = []
        checker = CheckerNode(settings=None)
        checker.set_audit_writer(written.append)
        checker.check(make_signal())
        assert len(written) == 1
        assert written[0]["approved"] is True
        assert written[0]["signal_id"] == checker.audit_trail()[-1]["signal_id"]
