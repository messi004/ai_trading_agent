"""Unit tests for the Telegram session listener (/session, /status, /backtest commands)."""

from __future__ import annotations

import asyncio
import types

from utils.telegram_listener import (
    TelegramSessionListener,
    _extract_command,
    _parse_backtest_args,
)


class FakeSessionManager:
    def __init__(self) -> None:
        self.updated_token: str | None = None

    def update_session_token(self, token: str) -> None:
        self.updated_token = token

    def status(self) -> dict:
        return {
            "has_token": True,
            "has_credentials": False,
            "has_client": self.updated_token is not None,
            "user_id": "demo",
            "token_age_seconds": 100.0,
        }


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send_ops(self, text: str) -> bool:
        self.sent.append(("ops", text))
        return True


def _listener() -> tuple[TelegramSessionListener, FakeSessionManager, FakeBot]:
    settings = types.SimpleNamespace(
        telegram_bot_token="token",
        telegram_ops_chat_id="12345",
    )
    session = FakeSessionManager()
    bot = FakeBot()
    listener = TelegramSessionListener(settings, session, notify=bot.send_ops)  # type: ignore[arg-type]
    return listener, session, bot


def test_extract_command_with_bot_suffix() -> None:
    assert _extract_command("/session@MyAgent freshtoken") == "/session freshtoken"
    assert _extract_command("/status") == "/status"
    assert _extract_command("hello") == ""
    assert _extract_command("/session") == "/session"


def test_session_command_updates_token_and_replies() -> None:
    listener, session, bot = _listener()
    listener._handle_update({"message": {"text": "/session freshtoken", "chat": {"id": "12345"}}})
    assert session.updated_token == "freshtoken"
    assert bot.sent and "Session updated" in bot.sent[0][1]


def test_session_command_invokes_on_session_updated() -> None:
    settings = types.SimpleNamespace(telegram_bot_token="token", telegram_ops_chat_id="12345")
    session = FakeSessionManager()
    bot = FakeBot()
    called: list[str] = []
    listener = TelegramSessionListener(
        settings,
        session,
        notify=bot.send_ops,
        on_session_updated=lambda: called.append("yes"),
    )  # type: ignore[arg-type]
    listener._handle_update({"message": {"text": "/session fresh", "chat": {"id": "12345"}}})
    assert called == ["yes"]


def test_session_command_without_token_replies_usage() -> None:
    listener, session, bot = _listener()
    listener._handle_update({"message": {"text": "/session", "chat": {"id": "12345"}}})
    assert session.updated_token is None
    assert bot.sent and "Usage" in bot.sent[0][1]


def test_status_command_replies() -> None:
    listener, _session, bot = _listener()
    listener._handle_update({"message": {"text": "/status", "chat": {"id": "12345"}}})
    assert bot.sent and "has_token=True" in bot.sent[0][1]


def test_unknown_chat_is_ignored() -> None:
    listener, session, bot = _listener()
    listener._handle_update({"message": {"text": "/session tok", "chat": {"id": "99999"}}})
    assert session.updated_token is None
    assert bot.sent == []


def test_non_command_message_ignored() -> None:
    listener, session, bot = _listener()
    listener._handle_update({"message": {"text": "random chatter", "chat": {"id": "12345"}}})
    assert session.updated_token is None
    assert bot.sent == []


def test_rejected_token_reports_error() -> None:
    settings = types.SimpleNamespace(telegram_bot_token="token", telegram_ops_chat_id="12345")

    class RejectingSession(FakeSessionManager):
        def update_session_token(self, token: str) -> None:
            raise RuntimeError("token expired")

    session = RejectingSession()
    bot = FakeBot()
    listener = TelegramSessionListener(settings, session, notify=bot.send_ops)  # type: ignore[arg-type]
    listener._handle_update({"message": {"text": "/session bad", "chat": {"id": "12345"}}})
    assert bot.sent and "rejected" in bot.sent[0][1]


def test_build_status_without_client() -> None:
    listener, _session, _bot = _listener()
    text = listener._build_status()
    assert "has_token=True" in text
    assert "demo" in text


def test_parse_backtest_args() -> None:
    assert _parse_backtest_args("") == {}
    assert _parse_backtest_args("--walk-forward") == {"walk_forward": True}
    args = _parse_backtest_args("--sl 4 --target 6 --max-hold 20 --walk-forward")
    assert args == {
        "sl_points": 4.0,
        "target_points": 6.0,
        "max_hold_bars": 20,
        "walk_forward": True,
    }
    assert _parse_backtest_args("--days 7") == {"days": 7}
    assert _parse_backtest_args("--months 3 --years 1") == {"months": 3, "years": 1}
    assert _parse_backtest_args("--days 0 --years -2") == {}
    assert _parse_backtest_args("--days abc") == {}


def test_backtest_command_runs_runner_and_replies() -> None:
    settings = types.SimpleNamespace(telegram_bot_token="token", telegram_ops_chat_id="12345")
    session = FakeSessionManager()
    bot = FakeBot()

    def runner(**kwargs):
        return f"report: {kwargs}"

    listener = TelegramSessionListener(
        settings, session, notify=bot.send_ops, backtest_runner=runner
    )  # type: ignore[arg-type]

    async def scenario() -> None:
        listener._handle_update(
            {"message": {"text": "/backtest --days 7", "chat": {"id": "12345"}}}
        )
        # let the created task finish
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert bot.sent and "Running backtest" in bot.sent[0][1]
    assert bot.sent and "report:" in bot.sent[-1][1]
    assert bot.sent and "'days': 7" in bot.sent[-1][1]


def test_backtest_command_no_args_shows_menu() -> None:
    listener, _session, bot = _listener()
    listener._handle_update({"message": {"text": "/backtest", "chat": {"id": "12345"}}})
    assert bot.sent and "Backtest" in bot.sent[0][1]


def test_backtest_command_unconfigured_replies_hint() -> None:
    listener, _session, bot = _listener()
    listener._handle_update({"message": {"text": "/backtest --days 7", "chat": {"id": "12345"}}})
    assert bot.sent and "not configured" in bot.sent[0][1]


def test_backtest_reports_value_error_to_operator() -> None:
    settings = types.SimpleNamespace(telegram_bot_token="token", telegram_ops_chat_id="12345")
    session = FakeSessionManager()
    bot = FakeBot()

    def runner(**kwargs):
        raise ValueError("no candles for NIFTY — ingest first")

    listener = TelegramSessionListener(
        settings, session, notify=bot.send_ops, backtest_runner=runner
    )  # type: ignore[arg-type]

    async def scenario() -> None:
        listener._handle_update(
            {"message": {"text": "/backtest --days 7", "chat": {"id": "12345"}}}
        )
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert bot.sent and "no candles" in bot.sent[-1][1]


def _fake_http_client(listener: TelegramSessionListener) -> None:
    """Short-circuit the real Bot API client so tests stay hermetic."""
    listener._client.post = lambda *_args, **_kwargs: types.SimpleNamespace(  # type: ignore[method-assign]
        status_code=200, json=lambda: {"ok": True}
    )


def test_start_command_shows_menu() -> None:
    listener, _session, bot = _listener()
    listener._handle_update({"message": {"text": "/start", "chat": {"id": "12345"}}})
    assert bot.sent and "AI Trading Agent" in bot.sent[0][1]


def test_premarket_command_replies_report() -> None:
    settings = types.SimpleNamespace(telegram_bot_token="token", telegram_ops_chat_id="12345")
    session = FakeSessionManager()
    bot = FakeBot()
    listener = TelegramSessionListener(
        settings, session, notify=bot.send_ops, premarket_report=lambda: "premarket levels"
    )  # type: ignore[arg-type]
    listener._handle_update({"message": {"text": "/premarket", "chat": {"id": "12345"}}})
    assert bot.sent and "premarket levels" in bot.sent[-1][1]


def test_daily_command_replies_report() -> None:
    settings = types.SimpleNamespace(telegram_bot_token="token", telegram_ops_chat_id="12345")
    session = FakeSessionManager()
    bot = FakeBot()
    listener = TelegramSessionListener(
        settings, session, notify=bot.send_ops, daily_report=lambda: "daily ops report"
    )  # type: ignore[arg-type]
    listener._handle_update({"message": {"text": "/daily", "chat": {"id": "12345"}}})
    assert bot.sent and "daily ops report" in bot.sent[-1][1]


def test_callback_backtest_range_runs_runner() -> None:
    settings = types.SimpleNamespace(telegram_bot_token="token", telegram_ops_chat_id="12345")
    session = FakeSessionManager()
    bot = FakeBot()
    calls: list[dict] = []

    def runner(**kwargs):
        calls.append(kwargs)
        return f"done: {kwargs}"

    listener = TelegramSessionListener(
        settings, session, notify=bot.send_ops, backtest_runner=runner
    )  # type: ignore[arg-type]
    _fake_http_client(listener)

    async def scenario() -> None:
        listener._handle_update(
            {
                "callback_query": {
                    "id": "q1",
                    "data": "bt:range:7d",
                    "message": {"chat": {"id": "12345"}},
                }
            }
        )
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert calls == [{"days": 7}]
    assert bot.sent and "done:" in bot.sent[-1][1]


def test_callback_backtest_menu_replies_menu() -> None:
    listener, _session, bot = _listener()
    _fake_http_client(listener)
    listener._handle_update(
        {"callback_query": {"id": "q1", "data": "bt:menu", "message": {"chat": {"id": "12345"}}}}
    )
    assert bot.sent and "Backtest" in bot.sent[0][1]


def test_callback_command_status_replies() -> None:
    listener, _session, bot = _listener()
    _fake_http_client(listener)
    listener._handle_update(
        {"callback_query": {"id": "q1", "data": "cmd:status", "message": {"chat": {"id": "12345"}}}}
    )
    assert bot.sent and "has_token=True" in bot.sent[0][1]


def test_callback_menu_main_shows_menu() -> None:
    listener, _session, bot = _listener()
    _fake_http_client(listener)
    listener._handle_update(
        {"callback_query": {"id": "q1", "data": "menu:main", "message": {"chat": {"id": "12345"}}}}
    )
    assert bot.sent and "AI Trading Agent" in bot.sent[0][1]


def test_callback_unknown_chat_ignored() -> None:
    listener, session, bot = _listener()
    _fake_http_client(listener)
    listener._handle_update(
        {
            "callback_query": {
                "id": "q1",
                "data": "bt:range:7d",
                "message": {"chat": {"id": "99999"}},
            }
        }
    )
    assert bot.sent == []
