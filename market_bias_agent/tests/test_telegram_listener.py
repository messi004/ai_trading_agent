"""Unit tests for the Telegram session listener (/session, /status commands)."""

from __future__ import annotations

import types

from utils.telegram_listener import TelegramSessionListener, _extract_command


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
