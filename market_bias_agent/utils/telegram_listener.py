"""Telegram session command listener.

Breeze session tokens expire ~daily. This module long-polls the Bot API
``getUpdates`` endpoint and accepts:

  * ``/session <token>`` — push a fresh ICICI Breeze session token; the
    session manager rebuilds its Breeze client and persists the token to
    Redis + ``.env`` so a restart reuses it.
  * ``/status`` — reply with a short session/health summary.

Runs as an asyncio task owned by :mod:`main`; degrades to log-only when no
bot token or ops chat id is configured.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from config.constants import TELEGRAM_GETUPDATES_TIMEOUT, TELEGRAM_POLL_INTERVAL_SECONDS
from config.settings import Settings
from core.breeze_session import BreezeSessionManager
from core.logger import get_logger

log = get_logger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
COMMAND_TIMEOUT_SECONDS = 60.0


class TelegramSessionListener:
    def __init__(
        self,
        settings: Settings,
        session_manager: BreezeSessionManager,
        notify: Any = None,
        on_session_updated: Any = None,
    ) -> None:
        self._settings = settings
        self._session = session_manager
        self._notify = notify  # optional callable(text) -> bool for replies
        self._on_session_updated = on_session_updated  # optional callback when a token lands
        self._token = settings.telegram_bot_token
        self._ops_chat_id = settings.telegram_ops_chat_id or ""
        self._client = httpx.Client(timeout=TELEGRAM_GETUPDATES_TIMEOUT + 10)
        self._offset = 0

    # ------------------------------------------------------------------
    # Long-polling
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Poll getUpdates until cancelled."""
        if not self._token or not self._ops_chat_id:
            log.warning("telegram_listener_not_configured")
            return
        log.info("telegram_listener_started")
        while True:
            try:
                updates = await asyncio.to_thread(self._poll_once)
                for update in updates or []:
                    self._handle_update(update)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - polling must survive hiccups
                log.warning("telegram_poll_error", extra={"error": str(exc)})
            await asyncio.sleep(TELEGRAM_POLL_INTERVAL_SECONDS)

    def _poll_once(self) -> list[dict]:
        """One getUpdates call (blocking; run via to_thread)."""
        url = TELEGRAM_API.format(token=self._token, method="getUpdates")
        try:
            response = self._client.get(
                url,
                params={
                    "timeout": TELEGRAM_GETUPDATES_TIMEOUT,
                    "offset": self._offset,
                    "allowed_updates": '["message"]',
                },
            )
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("telegram_getupdates_error", extra={"error": str(exc)})
            return []
        if not data.get("ok"):
            log.warning("telegram_getupdates_api_error", extra={"data": data})
            return []
        updates = data.get("result") or []
        if updates:
            self._offset = updates[-1]["update_id"] + 1
        return updates

    def _handle_update(self, update: dict) -> None:
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = str(chat.get("id") or "")
        if self._ops_chat_id and chat_id != self._ops_chat_id:
            log.info("telegram_command_from_unknown_chat", extra={"chat_id": chat_id})
            return
        text = _extract_command(message.get("text") or "")
        if not text:
            return
        command, _, argument = text.partition(" ")
        command = command.lower()
        if command == "/session":
            self._handle_session(argument.strip(), chat_id)
        elif command == "/status":
            self._reply(self._build_status(), chat_id)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------
    def _handle_session(self, token: str, chat_id: str) -> None:
        if not token:
            self._reply("Usage: /session <breeze session token>", chat_id)
            return
        try:
            self._session.update_session_token(token)
        except Exception as exc:  # noqa: BLE001 - report to operator
            log.error("session_token_rejected", extra={"error": str(exc)})
            self._reply(f"Session token rejected: {exc}", chat_id)
            return
        summary = self._session.status()
        age = summary["token_age_seconds"]
        age_text = f"{age:.0f}s" if age is not None else "n/a"
        self._reply(f"Session updated. user={summary['user_id']} token_age={age_text}", chat_id)
        if self._on_session_updated is not None:
            try:
                self._on_session_updated()
            except Exception as exc:  # noqa: BLE001 - callback must not break the listener
                log.warning("session_updated_callback_failed", extra={"error": str(exc)})

    def _build_status(self) -> str:
        summary = self._session.status()
        return (
            f"Breeze session: has_token={summary['has_token']} "
            f"has_credentials={summary['has_credentials']} "
            f"user={summary['user_id'] or 'n/a'}"
        )

    def _reply(self, text: str, chat_id: str) -> None:
        if self._notify is not None:
            try:
                self._notify(text)
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("telegram_reply_failed", extra={"error": str(exc)})
        self._send_text(chat_id, text)

    def _send_text(self, chat_id: str, text: str) -> bool:
        url = TELEGRAM_API.format(token=self._token, method="sendMessage")
        try:
            response = self._client.post(url, json={"chat_id": chat_id, "text": text})
            return bool(response.status_code == 200 and response.json().get("ok"))
        except httpx.HTTPError as exc:
            log.warning("telegram_reply_http_error", extra={"error": str(exc)})
            return False

    def close(self) -> None:
        self._client.close()


def _extract_command(message: str) -> str:
    """Strip the leading @botname from commands like ``/session@MyBot tok``."""
    if not message.startswith("/"):
        return ""
    head, _, _rest = message.partition(" ")
    if "@" in head:
        return message.replace(head, head.split("@")[0], 1)
    return message


def build_telegram_listener(
    settings: Settings,
    session_manager: BreezeSessionManager,
    notify: Any = None,
    on_session_updated: Any = None,
) -> TelegramSessionListener:
    return TelegramSessionListener(
        settings, session_manager, notify=notify, on_session_updated=on_session_updated
    )
