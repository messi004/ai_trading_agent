"""Telegram command listener + inline-button menu.

Breeze session tokens expire ~daily. This module long-polls the Bot API
``getUpdates`` endpoint and accepts:

  * ``/session <token>`` — push a fresh ICICI Breeze session token; the
    session manager rebuilds its Breeze client and persists the token to
    Redis + ``.env`` so a restart reuses it.
  * ``/status`` — reply with a short session/health summary.
  * ``/premarket`` — the latest premarket levels (if configured).
  * ``/daily`` — the daily ops report (if configured).
  * ``/backtest [--walk-forward] [--days N] [--months N] [--years N]`` —
    run the real-data backtest in a worker thread and reply with the report.
  * ``/start`` | ``/help`` — inline-button menu.

Inline buttons use ``callback_query`` updates so the operator can trigger
actions (menu navigation, quick backtest ranges) without typing commands.

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

MENU_CALLBACK = "menu:main"
BACKTEST_RANGE_OPTIONS: list[tuple[str, str]] = [
    ("7d", "7 days"),
    ("30d", "30 days"),
    ("90d", "90 days"),
    ("1y", "1 year"),
    ("all", "All history"),
]


def _main_menu_buttons() -> list[list[dict]]:
    return [
        [
            {"text": "📊 Backtest", "callback_data": "bt:menu"},
            {"text": "📈 Status", "callback_data": "cmd:status"},
        ],
        [
            {"text": "🌅 Premarket", "callback_data": "cmd:premarket"},
            {"text": "📋 Daily", "callback_data": "cmd:daily"},
        ],
        [
            {"text": "🔄 Session", "callback_data": "cmd:session"},
            {"text": "❓ Help", "callback_data": "menu:help"},
        ],
    ]


def _backtest_range_buttons() -> list[list[dict]]:
    def button(key: str, label: str) -> dict:
        return {"text": label, "callback_data": f"bt:range:{key}"}

    return [
        [button(key, label) for key, label in BACKTEST_RANGE_OPTIONS[:2]],
        [button(key, label) for key, label in BACKTEST_RANGE_OPTIONS[2:4]],
        [button(key, label) for key, label in BACKTEST_RANGE_OPTIONS[4:]]
        + [{"text": "⬅️ Menu", "callback_data": "menu:main"}],
    ]


class TelegramSessionListener:
    def __init__(
        self,
        settings: Settings,
        session_manager: BreezeSessionManager,
        notify: Any = None,
        on_session_updated: Any = None,
        backtest_runner: Any = None,
        premarket_report: Any = None,
        daily_report: Any = None,
    ) -> None:
        self._settings = settings
        self._session = session_manager
        self._notify = notify  # optional callable(text) -> bool for replies
        self._on_session_updated = on_session_updated  # optional callback when a token lands
        self._backtest_runner = backtest_runner  # optional async worker for /backtest
        self._premarket_report = premarket_report  # optional callable() -> str
        self._daily_report = daily_report  # optional callable() -> str
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
                    "allowed_updates": '["message", "callback_query"]',
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
        if "callback_query" in update:
            self._handle_callback(update["callback_query"])
            return
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
        elif command == "/backtest":
            self._handle_backtest(argument.strip(), chat_id)
        elif command in ("/premarket",):
            self._handle_premarket(chat_id)
        elif command in ("/daily", "/report"):
            self._handle_daily(chat_id)
        elif command in ("/start", "/help", "/menu"):
            self._reply_menu(chat_id, menu="main")

    # ------------------------------------------------------------------
    # Inline-button callbacks
    # ------------------------------------------------------------------
    def _handle_callback(self, callback: dict) -> None:
        data = str(callback.get("data") or "")
        message = callback.get("message") or {}
        chat_id = str((message.get("chat") or {}).get("id") or "")
        query_id = str(callback.get("id") or "")
        if not chat_id or not query_id:
            return
        if self._ops_chat_id and chat_id != self._ops_chat_id:
            return
        try:
            self._client.post(
                TELEGRAM_API.format(token=self._token, method="answerCallbackQuery"),
                json={"callback_query_id": query_id},
            )
        except httpx.HTTPError as exc:
            log.warning("telegram_callback_ack_error", extra={"error": str(exc)})

        if data.startswith("bt:"):
            self._handle_callback_backtest(data, chat_id)
        elif data.startswith("cmd:"):
            self._handle_callback_command(data, chat_id)
        elif data.startswith("menu:"):
            self._reply_menu(chat_id, menu=data.split(":", 1)[1])

    def _handle_callback_command(self, data: str, chat_id: str) -> None:
        command = data.split(":", 1)[1]
        if command == "status":
            self._reply(self._build_status(), chat_id)
        elif command == "premarket":
            self._handle_premarket(chat_id)
        elif command == "daily":
            self._handle_daily(chat_id)
        elif command == "session":
            self._reply(
                "Send the token with: /session &lt;token&gt;\n"
                "Get it from https://api.icicidirect.com/apiuser/home",
                chat_id,
            )

    def _handle_callback_backtest(self, data: str, chat_id: str) -> None:
        _, kind, *rest = data.split(":")
        if kind == "menu":
            self._reply_menu(chat_id, menu="backtest")
            return
        if kind == "range" and rest:
            key = rest[0]
            args: dict[str, Any] = {}
            if key == "7d":
                args["days"] = 7
            elif key == "30d":
                args["days"] = 30
            elif key == "90d":
                args["days"] = 90
            elif key == "1y":
                args["years"] = 1
            elif key != "all":
                return
            self._start_backtest_task(args, chat_id)

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

    def _handle_backtest(self, argument: str, chat_id: str) -> None:
        """Run the backtest in a worker thread and reply with the report."""
        kwargs = _parse_backtest_args(argument)
        if not kwargs:
            # No args -> show the range menu so the operator picks a window.
            self._reply_menu(chat_id, menu="backtest")
            return
        self._start_backtest_task(kwargs, chat_id)

    def _start_backtest_task(self, kwargs: dict[str, Any], chat_id: str) -> None:
        if self._backtest_runner is None:
            self._reply("Backtest runner not configured on this instance.", chat_id)
            return
        self._reply("⏳ Running backtest on real ingested data… (may take a minute)", chat_id)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._reply("Backtest command needs the async event loop.", chat_id)
            return
        loop.create_task(self._run_backtest(kwargs, chat_id))

    async def _run_backtest(self, kwargs: dict[str, Any], chat_id: str) -> None:
        try:
            report = await asyncio.to_thread(self._backtest_runner, **kwargs)
        except ValueError as exc:
            self._reply(str(exc), chat_id)
            return
        except Exception as exc:  # noqa: BLE001 - report any failure to the operator
            log.error("telegram_backtest_failed", extra={"error": str(exc)})
            self._reply(f"Backtest failed: {exc}", chat_id)
            return
        self._reply(report, chat_id)

    # ------------------------------------------------------------------
    # Additional commands + menu
    # ------------------------------------------------------------------
    def _handle_premarket(self, chat_id: str) -> None:
        if self._premarket_report is None:
            self._reply("Premarket report not configured on this instance.", chat_id)
            return
        try:
            text = self._premarket_report()
        except Exception as exc:  # noqa: BLE001 - never crash the listener
            log.error("telegram_premarket_failed", extra={"error": str(exc)})
            self._reply(f"Premarket report failed: {exc}", chat_id)
            return
        self._reply(text, chat_id)

    def _handle_daily(self, chat_id: str) -> None:
        if self._daily_report is None:
            self._reply("Daily report not configured on this instance.", chat_id)
            return
        try:
            text = self._daily_report()
        except Exception as exc:  # noqa: BLE001 - never crash the listener
            log.error("telegram_daily_failed", extra={"error": str(exc)})
            self._reply(f"Daily report failed: {exc}", chat_id)
            return
        self._reply(text, chat_id)

    def _reply_menu(self, chat_id: str, menu: str = "main") -> None:
        if menu == "backtest":
            self._reply_with_menu(
                chat_id,
                "📊 <b>Backtest</b>\nPick a data window, or use "
                "<code>/backtest --days 7 --walk-forward</code>.",
                _backtest_range_buttons(),
            )
        else:
            self._reply_with_menu(
                chat_id,
                "🤖 <b>AI Trading Agent</b>\n"
                "Nifty 50 index-options dual-engine. Choose an action:",
                _main_menu_buttons(),
            )

    def _reply_with_menu(self, chat_id: str, text: str, buttons: list[list[dict]]) -> None:
        # Menus MUST carry reply_markup, so send directly (notify has no keyboard).
        self._send_text(chat_id, text, reply_markup={"inline_keyboard": buttons})

    def _reply(self, text: str, chat_id: str) -> None:
        if self._notify is not None:
            try:
                self._notify(text)
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("telegram_reply_failed", extra={"error": str(exc)})
        self._send_text(chat_id, text)

    def _send_text(self, chat_id: str, text: str, reply_markup: dict | None = None) -> bool:
        url = TELEGRAM_API.format(token=self._token, method="sendMessage")
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            response = self._client.post(url, json=payload)
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


def _parse_backtest_args(argument: str) -> dict[str, Any]:
    """Parse ``/backtest --walk-forward --sl 4 --target 6 --days 7`` style options."""
    kwargs: dict[str, Any] = {}
    tokens = argument.split()
    walk_forward = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "--walk-forward":
            walk_forward = True
        elif token in ("--sl", "--target", "--max-hold"):
            if i + 1 < len(tokens):
                try:
                    value = float(tokens[i + 1])
                except ValueError:
                    value = None
                if value is not None:
                    if token == "--sl":
                        kwargs["sl_points"] = value
                    elif token == "--target":
                        kwargs["target_points"] = value
                    else:
                        kwargs["max_hold_bars"] = int(value)
                i += 1
        elif token in ("--days", "--months", "--years"):
            if i + 1 < len(tokens):
                try:
                    value = int(tokens[i + 1])
                except ValueError:
                    value = None
                if value is not None and value > 0:
                    key = token.lstrip("-")
                    kwargs[key] = value
                i += 1
        i += 1
    if walk_forward:
        kwargs["walk_forward"] = True
    return kwargs


def build_telegram_listener(
    settings: Settings,
    session_manager: BreezeSessionManager,
    notify: Any = None,
    on_session_updated: Any = None,
    backtest_runner: Any = None,
    premarket_report: Any = None,
    daily_report: Any = None,
) -> TelegramSessionListener:
    return TelegramSessionListener(
        settings,
        session_manager,
        notify=notify,
        on_session_updated=on_session_updated,
        backtest_runner=backtest_runner,
        premarket_report=premarket_report,
        daily_report=daily_report,
    )
