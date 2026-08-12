"""Telegram bot (PRD Module 5 / Enhancement Phase 6).

Sends text + photo messages via the Bot API with a small retry/backoff.
When no bot token is configured it degrades gracefully to log-only so the
pipeline never crashes in offline/local mode.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from config.constants import TELEGRAM_MEDIA_GROUP_MAX
from config.settings import Settings
from core.logger import get_logger

log = get_logger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 2.0


class TelegramBot:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._token = settings.telegram_bot_token
        self._client = httpx.Client(timeout=10.0)

    def _post(self, method: str, payload: dict) -> bool:
        if not self._token:
            log.warning("telegram_not_configured", extra={"method": method})
            return False
        url = TELEGRAM_API.format(token=self._token, method=method)
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._client.post(url, json=payload)
                if response.status_code == 200 and response.json().get("ok"):
                    return True
                log.warning(
                    "telegram_api_error",
                    extra={"method": method, "status": response.status_code, "attempt": attempt},
                )
            except httpx.HTTPError as exc:
                log.warning(
                    "telegram_http_error",
                    extra={"method": method, "error": str(exc), "attempt": attempt},
                )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
        return False

    def send_text(self, chat_id: str | None, text: str, reply_markup: dict | None = None) -> bool:
        if not chat_id:
            log.warning("telegram_no_chat_id", extra={"text_len": len(text)})
            return False
        payload: dict = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return self._post("sendMessage", payload)

    def send_menu(self, chat_id: str | None, text: str, buttons: list[list[dict]]) -> bool:
        """Send text with an inline keyboard (menu/quick actions)."""
        return self.send_text(chat_id, text, reply_markup={"inline_keyboard": buttons})

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> bool:
        """Acknowledge an inline-button press (clears the spinner on the button)."""
        if not self._token:
            return False
        payload: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
            payload["show_alert"] = False
        try:
            response = self._client.post(
                TELEGRAM_API.format(token=self._token, method="answerCallbackQuery"),
                json=payload,
            )
            return bool(response.status_code == 200 and response.json().get("ok"))
        except httpx.HTTPError as exc:
            log.warning("telegram_callback_answer_error", extra={"error": str(exc)})
            return False

    def send_photo(self, chat_id: str | None, png_bytes: bytes, caption: str = "") -> bool:
        if not chat_id:
            log.warning("telegram_no_chat_id", extra={"photo_bytes": len(png_bytes)})
            return False
        if not self._token:
            return False
        url = TELEGRAM_API.format(token=self._token, method="sendPhoto")
        try:
            response = self._client.post(
                url,
                data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
                files={"photo": ("chart.png", png_bytes, "image/png")},
            )
            return bool(response.status_code == 200 and response.json().get("ok"))
        except httpx.HTTPError as exc:
            log.warning("telegram_photo_error", extra={"error": str(exc)})
            return False

    def send_media_group(
        self,
        chat_id: str | None,
        photos: list[tuple[bytes, str]],
    ) -> bool:
        """Batch up to `TELEGRAM_MEDIA_GROUP_MAX` charts in one request (Phase 7)."""
        if not chat_id or not photos:
            return False
        if not self._token:
            return False
        if len(photos) > TELEGRAM_MEDIA_GROUP_MAX:
            log.warning(
                "telegram_media_group_truncated",
                extra={"count": len(photos), "max": TELEGRAM_MEDIA_GROUP_MAX},
            )
            photos = photos[:TELEGRAM_MEDIA_GROUP_MAX]
        url = TELEGRAM_API.format(token=self._token, method="sendMediaGroup")
        media = []
        for index, (_png, caption) in enumerate(photos):
            media.append(
                {
                    "type": "photo",
                    "media": f"attach://chart_{index}.png",
                    "caption": caption,
                    "parse_mode": "HTML",
                }
            )
        try:
            response = self._client.post(
                url,
                data={"chat_id": chat_id, "media": __import__("json").dumps(media)},
                files={
                    f"chart_{index}.png": (f"chart_{index}.png", png, "image/png")
                    for index, (png, _caption) in enumerate(photos)
                },
            )
            return bool(response.status_code == 200 and response.json().get("ok"))
        except httpx.HTTPError as exc:
            log.warning("telegram_media_group_error", extra={"error": str(exc)})
            return False

    def send_alert(self, text: str, photo: bytes = b"") -> bool:
        chat_id = self._settings.telegram_alert_chat_id or None
        if photo:
            return self.send_photo(chat_id, photo, caption=text)
        return self.send_text(chat_id, text)

    def send_ops(self, text: str) -> bool:
        return self.send_text(self._settings.telegram_ops_chat_id or None, text)

    def close(self) -> None:
        self._client.close()
