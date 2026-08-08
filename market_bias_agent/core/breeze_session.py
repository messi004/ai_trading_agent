"""ICICI Breeze session maintenance.

Breeze session tokens expire every ~24h (or at midnight). This module owns a
single :class:`BreezeConnect` client and can:

  * Build it from a configured ``ICICI_SESSION_TOKEN``.
  * Auto-login with ``ICICI_USER_ID`` / ``ICICI_PASSWORD`` /
    ``ICICI_DATE_OF_BIRTH`` via the ``customerlogin`` endpoint.
  * Refresh the token on a schedule before it expires.
  * Accept a fresh token pushed in from Telegram (see
    :mod:`utils.telegram_listener`), persist it to Redis + ``.env`` so a
    restart reuses the newest token.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import time
from typing import Any, cast
from urllib.request import urlopen
from zipfile import ZipFile

import httpx

from config.constants import (
    BREEZE_CUSTOMER_LOGIN_URL,
    BREEZE_SESSION_MAX_AGE_SECONDS,
    BREEZE_SESSION_TOKEN_TTL_SECONDS,
    KEY_BREEZE_SESSION_TOKEN,
)
from config.settings import Settings
from core.logger import get_logger
from core.redis_manager import RedisManager

log = get_logger(__name__)

_LOGIN_HEADERS = {"Content-Type": "application/json"}
_LOGIN_TIMEOUT_SECONDS = 30.0

# ICICI moved the security-master download to a ZIP bundle; the old CSV link
# (traderweb.icicidirect.com/.../StockScriptNew.csv) that the bundled SDK
# (breeze-connect==1.0.12) still hits is dead, which breaks generate_session.
# We reimplement get_stock_script_list against the working ZIP and keep the
# old SDK's contract-key format (OPT-<SYM>-<YYYY-MM-DD>-<STRIKE>-<CE|PE>) so
# the rest of the pipeline (strikes provider, transport) is unchanged.
_SECURITY_MASTER_URL = "https://directlink.icicidirect.com/MotherAppMaster/SecurityMaster.zip"
_SECURITY_MASTER_TIMEOUT_SECONDS = 60.0
_NFO_INDEX = 4  # stock_script_dict_list slot for NFO contracts (old SDK layout)

_MASTER_EXPIRY_RE = re.compile(
    r"(\d{1,2})-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{4})"
)
_MASTER_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _convert_expiry(expiry: str) -> str:
    """Normalise the master's '29-Sep-2026' expiry to 'YYYY-MM-DD'."""
    match = _MASTER_EXPIRY_RE.search(expiry.strip().strip('"'))
    if match is None:
        return expiry.strip().strip('"')
    day, month, year = match.groups()
    return f"{year}-{_MASTER_MONTHS[month]:02d}-{int(day):02d}"


def _patched_get_stock_script_list(self: Any) -> list[dict[str, str]]:
    """Download the SecurityMaster ZIP and build the old SDK scrip tables.

    Compatible with breeze-connect 1.0.12's `get_stock_token_value`, which
    looks up:
      * NSE/BSE equity by short name  -> ``stock_script_dict_list[1][name]``
      * NFO options by ``OPT-<SYM>-<YYYY-MM-DD>-<STRIKE>-<CE|PE>``
        -> ``stock_script_dict_list[4][contract]``
    """
    self.stock_script_dict_list = [{}, {}, {}, {}, {}]
    try:
        with urlopen(_SECURITY_MASTER_URL, timeout=_SECURITY_MASTER_TIMEOUT_SECONDS) as resp:  # noqa: S310
            archive = ZipFile(io.BytesIO(resp.read()))
        for file_name in archive.namelist():
            upper = file_name.upper()
            if not file_name.lower().endswith(".txt"):
                continue
            if "FONSE" in upper:
                exchange, idx = "NFO", _NFO_INDEX
            elif "FOBSE" in upper:
                exchange, idx = "BFO", 5
            elif "MCX" in upper:
                exchange, idx = "MCX", 3
            elif "NDX" in upper:
                exchange, idx = "NDX", 2
            elif "NSE" in upper:
                exchange, idx = "NSE", 1
            elif "BSE" in upper:
                exchange, idx = "BSE", 0
            else:
                continue
            with archive.open(file_name) as handle:
                reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8"))
                next(reader, None)  # skip header
                for columns in reader:
                    if len(columns) < 7:
                        continue
                    token = columns[0].strip('"')
                    short = (
                        columns[1].strip('"')
                        if exchange in ("NSE", "BSE")
                        else columns[2].strip('"')
                    )
                    if exchange in ("NSE", "BSE"):
                        self.stock_script_dict_list[idx][short] = token
                        continue
                    if exchange == "NFO":
                        product = columns[3].strip('"').upper()
                        expiry = _convert_expiry(columns[4])
                        if product in ("OPTION", "OPTSTK", "OPTIDX"):
                            right = columns[6].strip('"').upper()
                            if right not in ("CE", "PE"):
                                continue
                            strike = columns[5].strip('"')
                            key = f"OPT-{short}-{expiry}-{strike}-{right}"
                        elif product in ("FUTURE", "FUTSTK", "FUTIDX"):
                            key = f"FUT-{short}-{expiry}"
                        else:
                            continue
                        self.stock_script_dict_list[idx][key] = token
        log.info(
            "breeze_security_master_loaded",
            extra={"nfo_contracts": len(self.stock_script_dict_list[_NFO_INDEX])},
        )
    except Exception as exc:  # noqa: BLE001 - a stale master must not block session
        log.warning("breeze_security_master_failed", extra={"error": str(exc)})
    return self.stock_script_dict_list


def _apply_security_master_patch() -> None:
    """Monkeypatch the SDK's broken get_stock_script_list (idempotent)."""
    try:
        from breeze_connect import BreezeConnect  # type: ignore[import-untyped]
    except ImportError:
        return
    if getattr(BreezeConnect.get_stock_script_list, "_patched", False):
        return
    BreezeConnect.get_stock_script_list = _patched_get_stock_script_list  # type: ignore[method-assign]
    BreezeConnect.get_stock_script_list._patched = True  # type: ignore[attr-defined]


class BreezeSessionError(RuntimeError):
    """Raised when a Breeze session cannot be established or refreshed."""


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _extract_session_token(response: dict[str, Any]) -> str:
    """Pull the session token out of the customerlogin JSON response.

    The endpoint's payload shape has varied across ICICI API versions, so we
    try the common key names defensively.
    """
    success = response.get("Success")
    if isinstance(success, dict):
        for key in ("session_token", "Session", "API_Session", "api_session", "session"):
            value = success.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    if isinstance(success, str):
        return success.strip()
    return ""


def _customer_login_request(
    *,
    url: str,
    api_key: str,
    user_id: str,
    password: str,
    date_of_birth: str,
) -> str:
    """Call the customerlogin endpoint and return the raw session token."""
    body = {
        "password": password,
        "dOB": date_of_birth,
        "iP_ID": "1.1.1.1",
        "appKey": api_key,
        "idirect_Userid": user_id,
        "user_Data": "ALL",
    }
    payload = json.dumps(body, separators=(",", ":"))
    with httpx.Client(timeout=_LOGIN_TIMEOUT_SECONDS) as client:
        response = client.request(
            "GET",
            url,
            content=payload,
            headers=_LOGIN_HEADERS,
        )
    response.raise_for_status()
    data = response.json()
    token = _extract_session_token(data)
    if not token:
        raise BreezeSessionError(
            f"Auto-login returned no session token (status={data.get('Status')}, "
            f"error={data.get('Error')!r})"
        )
    return token


def _write_env_token(token: str) -> None:
    """Persist the fresh session token into ``.env`` for boot-time reuse."""
    env_path = os.path.join(_project_root(), ".env")
    try:
        if not os.path.exists(env_path):
            return
        with open(env_path, encoding="utf-8") as handle:
            lines = handle.readlines()
        found = False
        for index, line in enumerate(lines):
            if line.startswith("ICICI_SESSION_TOKEN="):
                lines[index] = f"ICICI_SESSION_TOKEN={token}\n"
                found = True
                break
        if not found:
            lines.append(f"ICICI_SESSION_TOKEN={token}\n")
        with open(env_path, "w", encoding="utf-8") as handle:
            handle.writelines(lines)
    except OSError as exc:
        log.warning("env_token_write_failed", extra={"error": str(exc)})


class BreezeSessionManager:
    """Owns the BreezeConnect client and keeps its session alive."""

    def __init__(self, settings: Settings, redis: RedisManager | None = None) -> None:
        self._settings = settings
        self._redis = redis
        self._client: Any = None
        self._session_token = settings.icici_session_token
        self._token_fetched_at = 0.0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def has_credentials(self) -> bool:
        return bool(
            self._settings.icici_user_id
            and self._settings.icici_password
            and self._settings.icici_date_of_birth
        )

    @property
    def has_token(self) -> bool:
        return bool(self._session_token)

    @property
    def has_client(self) -> bool:
        return self._client is not None

    @property
    def user_id(self) -> str:
        return str(getattr(self._client, "user_id", "") or "")

    def status(self) -> dict[str, Any]:
        """Status snapshot for /ops/session and logging."""
        return {
            "has_token": bool(self._session_token),
            "has_credentials": self.has_credentials,
            "has_client": self.has_client,
            "user_id": self.user_id,
            "token_age_seconds": max(0.0, time.time() - self._token_fetched_at)
            if self._token_fetched_at
            else None,
        }

    # ------------------------------------------------------------------
    # Session bootstrap / refresh
    # ------------------------------------------------------------------
    def _load_cached_token(self) -> str | None:
        if self._redis is None or self._redis.client is None:
            return None
        try:
            return cast(str | None, self._redis.client.get(KEY_BREEZE_SESSION_TOKEN))
        except Exception as exc:  # noqa: BLE001 - Redis hiccup must not block login
            log.warning("session_cache_read_failed", extra={"error": str(exc)})
            return None

    def _persist_token(self, token: str) -> None:
        if self._redis is not None and self._redis.client is not None:
            try:
                self._redis.client.set(
                    KEY_BREEZE_SESSION_TOKEN, token, ex=BREEZE_SESSION_TOKEN_TTL_SECONDS
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("session_cache_write_failed", extra={"error": str(exc)})
        _write_env_token(token)

    def _build_client(self, token: str) -> Any:
        try:
            from breeze_connect import BreezeConnect  # type: ignore[import-untyped]

            _apply_security_master_patch()
            client = BreezeConnect(api_key=self._settings.icici_api_key)
            client.generate_session(
                api_secret=self._settings.icici_api_secret,
                session_token=token,
            )
        except Exception as exc:  # noqa: BLE001 - SDK raises generic exceptions
            raise BreezeSessionError(f"Breeze generate_session failed: {exc}") from exc
        if not getattr(client, "user_id", None):
            raise BreezeSessionError("Breeze session rejected: token invalid or expired")
        self._client = client
        self._session_token = token
        self._token_fetched_at = time.time()
        self._persist_token(token)
        log.info("breeze_session_ready", extra={"user_id": client.user_id})
        return client

    def auto_login(self) -> str:
        """Login with ICICI credentials and return the fresh session token."""
        if not self.has_credentials:
            raise BreezeSessionError(
                "ICICI_USER_ID / ICICI_PASSWORD / ICICI_DATE_OF_BIRTH not configured"
            )
        token = _customer_login_request(
            url=BREEZE_CUSTOMER_LOGIN_URL,
            api_key=self._settings.icici_api_key,
            user_id=self._settings.icici_user_id,
            password=self._settings.icici_password,
            date_of_birth=self._settings.icici_date_of_birth,
        )
        self._build_client(token)
        return token

    def get_client(self) -> Any:
        """Return a ready BreezeConnect client (bootstrap / reuse).

        Falls back to a Redis-cached token if the process-level one is empty.
        """
        if self._client is not None:
            return self._client
        token = self._session_token or self._load_cached_token() or ""
        if not token:
            if self.has_credentials:
                return self._build_client(self.auto_login())
            raise BreezeSessionError(
                "No ICICI session token configured and no auto-login credentials"
            )
        return self._build_client(token)

    def update_session_token(self, token: str) -> None:
        """Apply a fresh token (e.g. from Telegram) and rebuild the client."""
        token = token.strip()
        if not token:
            raise BreezeSessionError("Empty session token")
        if self._client is not None and token == self._session_token:
            log.info("breeze_session_token_unchanged")
            return
        self._build_client(token)
        log.info("breeze_session_token_updated")

    def maybe_refresh(self, now: float | None = None) -> bool:
        """Auto-login if the current token is near expiry. Returns True on refresh."""
        if not self.has_credentials:
            return False
        current = now if now is not None else time.time()
        if (
            self._token_fetched_at
            and (current - self._token_fetched_at) < BREEZE_SESSION_MAX_AGE_SECONDS
        ):
            return False
        log.warning(
            "breeze_session_refreshing", extra={"age_seconds": current - self._token_fetched_at}
        )
        self.auto_login()
        return True


def build_session_manager(
    settings: Settings, redis: RedisManager | None = None
) -> BreezeSessionManager:
    """Factory: construct a session manager, tolerating an absent Redis."""
    return BreezeSessionManager(settings, redis)
