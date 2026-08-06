"""Environment-driven configuration with fail-fast validation.

All credentials are read from environment variables (.env file).
Startup aborts if critical secrets are missing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

from config.constants import get_trigger_profile


class ConfigError(Exception):
    """Raised when configuration is invalid or incomplete."""


REQUIRED_SECRETS = (
    "ICICI_API_KEY",
    "ICICI_API_SECRET",
    "ICICI_SESSION_TOKEN",
    "GEMINI_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_ALERT_CHAT_ID",
)

# Secrets that may be empty in local/dev mode but must be present in production.
PROD_ONLY_REQUIRED = ("QDRANT_API_KEY", "REDIS_PASSWORD")


def _load_env_file() -> None:
    # Project root is one level up from config/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(project_root, ".env"))


def _get_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got: {os.getenv(name)!r}") from None


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except ValueError:
        raise ConfigError(f"{name} must be a float, got: {os.getenv(name)!r}") from None


@dataclass(frozen=True)
class Settings:
    app_env: str = "development"
    log_level: str = "INFO"
    log_json: bool = True
    health_port: int = 8080

    # Breeze
    icici_api_key: str = ""
    icici_api_secret: str = ""
    icici_session_token: str = ""
    nifty_symbol: str = "NIFTY"
    nifty_expiry: str = "weekly"

    # Google Gemini (via OpenAI-compatible endpoint)
    gemini_api_key: str = ""
    embedding_model: str = "gemini-embedding-001"
    llm_model: str = "gemini-3.5-flash"
    llm_temperature: float = 0.3

    # Telegram
    telegram_bot_token: str = ""
    telegram_alert_chat_id: str = ""
    telegram_ops_chat_id: str = ""

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_collection: str = "nifty_historical_traps"

    # Schedule
    market_open_ist: str = "09:15"
    market_close_ist: str = "15:30"

    # Trigger profile
    threshold_profile: str = "MODERATE"

    @property
    def trigger(self) -> dict:
        """Profile-scaled trigger thresholds (computed, not stored)."""
        return get_trigger_profile(self.threshold_profile)

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    def validate(self) -> None:
        skip = _get_bool("SKIP_SECRETS_CHECK", False)
        if not skip:
            missing = [k for k in REQUIRED_SECRETS if not getattr(self, _attr_for(k))]
            if missing:
                raise ConfigError(f"Missing required secrets: {', '.join(missing)}")
        if self.is_production and not skip:
            prod_missing = [k for k in PROD_ONLY_REQUIRED if not getattr(self, _attr_for(k))]
            if prod_missing:
                raise ConfigError(f"Production requires secrets: {', '.join(prod_missing)}")
        if self.telegram_bot_token and not self.telegram_alert_chat_id:
            raise ConfigError("TELEGRAM_ALERT_CHAT_ID is required when bot token is set")


def _attr_for(env_name: str) -> str:
    """Map ICICI_API_KEY -> icici_api_key."""
    return env_name.lower()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load, build, and validate settings once per process."""
    _load_env_file()

    settings = Settings(
        app_env=_get_str("APP_ENV", "development"),
        log_level=_get_str("LOG_LEVEL", "INFO").upper(),
        log_json=_get_bool("LOG_JSON", True),
        health_port=_get_int("HEALTH_PORT", 8080),
        icici_api_key=_get_str("ICICI_API_KEY"),
        icici_api_secret=_get_str("ICICI_API_SECRET"),
        icici_session_token=_get_str("ICICI_SESSION_TOKEN"),
        nifty_symbol=_get_str("NIFTY_SYMBOL", "NIFTY"),
        nifty_expiry=_get_str("NIFTY_EXPIRY", "weekly"),
        gemini_api_key=_get_str("GEMINI_API_KEY"),
        embedding_model=_get_str("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        llm_model=_get_str("GEMINI_LLM_MODEL", "gemini-3.5-flash"),
        llm_temperature=_get_float("LLM_TEMPERATURE", 0.3),
        telegram_bot_token=_get_str("TELEGRAM_BOT_TOKEN"),
        telegram_alert_chat_id=_get_str("TELEGRAM_ALERT_CHAT_ID"),
        telegram_ops_chat_id=_get_str("TELEGRAM_OPS_CHAT_ID"),
        redis_host=_get_str("REDIS_HOST", "redis"),
        redis_port=_get_int("REDIS_PORT", 6379),
        redis_db=_get_int("REDIS_DB", 0),
        redis_password=_get_str("REDIS_PASSWORD"),
        qdrant_host=_get_str("QDRANT_HOST", "qdrant"),
        qdrant_port=_get_int("QDRANT_PORT", 6333),
        qdrant_api_key=_get_str("QDRANT_API_KEY"),
        qdrant_collection=_get_str("QDRANT_COLLECTION", "nifty_historical_traps"),
        market_open_ist=_get_str("MARKET_OPEN_IST", "09:15"),
        market_close_ist=_get_str("MARKET_CLOSE_IST", "15:30"),
        threshold_profile=_get_str("THRESHOLD_PROFILE", "MODERATE").upper(),
    )
    settings.validate()
    return settings
