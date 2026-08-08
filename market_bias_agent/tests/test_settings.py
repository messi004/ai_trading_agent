"""Unit tests for config settings loading + validation."""

import pytest

from config.constants import get_trigger_profile
from config.settings import ConfigError, get_settings


def test_trigger_profiles_scaling() -> None:
    base = get_trigger_profile("MODERATE")
    agg = get_trigger_profile("AGGRESSIVE")
    cons = get_trigger_profile("CONSERVATIVE")
    assert agg["scalp_velocity_1m"] < base["scalp_velocity_1m"] < cons["scalp_velocity_1m"]
    assert base["scalp_velocity_1m"] == 40_000
    assert base["intraday_velocity_5m"] == 150_000


def test_unknown_profile_falls_back_to_moderate() -> None:
    assert get_trigger_profile("BANANAS") == get_trigger_profile("MODERATE")


def _set_dummy_secrets(monkeypatch) -> None:
    monkeypatch.setenv("ICICI_API_KEY", "k")
    monkeypatch.setenv("ICICI_API_SECRET", "s")
    monkeypatch.setenv("ICICI_SESSION_TOKEN", "t")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "b")
    monkeypatch.setenv("TELEGRAM_ALERT_CHAT_ID", "c")


def test_settings_loads_from_env(monkeypatch) -> None:
    _set_dummy_secrets(monkeypatch)
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("REDIS_HOST", "localhost")
    settings = get_settings()
    assert settings.redis_host == "localhost"
    assert settings.trigger["scalp_velocity_1m"] == 40_000


def test_missing_required_secret_raises(monkeypatch) -> None:
    for key in (
        "ICICI_API_KEY",
        "ICICI_API_SECRET",
        "GEMINI_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_ALERT_CHAT_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    # Don't let the local .env re-populate secrets behind the test's back.
    monkeypatch.setattr("config.settings._load_env_file", lambda: None)
    with pytest.raises(ConfigError):
        get_settings.cache_clear()
        get_settings()
    get_settings.cache_clear()


def test_session_token_may_be_empty(monkeypatch) -> None:
    _set_dummy_secrets(monkeypatch)
    monkeypatch.delenv("ICICI_SESSION_TOKEN", raising=False)
    monkeypatch.setattr("config.settings._load_env_file", lambda: None)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.icici_session_token == ""
    get_settings.cache_clear()
