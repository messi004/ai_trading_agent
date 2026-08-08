"""Unit tests for Breeze session management (auto-login, token handling)."""

from __future__ import annotations

import types
from typing import Any

import pytest

from core.breeze_session import (
    BreezeSessionError,
    BreezeSessionManager,
    _customer_login_request,
    _extract_session_token,
)


class FakeRedisClient:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key: str, value: str, ex=None) -> None:
        self.store[key] = value


class FakeRedisManager:
    def __init__(self) -> None:
        self.client = FakeRedisClient()


def _settings(**overrides) -> types.SimpleNamespace:
    base = dict(
        icici_api_key="key",
        icici_api_secret="secret",
        icici_session_token="",
        icici_user_id="",
        icici_password="",
        icici_date_of_birth="",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_extract_session_token_variants() -> None:
    assert _extract_session_token({"Success": {"session_token": "abc"}}) == "abc"
    assert _extract_session_token({"Success": {"Session": "def"}}) == "def"
    assert _extract_session_token({"Success": "ghi"}) == "ghi"
    assert _extract_session_token({"Success": {}}) == ""
    assert _extract_session_token({}) == ""


def test_has_token_and_has_credentials() -> None:
    mgr = BreezeSessionManager(_settings(icici_session_token="tok"))
    assert mgr.has_token
    assert not mgr.has_credentials

    mgr = BreezeSessionManager(
        _settings(icici_user_id="u", icici_password="p", icici_date_of_birth="01-01-1990")
    )
    assert not mgr.has_token
    assert mgr.has_credentials

    mgr = BreezeSessionManager(_settings())
    assert not mgr.has_token
    assert not mgr.has_credentials


def test_get_client_raises_without_any_credentials() -> None:
    mgr = BreezeSessionManager(_settings())
    with pytest.raises(BreezeSessionError):
        mgr.get_client()


def test_update_session_token_rebuilds_client(monkeypatch) -> None:
    manager = BreezeSessionManager(_settings(icici_session_token="old"))
    client = types.SimpleNamespace(user_id="demo", generate_session=lambda **_: None)

    class FakeModule:
        class BreezeConnect:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key
                self.user_id = None

            def generate_session(self, api_secret, session_token) -> None:
                self.user_id = "demo"

    monkeypatch.setitem(__import__("sys").modules, "breeze_connect", FakeModule())

    def fake_build(token: str) -> Any:
        client.user_id = "demo"
        manager._client = client  # noqa: SLF001
        manager._session_token = token
        return client

    manager._build_client = fake_build  # type: ignore[method-assign]
    manager.update_session_token("fresh-token")
    assert manager.has_client
    assert manager._session_token == "fresh-token"


def test_redis_token_cached_and_persisted(monkeypatch) -> None:
    redis = FakeRedisManager()
    # Never touch the real .env from tests.
    monkeypatch.setattr("core.breeze_session._write_env_token", lambda token: None)
    manager = BreezeSessionManager(_settings(), redis)  # type: ignore[arg-type]
    manager._session_token = "tok"
    manager._persist_token("tok")
    assert redis.client.get("breeze_session_token") == "tok"
    assert manager._load_cached_token() == "tok"


def test_maybe_refresh_only_when_due(monkeypatch) -> None:
    manager = BreezeSessionManager(
        _settings(icici_user_id="u", icici_password="p", icici_date_of_birth="01-01-1990")
    )
    manager._token_fetched_at = 100.0
    monkeypatch.setattr(manager, "auto_login", lambda: "tok")
    # Under max age -> no refresh
    assert manager.maybe_refresh(now=100.0) is False
    # Past max age -> refresh
    assert manager.maybe_refresh(now=100.0 + 23 * 3600) is True


def test_customer_login_request_sends_expected_payload(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"Success": {"session_token": "fresh"}}

    class FakeClient:
        def __init__(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def request(self, method, url, content=None, headers=None):
            captured["method"] = method
            captured["url"] = url
            captured["content"] = content
            return FakeResponse()

    import core.breeze_session as bs

    monkeypatch.setattr(bs.httpx, "Client", lambda timeout=None: FakeClient())
    token = _customer_login_request(
        url="https://login.test",
        api_key="key",
        user_id="user",
        password="pass",
        date_of_birth="01-01-1990",
    )
    assert token == "fresh"
    assert captured["method"] == "GET"
    assert "pass" in captured["content"]
    assert "user" in captured["content"]
