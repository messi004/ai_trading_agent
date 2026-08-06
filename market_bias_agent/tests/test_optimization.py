"""Phase 7 tests: chart cache, LLM cache, media group, Redis TTL/memory."""

import fakeredis

from config.constants import OI_BUFFER_TTL_SECONDS
from config.settings import Settings
from core.llm_cache import LLMCache, cached_maker_output, market_state_key
from core.redis_manager import RedisManager
from utils.chart_generator import ChartCache, generate_oi_chart
from utils.telegram_bot import TelegramBot


class TestChartCache:
    def test_generate_png(self) -> None:
        png = generate_oi_chart({24000: 100, 24100: 90}, {24000: 80, 24100: 110}, spot=24000)
        assert png.startswith(b"\x89PNG")

    def test_empty_data(self) -> None:
        png = generate_oi_chart({}, {})
        assert png.startswith(b"\x89PNG")

    def test_cache_reuses_render(self) -> None:
        cache = ChartCache(max_size=2)
        calls = {"n": 0}

        def render():
            calls["n"] += 1
            return b"chart-data"

        key = "abc"
        first = cache.get_or_render(key, render)
        second = cache.get_or_render(key, render)
        assert first == second == b"chart-data"
        assert calls["n"] == 1
        assert cache.hit_rate == 0.5

    def test_cache_eviction_lru(self) -> None:
        cache = ChartCache(max_size=2)
        calls = {"n": 0}

        def render():
            calls["n"] += 1
            return b"chart"

        cache.get_or_render("a", render)
        cache.get_or_render("b", render)
        cache.get_or_render("c", render)  # evicts "a" (LRU)
        assert cache.get_or_render("a", render) == b"chart"  # re-renders
        assert calls["n"] == 4  # a:2, b:1, c:1

    def test_chart_cache_png(self) -> None:
        cache = ChartCache()
        call = {24000: 100}
        put = {24000: 90}
        first = generate_oi_chart(call, put, spot=24000, cache=cache)
        second = generate_oi_chart(call, put, spot=24000, cache=cache)
        assert first == second
        assert cache.hit_rate == 0.5


class TestLLMCache:
    def test_market_state_key_stable_and_rounded(self) -> None:
        a = market_state_key({"pcr": 0.951, "spot": 24005.411})
        b = market_state_key({"pcr": 0.950, "spot": 24005.409})
        c = market_state_key({"pcr": 0.6, "spot": 23999.0})
        assert a == b  # rounding to 2 decimals groups near-identical states
        assert a != c

    def test_cache_hit_skips_producer(self) -> None:
        cache = LLMCache()
        calls = {"n": 0}

        def producer():
            calls["n"] += 1
            return {"direction": "BULLISH"}

        out1 = cached_maker_output(cache, {"pcr": 0.95}, producer)
        out2 = cached_maker_output(cache, {"pcr": 0.95}, producer)
        assert out1 == out2 == {"direction": "BULLISH"}
        assert calls["n"] == 1
        assert cache.hit_rate == 0.5

    def test_cache_ttl_expiry(self) -> None:
        cache = LLMCache(ttl_seconds=0.01)
        cache.put("k", "v")
        assert cache.get("k") == "v"
        import time

        time.sleep(0.02)
        assert cache.get("k") is None

    def test_cache_eviction(self) -> None:
        cache = LLMCache(max_size=2)
        for i in range(5):
            cache.put(str(i), i)
        assert cache.size == 2


class TestTelegramBatching:
    def test_media_group_without_token_returns_false(self) -> None:
        bot = TelegramBot(Settings(telegram_bot_token=""))
        assert bot.send_media_group("123", [(b"png", "caption")]) is False
        bot.close()

    def test_media_group_no_chat(self) -> None:
        bot = TelegramBot(Settings(telegram_bot_token="fake"))
        assert bot.send_media_group(None, [(b"png", "caption")]) is False
        bot.close()

    def test_media_group_empty(self) -> None:
        bot = TelegramBot(Settings(telegram_bot_token="fake"))
        assert bot.send_media_group("123", []) is False
        bot.close()


class TestRedisTTLAndMemory:
    def test_oi_buffer_ttl_set_on_first_write(self) -> None:
        mgr = RedisManager(Settings())
        mgr.client = fakeredis.FakeRedis(decode_responses=True)
        mgr.push_call_oi(24000, 100.0)
        ttl = mgr.client.ttl("call_oi_strike_24000")
        assert 0 < ttl <= OI_BUFFER_TTL_SECONDS

    def test_ttl_not_reset_on_every_write(self) -> None:
        mgr = RedisManager(Settings())
        mgr.client = fakeredis.FakeRedis(decode_responses=True)
        mgr.push_call_oi(24000, 100.0)
        first = mgr.client.ttl("call_oi_strike_24000")
        mgr.push_call_oi(24000, 101.0)
        second = mgr.client.ttl("call_oi_strike_24000")
        assert second <= first

    def test_memory_usage(self) -> None:
        mgr = RedisManager(Settings())
        mgr.client = fakeredis.FakeRedis(decode_responses=True)
        mgr.client.set("k", "v")
        assert mgr.dbsize() >= 1
        # fakeredis lacks INFO -> returns None without raising
        assert mgr.memory_usage_bytes() is None
