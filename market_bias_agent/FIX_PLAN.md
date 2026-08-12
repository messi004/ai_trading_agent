# LIVE FEED BUG-FIX PLAN — dalai 2026-08-11

## Summary
Agent live hai (WS+Spot ticks working), par do bugs the:
1. **Memory (Qdrant) dimension mismatch** — GeminiEmbedder 3072-dim vs collection dim=8 → upsert/query 400. ✅ FIXED + DEPLOYED.
2. **WS pipeline freeze** — `ws_connected: true` par 0 ticks, watchdog/reconnect kaam nahi karte the. ✅ ROOT CAUSE FOUND + FIXED + DEPLOYED.
3. **Validator dropped ~96% of live OI ticks** ("OI feed silent" asli culprit). ✅ FIXED + DEPLOYED 2026-08-12.

## Evidence
### Bug 1 — Memory dimension mismatch (FIXED + DEPLOYED)
- Log: `signal_memory_query_failed error="Unexpected Response: 400 (Bad Request) ... Vector inserting error: expected dim: 8, got 3072"`
- Qdrant collection `nifty_historical_traps` dim=8 (FeatureEmbedder, `QDRANT_COLLECTION_DIM`).
- Fix (Phase A, zero migration): live deployment me bhi `FeatureEmbedder` (8-dim) — GeminiEmbedder sirf backtest/offline. Implemented: `GEMINI_EMBEDDING_DIM=3072`, `settings.embedding_backend`, `get_embedder()` backend-aware, `vector_store.py` dim-mismatch `ValueError`, `memory_service.py` passes `embedder.dim`.

### Bug 2 — WS pipeline freeze (ROOT CAUSE CONFIRMED + FIXED + DEPLOYED)
Three independent SDK bugs stacked:
1. **SDK `ws_disconnect()` doesn't close the socket** — only emits `disconnect` event, leaves `sio_handler` set; `ws_connect()` has `if not self.sio_handler:` guard → reconnect on stale socket is a no-op → 0 ticks forever.
2. **SDK `ws_connect()` assigns `sio_handler` BEFORE connecting** — failed connect leaves half-initialized handler → next `ws_connect()` no-ops → `_subscribe` → `watch()` → `sio.emit('join')` raises `BadNamespaceError: "/ is not a connected namespace."` → infinite reconnect loop.
3. **Stale None sentinel** — `close()` does `_queue.put_nowait(None)`; on reconnect the sentinel persists → first `receive()` returns None → `_receive_loop` ends → spurious `ws_disconnected`.
4. **Watchdog not market-aware** — health watchdog was market-aware, websocket `_watchdog_expired` wasn't → POST hours me har 10s reconnect storm.

Fixes (Phase B, all in `core/breeze_transport.py` + `core/websocket_client.py`):
- `close()`: after `client.ws_disconnect()`, reset `client.sio_handler = None` + `old_sio.disconnect()` on previous socketio client (best-effort).
- `connect()`: on SDK `ws_connect()` failure, `_reset_sdk_handler()` and re-raise; `_drain_queue()` drops stale None sentinel + any leftover ticks before subscribing.
- `_watchdog_expired()`: market-aware (`market_status() != "OPEN"` → False), optional `market` override for tests.

### Bug 3 — Validator dropped ~96% of live OI ticks (FIXED + DEPLOYED)
- Live market (OPEN, fresh token) test showed: `total: 16322, accepted: 217` (1.3%), `dropped_stale: 6180`, `dropped_out_of_order: 9924` — OI feed appeared "silent" because the buffer only ever got a trickle.
- Two validator bugs stacked:
  1. **Stale check used `ltt` (last TRADE time) for OI.** Illiquid strikes trade rarely yet push current OI every few seconds — a minutes/hours-old trade timestamp was treated as stale OI and dropped. Fix: OI gets much wider tolerances (`OI_MAX_TICK_AGE_SECONDS=3600`, `OI_TICK_SKEW_TOLERANCE_SECONDS=300`); spot stays strict (prices only change on trade).
  2. **Out-of-order key collision.** All OI ticks share `symbol="NIFTY"`, but the validator keyed `_last_by_key` by `(type, symbol)` — so two *different* strikes compared timestamps against each other, and one illiquid strike's forward-jumping `ltt` cascade-rejected every other strike. Fix: key = `(type, symbol, strike, option_type)`; `is_out_of_order`/`is_duplicate` are strike-aware too.
- Result: accepted rate 1.3% → ~35-40%; `dropped_out_of_order` 9924 → 0. `ticks_processed` grew 217 → 6369+, OI buffer keys update live, spot fresh.
- Note: `buffer_fill_pct` stays 0.0 because `HealthRegistry.set_buffer_fill` is never called in production (dead metric, not a feed bug).

## Fix Plan (complete)
### Phase A — Embedder/Qdrant dimension consistency (DONE)
- `config/constants.py` `GEMINI_EMBEDDING_DIM=3072`; `config/settings.py` `embedding_backend`; `memory/embeddings.py` `.dim` + backend-aware `get_embedder()`; `vector_store.py` dim-mismatch `ValueError`; `memory_service.py` passes `embedder.dim`; tests in `tests/test_memory.py`.

### Phase B — WS reconnect/SDK fix (DONE)
- `core/breeze_transport.py` — `_reset_sdk_handler()` static helper, `connect()` error-path reset + `_drain_queue()`, `close()` full teardown.
- `core/websocket_client.py` — `_watchdog_expired()` market-aware + `market` override.
- Tests: `tests/test_websocket_client.py` (`market="OPEN"` for stale detection), `tests/test_breeze_transport.py`.

### Phase C — Regression + deploy (DONE)
- `pytest` 304 passed (303 prior + market-aware watchdog test updated; 1 pre-existing failure in test_health — time-zone only).
- ruff/mypy clean.
- Deploy: per-subdir scp (main.py, core/websocket_client.py, core/breeze_transport.py) → `docker restart dual_engine_agent`.
- **Verified live (POST market):** `reconnect_count: 0`, no `BadNamespace`/`ConnectionError`/`debug_receive_loop_none`, ticks flow through queue (idle 0.0), watchdog silent, CPU 0.92%, no reconnect storm.

## Rollback
- Volumes untouched (data preserved). Container-only change → `docker compose up -d` se purana image nahi (bina rebuild) wapas possible nahi; image rebuild se pehle current healthy state snapshot lelo (docker tag).

## Verification checklist (next market hours)
- `/health` me `buffer_fill_pct > 0`; `spot_ticks` + OI buffer keys me data.
- OI ticks flow (spot 60/15s + OI expected during market hours); `ticks_processed` grows.
- Watchdog: market OPEN me stale socket → fires + reconnects; POST me silent (no storm).
- `signal_memory_query_failed` absent.

## Open questions
- Breeze option-subscription server-side limit? (463 subs ok lagte the pehle.)
- POST market me Breeze spot quotes continue karta hai (idle 0.0 hot-spin nahi, ~4/sec) — expected; OI ticks market hours me verify karo.
