# Project Memory

Dual-Engine Institutional AI Trading Agent (NSE Nifty 50 Index Options). Production-ready build, PRD + ENHANCEMENT_PLAN.md driven.

## Source of truth
- `/home/messi/AI_Trading_Agent/PRD` — master requirements (formulas, thresholds, guardrails)
- `/home/messi/AI_Trading_Agent/ENHANCEMENT_PLAN.md` — phase roadmap + quick-win checklist
- `/home/messi/AI_Trading_Agent/market_bias_agent/` — project root (all code)

## Status (updated 2026-08-09)
- Phases 0-8 complete. **Live Breeze feed WORKING** end-to-end on real ICICI session.
- 301 tests passing; ruff clean; mypy clean (52 source files).
- Git: repo active (`main`), remote `github.com/messi004/ai_trading_agent.git`. Local + VPS both pushed.
- **Deployed on VPS** `/opt/ai_trading_agent/market_bias_agent` — SSH host alias `ssh-vps` (root@187.127.163.181, id_rsa). Same docker-compose, containers: dual_engine_agent + agent_redis + agent_qdrant.
- Golden rule: go-live on real capital only after backtest + paper trading show >=3 weeks positive expectancy.

## LIVE FEED (IMPORTANT — verified today)
- Real ICICI session works via token push: Telegram `/session <token>` (browser login token from `https://api.icicidirect.com/apiuser/home`, ~24h expiry, ~8-10 chars). User: AD320982.
- **Auto-login is DEAD**: ICICI returns 500 `Resource Not Available. Please use browser login`. ICICI_USER_ID/PASSWORD/DOB in `.env` do NOT work anymore. Token must be pushed each day via Telegram.
- **SDK patch (critical)**: bundled `breeze-connect==1.0.12`'s `get_stock_script_list()` downloads a DEAD CSV (traderweb.icicidirect.com/.../StockScriptNew.csv → connection reset). Fix in `core/breeze_session.py`: `_apply_security_master_patch()` + `_patched_get_stock_script_list()` downloads `https://directlink.icicidirect.com/MotherAppMaster/SecurityMaster.zip` and rebuilds old-format tables (`OPT-NIFTY-<YYYY-MM-DD>-<STRIKE>-<CE|PE>`, expiry normalised). Without this patch generate_session fails with `Connection aborted / RemoteDisconnected`.
- `.env` has REAL creds (ICICI, Gemini, Telegram, Redis, Qdrant). `.env` is gitignored; `.env.example` is the template. Do NOT commit `.env`.
- Container runs as user `agent` uid **999** (NOT 1000 despite Dockerfile comment). Host `data/` dir must be writable (was chmod 777'd). If SQLite `readonly database` error → check `data/` permissions.

## Architecture (2-engine pattern)
- **Analyzer**: WebSocket (Breeze) -> OI buffer -> tick pipeline -> LLM Maker (`modules/maker_node.py`) -> structured signal
- **Checker**: `modules/checker_node.py` — Rules A-G (guardrails) -> approved signal
- **Memory**: `memory/` — TrapRecord -> FeatureEmbedder (8-dim) -> Qdrant/MemoryVectorStore -> MemoryService (similar situations + regime boost)
- **Post-trade**: `core/signal_store.py` (SQLite lifecycle) -> `modules/post_analysis.py` (outcome write-back, weekly report, bias correction)
- **EOD**: `modules/eod_engine.py` — live trap collector (SQLite SignalLogStore -> TrapRecord -> Qdrant) + NiftyTrader FII/PRO participant-OI report via `core/participant_oi.py` -> Telegram `send_ops`
- **Premarket**: `modules/premarket_engine.py` (08:30 IST) — pivot S/R, psych levels, max pain; consumed ONLY by `signal_engine._near_level()` -> LLM Maker context + dashboard (NOT the mechanical trigger matrix — half-wired)
- **Observability**: `core/health.py`, `modules/monitoring.py` (watchdog + daily report), Redis audit trail, Telegram alerts, `utils/chart_generator.py` OI charts

## Real-data backtest
- No synthetic data anywhere. `scripts/ingest_history.py --with-oi` fetches real per-strike Breeze NFO history -> parquet (`data_store.save_oi_series`, `NIFTY_oi_1m.parquet`). `HistoricalOIProvider` (modules/replay_engine.py) reads it.
- `scripts/run_backtest.py` CLI: `--symbol --data-dir --sl-points --target-points --max-hold-bars --walk-forward`. Raises SystemExit telling user to ingest if candles/OI missing.
- Requires a live session token (`/session <token>`) to fetch history.

## Verify commands
Run from `/home/messi/AI_Trading_Agent/market_bias_agent`:
- `./.venv/bin/python -m pytest -q` (301 passed)
- `./.venv/bin/ruff format .` and `./.venv/bin/ruff check .`
- `./.venv/bin/python -m mypy config/ core/ utils/ modules/ memory/ scripts/ main.py`
- CLI/offline tools: `SKIP_SECRETS_CHECK=true` env (config/settings.py validate())
- Local container health: `curl localhost:8090/health` (host:8090 via docker-compose port mapping)
- VPS: `ssh ssh-vps`, app at `/opt/ai_trading_agent/market_bias_agent`, same compose commands.

## Key semantics (easy to get wrong)
- `OUTCOMES = ("WIN","LOSS","BE")` = result labels (PnL derived). `EXIT_REASONS = ("TARGET_HIT","SL_HIT","TIME_EXIT","DIRECTION_INVALIDATED")` = exit labels.
- Qdrant memory `historical_outcome`: `OUTCOME_WIN="TARGET_HIT"` / `OUTCOME_LOSS="SL_HIT"` (TRAP_OUTCOMES).
- Volatility regime strings: `CALM / ACTIVE / HIGH_VOL` (not "HIGH"); FeatureEmbedder one-hot at vector indices 5-8.
- `ws_client.reconnect_count` is a **property**, not callable (main.py uses it without parens).
- `StructuredSignal.direction` = `BULLISH/BEARISH/NEUTRAL`; `side()` -> LONG/SHORT; `side_to_direction()` maps.
- Backtest `hit_rate` uses `BacktestResult.outcome_labels` (WIN/LOSS/BE by PnL), not raw exit reasons.
- LLM guardrails: temperature clamp 0.2-0.4, JSON parse retry-once, LLMTokenBudget daily cap, rule-only fallback.
- Checker Rule D is a daily-loss halt circuit (`record_exit_pnl`); Rule E = signal rate + strike cooldown.
- Python 3.10.12 local, venv at `./.venv`. Dockerfile uses python:3.12-slim. pyarrow pinned `20.0.0` in requirements (installed 25.0.0 — pin mismatch if rebuilt).
- `get_settings()` (config/settings.py:144) loads `.env`; raw `Settings()` does NOT. Use `get_settings()` in scripts.
- Session priority in `BreezeSessionManager.get_client()`: process token -> Redis cache -> auto-login (dead) -> BreezeSessionError. `update_session_token()` persists to Redis + `.env`.
- NiftyTrader participant-OI endpoint: `https://webapi.niftytrader.in/webapi/Resource/participant-wise-oi-chart-data` (Referer/Origin `https://www.niftytrader.in/participant-wise-oi`). Rows: client_type FII/DII/Pro/Client (+TOTAL to skip), `future_index_long/short`, `option_index_*`, `prev_*`, `date`, `nifty50`. NSE official participant-OI endpoints are 404/dead.

## Phase 7 specifics (Performance & Cost)
- `utils/chart_generator.py`: ChartCache (LRU) + `generate_oi_chart(call_oi, put_oi, spot, cache=)` -> Agg PNG (Agg backend avoids GUI dep).
- `core/llm_cache.py`: LLMCache (LRU+TTL 900s, max 256) + `market_state_key` (sha1 of floats formatted to 2 decimals — groups near-identical states) + `cached_maker_output`.
- `utils/telegram_bot.py`: `send_media_group` batches up to `TELEGRAM_MEDIA_GROUP_MAX=10`.
- `memory/vector_store.py`: Qdrant `ensure_collection` HNSW m=16/ef_construct=200/max_indexing_threads=2 + KEYWORD payload indexes on `QDRANT_PAYLOAD_INDEX_FIELDS`.
- `core/redis_manager.py`: OI buffer TTL (`OI_BUFFER_TTL_SECONDS=10*3600`) set on first write only; `memory_usage_bytes()` (None when INFO unavailable — fakeredis); `dbsize()`.
- `main.py` `/status`: includes `redis_memory_bytes`, `redis_keys`.
