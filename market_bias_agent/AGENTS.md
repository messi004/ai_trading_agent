# Project Memory

Dual-Engine Institutional AI Trading Agent (NSE Nifty 50 Index Options). Production-ready build, PRD + ENHANCEMENT_PLAN.md driven.

## Source of truth
- `/home/messi/AI_Trading_Agent/PRD` — master requirements (formulas, thresholds, guardrails)
- `/home/messi/AI_Trading_Agent/ENHANCEMENT_PLAN.md` — phase roadmap + quick-win checklist
- `/home/messi/AI_Trading_Agent/market_bias_agent/` — project root (all code)

## Status
- Phases 0,1,2,3,4,5,6,8 complete. **Phase 7 (Performance & Cost) complete.**
- 211 tests passing; ruff clean; mypy clean (43 source files).
- Quick-win checklist: 1-7, 9-10 [x]. Item 8 (CI workflow) pending — requires git repo (project is NOT a git repo yet).
- Dockerfile already multi-stage + non-root + healthcheck; deps version-pinned. docker-compose.override.yml + promtail sidecar optional.
- Golden rule: go-live on real capital only after backtest + paper trading show >=3 weeks positive expectancy.

## Architecture (2-engine pattern)
- **Analyzer**: WebSocket (Breeze) -> OI buffer -> tick pipeline -> LLM Maker (`modules/maker_node.py`) -> structured signal
- **Checker**: `modules/checker_node.py` — Rules A-G (guardrails) -> approved signal
- **Memory**: `memory/` — TrapRecord -> FeatureEmbedder (8-dim) -> Qdrant/MemoryVectorStore -> MemoryService (similar situations + regime boost)
- **Post-trade**: `core/signal_store.py` (SQLite lifecycle) -> `modules/post_analysis.py` (outcome write-back, weekly report, bias correction)
- **Observability**: `core/health.py`, `modules/monitoring.py` (watchdog + daily report), Redis audit trail, Telegram alerts, `utils/chart_generator.py` OI charts

## Verify commands
Run from `/home/messi/AI_Trading_Agent/market_bias_agent`:
- `./.venv/bin/python -m pytest -q` (211 passed)
- `./.venv/bin/ruff format .` and `./.venv/bin/ruff check .`
- `./.venv/bin/python -m mypy config/ core/ utils/ modules/ memory/ scripts/ main.py`
- CLI/offline tools: `SKIP_SECRETS_CHECK=true` env (config/settings.py validate())
- `.env` has dummy values; real Breeze/Gemini/Telegram creds needed for live. Stubs allow offline boot + tests.

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

## Phase 7 specifics (Performance & Cost)
- `utils/chart_generator.py`: ChartCache (LRU) + `generate_oi_chart(call_oi, put_oi, spot, cache=)` -> Agg PNG (Agg backend avoids GUI dep).
- `core/llm_cache.py`: LLMCache (LRU+TTL 900s, max 256) + `market_state_key` (sha1 of floats formatted to 2 decimals — groups near-identical states) + `cached_maker_output`.
- `utils/telegram_bot.py`: `send_media_group` batches up to `TELEGRAM_MEDIA_GROUP_MAX=10`.
- `memory/vector_store.py`: Qdrant `ensure_collection` HNSW m=16/ef_construct=200/max_indexing_threads=2 + KEYWORD payload indexes on `QDRANT_PAYLOAD_INDEX_FIELDS`.
- `core/redis_manager.py`: OI buffer TTL (`OI_BUFFER_TTL_SECONDS=10*3600`) set on first write only; `memory_usage_bytes()` (None when INFO unavailable — fakeredis); `dbsize()`.
- `main.py` `/status`: includes `redis_memory_bytes`, `redis_keys`.
