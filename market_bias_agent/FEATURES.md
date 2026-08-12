# Agent Features

Dual-Engine Institutional AI Trading Agent — NSE Nifty 50 Index Options.
Live data pipeline → AI signal → guardrails → paper trade → memory → self-improvement.

## 1. Live Data Pipeline (WebSocket → Redis)

- **Breeze WebSocket** (ICICI Direct) se live ticks: spot + OI per-strike (CALL/PUT).
  - `core/breeze_transport.py` — WebSocket transport + session mgmt
  - `core/websocket_client.py` — tick subscription + auto-reconnect + watchdog
- Tick flow: WebSocket → queue → `tick_pipeline.py` (validate → persist) → Redis buffers.
  - `core/tick_validator.py` — stale/duplicate/OOO rejection (spot strict, OI wider tolerances)
- **Backfill** restart par REST snapshot se buffers warm karta hai.
- Session token Telegram `/session <token>` se push hota hai (auto-login ICICI me dead).

## 2. Signal Engine — "Analyzer"

Har validated tick par feature calculation + LLM prompt (`modules/signal_engine.py`):

- **PCR** (put/call ratio) + **total OI** call vs put
- **OI velocity** 1m/5m (fresh OI inflow) + spot velocity + **volume delta**
- **ATR** (volatility) + **regime** classify (CALM / ACTIVE / HIGH_VOL)
- **ATR bands** (support/resistance) + **premarket pivot S/R levels** + **max pain**
- Feature assembly → **LLM Maker** (`modules/maker_node.py`) → structured signal
  (direction + confidence + entry/target/SL)

## 3. Checker — Guardrails

Maker ka signal approve/reject (`modules/checker_node.py`) — Rules A–G:

- A: PCR / total-OI consistency
- B: OI velocity confirmation
- C: ATR / regime check
- D: **daily-loss halt** (loss circuit)
- E: **signal rate + strike cooldown** (no overtrading)
- LLM guardrails: temperature clamp (0.2–0.4), JSON parse retry-once, daily token budget,
  rule-only fallback (LLM down par bhi checker chalta hai)

## 4. Paper Trader

Approved signals → **paper positions** (no real money).

- Live premium feed se PnL tracking (`modules/paper_trader.py`)
- SL / TARGET / TIME exit lifecycle
- Positions + closed stats, outcome → SQLite signal store

## 5. Memory — Qdrant (auto-learning)

- **TrapRecord** → FeatureEmbedder (8-dim) → Qdrant vector store (`memory/`)
- Similar past situations retrieve karke **regime boost** signal decision ko
- EOD par live traps collect → memory (reinforcement loop)
- `MemoryService.similar_situations()` + `compact_stale()` weekly cleanup

## 6. Post-Trade Analysis

- Signal lifecycle (SQLite `core/signal_store.py`):
  created → approved → target/SL/time exit
- Outcome write-back → **weekly report** + bias correction (`modules/post_analysis.py`)

## 7. EOD Engine (18:00 IST cron)

`modules/eod_engine.py` — 2 real data-backed tasks:

1. **Institutional footprint** — NSE participant-wise OI report (FII/DII/Pro/Client)
   via NiftyTrader API (`core/participant_oi.py`):
   - FII net index futures, written calls vs puts
   - Pro written calls/puts
   - Retail (Client) long vs institutional short → "Sell on Rise"
   - → **next-day structural bias** (BULLISH / BEARISH / NEUTRAL) → Telegram
2. **Memory ingestion** — day ke real trap events (SQLite se) → Qdrant index

Failure isolation: participant-OI fetch fail → report "unavailable", trap indexing abhi bhi chalta hai.

## 8. Premarket Engine (08:30 IST cron)

- Pivot S/R, psych levels, max pain (`modules/premarket_engine.py`)
- Signal engine context + dashboard (half-wired — only `_near_level()` consumes)

## 9. Observability & Ops

- **Telegram bot**: daily ops report, OPS alerts (watchdog), `/status`, `/session`
- **Health endpoints**: `/health` (live metrics), `/status`, `/ops/watchdog`, `/ops/daily-report`
- **Watchdog**: market hours me tick stall detect → Telegram alert
- **Redis audit trail** + OI charts (`utils/chart_generator.py`, Agg PNG + LRU cache)
- LLM cache (LRU + TTL) market-state key par

## 10. Backtest (offline)

- Real OI history (`scripts/ingest_history.py --with-oi` → parquet)
- `scripts/run_backtest.py`: walk-forward, SL/target, max-hold, hit-rate

---

## End-to-End Flow

```
Live ticks (Breeze WS)
   ↓ validate (tick_pipeline)
   ↓ feature assembly (signal_engine)
   ↓ LLM Maker signal (maker_node)
   ↓ Checker guardrails Rules A–G (checker_node)
   ↓ Paper trade (paper_trader)
   ↓ Outcome → SQLite → EOD memory ingest (eod_engine → Qdrant)
   ↓ Next decision memory boost (memory_service)
```
