# 🚀 ENHANCEMENT PLAN - AI Trading Agent V2.0+

**Base:** Master PRD V2.0 (Dual-Engine Institutional AI Trading Agent)
**Target:** Production-hardened, backtestable, capital-safe upgrade of the core architecture.

---

## 1. PHASE-WISE ENHANCEMENT ROADMAP

| Phase | Focus Area | Estimated Effort | Priority |
|-------|-----------|------------------|----------|
| 0 | Foundation Hardening | 3-4 days | CRITICAL |
| 1 | Data Quality & Reliability | 3-4 days | CRITICAL |
| 2 | Feature Engine Upgrade | 4-5 days | HIGH |
| 3 | Backtesting & Paper Trading | 5-7 days | HIGH |
| 4 | Risk & Guardrail Deepening | 3-4 days | HIGH |
| 5 | Memory Intelligence (Qdrant) | 3-4 days | MEDIUM |
| 6 | Observability & Alerting | 2-3 days | MEDIUM |
| 7 | Performance & Cost Optimization | 2-3 days | LOW |
| 8 | Post-Trade Signal Analysis | 3-4 days | HIGH |

---

## 2. PHASE 0 - FOUNDATION HARDENING (CRITICAL)

### 2.1 Repository Structure Upgrade
Current PRD structure lacks testability and separation of concerns. Proposed:

```
market_bias_agent/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example            # <-- ADD: versioned, no secrets
├── config/
│   ├── settings.py
│   └── constants.py        # <-- ADD: thresholds, model names, channels
├── core/
│   ├── websocket_client.py
│   ├── redis_manager.py
│   ├── math_engine.py
│   └── logger.py           # <-- ADD: structured logging (JSON)
├── memory/
│   └── vector_store.py
├── graph/
│   ├── workflow.py
│   └── nodes.py
├── modules/
│   ├── eod_engine.py
│   ├── premarket_engine.py
│   ├── backtest_engine.py  # <-- ADD
│   ├── paper_trader.py     # <-- ADD
│   └── post_analysis.py    # <-- ADD: post-trade outcome tracking & analysis
├── utils/
│   ├── chart_generator.py
│   ├── telegram_bot.py
│   └── metrics.py          # <-- ADD: hit-rate, PnL, drawdown calc
├── tests/                  # <-- ADD
│   ├── test_math_engine.py
│   ├── test_guardrails.py
│   └── test_redis.py
├── scripts/                # <-- ADD
│   ├── ingest_eod_history.py
│   └── seed_vector_db.py
└── main.py
```

### 2.2 Configuration & Secrets Management
- Move all credentials to `.env` (never committed). Add `.env.example` with placeholders.
- Add config validation on startup (fail-fast if `ICICI_API_KEY`, `BREEZE_TOKEN`, `TELEGRAM_BOT_TOKEN`, `OPENAI_API_KEY` missing).
- Centralize all magic numbers from PRD (thresholds, distances, time windows) into `constants.py`.

### 2.3 Dependency Pinning
- Pin all versions in `requirements.txt` with `==` to guarantee reproducible builds.
- Add `requirements-dev.txt` (pytest, pytest-asyncio, ruff, mypy).

### 2.4 Logging & Tracing
- Structured JSON logging with correlation IDs across the async pipeline.
- Trace a tick from WebSocket → Redis → Feature Engine → Trigger → LLM → Telegram for debugging.

---

## 3. PHASE 1 - DATA QUALITY & RELIABILITY (CRITICAL)

### 3.1 WebSocket Resilience
- **Reconnection logic:** Exponential backoff with jitter; auto-resubscribe to symbols/strikes on reconnect.
- **Heartbeat/last-tick watchdog:** If no tick received for N seconds, alert + attempt reconnect.
- **Backfill on restart:** On boot, fetch recent REST snapshot to fill Redis buffers so 1m/5m velocity windows are warm immediately.

### 3.2 Tick Integrity Checks
- Drop out-of-sequence or stale ticks (timestamp skew).
- Detect duplicate ticks and normalize tick granularity.
- Store raw ticks in an immutable append-only Redis Stream as the "source of truth" before computing derived buffers.

### 3.3 Strikes Management
- Dynamic strike list sync: pull current Nifty 50 index option strikes from Breeze REST periodically (and on expiry day) so `call_oi_strike_{strike}` lists stay current.
- Handle contract expiry rollover (last Thursday) automatically.

### 3.4 Clock & Timezone
- Single source of truth for IST time inside container (`TZ=Asia/Kolkata` already set) — also enforce in code with `zoneinfo`/`pytz`, never server-local time.

---

## 4. PHASE 2 - FEATURE ENGINE UPGRADE (HIGH)

### 4.1 Expand Trigger Matrix (beyond PRD minimum)
Keep all PRD formulas, but add:
- **Volume Delta (ΔVol)** between buying/selling pressure.
- **OI + Price Divergence detection** (price up, OI down = short covering vs long unwinding).
- **Rate-of-change acceleration** (velocity of velocity = momentum shift).
- **Volatility regime detection** (ATR(14) bands to classify calm/active/high-vol states; scale trigger thresholds by regime to reduce noise in calm markets).
- **Candle patterns** on 1m (engulfing, pin bars, sweep of previous high/low).

### 4.2 Configurable Thresholds
- Move all trigger thresholds into config so they can be tuned without code change.
- Support threshold profiles: `AGGRESSIVE`, `MODERATE`, `CONSERVATIVE`.

### 4.3 Cache & Compute Efficiency
- Precompute level distance matrix per tick (Spot vs all round levels) in O(1) lookups.
- Use numpy/pandas vectorized ops; avoid Python loops in hot tick path.

---

## 5. PHASE 3 - BACKTESTING & PAPER TRADING (HIGH)

### 5.1 Historical Data Ingestion
- Script to pull historical Nifty 50 index option/spot data (minute bars) into a local parquet/duckdb store.
- Replay engine that feeds historical ticks/bars through the exact same live pipeline for validation.

### 5.2 Backtest Engine
- Implement PRD rules as pure functions (deterministic) so backtest = live.
- Metrics computed per alert/trigger: hit-rate, avg win/loss, expectancy, max drawdown, profit factor, max consecutive losses.
- Slippage & cost model: add per-trade brokerage + slippage estimate.

### 5.3 Walk-Forward Validation
- Time-series split: train/validate/trade on consecutive non-overlapping periods.
- Report monthly/weekly aggregated performance table.

### 5.4 Paper Trader
- Simulated order routing (entry SL/Target) that runs the live signal path in shadow mode.
- Track PnL vs signals; compare "as-alerted" vs "as-executed".

---

## 6. PHASE 4 - RISK & GUARDRAIL DEEPENING (HIGH)

### 6.1 Extend Checker Node Rules
Keep PRD Rules A/B/C and add:
- **Rule D - Max Daily Loss Circuit:** Hard halt of all signals if daily simulated loss > X%.
- **Rule E - Signal Rate Limiting:** max N alerts per hour; cooldown per strike.
- **Rule F - Spread/Slippage guard:** reject if current bid-ask spread > threshold points.
- **Rule G - ATR sanity:** reject signal if implied target unreachable within 1 bar ATR.

### 6.2 Structured Signal Output
Enforce a strict, parseable signal schema from Maker node:
```json
{
  "direction": "BULLISH|BEARISH|NEUTRAL",
  "confidence": 0.0-1.0,
  "entry_zone": [low, high],
  "sl": float,
  "target": float,
  "rationale": "short string",
  "trap_type": "BULL_TRAP|BEAR_TRAP|BREAKOUT|NONE"
}
```

### 6.3 LLM Guardrails
- Temperature low (0.2-0.4) for deterministic trade bias.
- Output JSON schema validation + retry-once on malformed JSON.
- Total budget (tokens/cost) caps per day; fallback to rule-only mode if LLM budget exhausted.

---

## 7. PHASE 5 - MEMORY INTELLIGENCE (Qdrant) (MEDIUM)

### 7.1 Richer Embeddings & Payloads
Extend PRD payload:
```json
{
  "vector_id": "uuid",
  "timestamp": "ISO-8601 (IST)",
  "market_state": "PCR: 0.95, Spot: 24005, Call_OI_Vel: -85000",
  "features": {"pcr": 0.95, "spot": 24005, "call_oi_vel_1m": -85000, "velocity_5m": 150000, "volatility": "HIGH"},
  "historical_outcome": "BULL_TRAP_REJECTION",
  "subsequent_move": "-45 points in 15 mins",
  "session_date": "2026-08-05",
  "expiry_week": 3
}
```
- Store raw numeric features so similarity search can combine vector + scalar filtering.

### 6.2 Query Strategy
- On trigger: fetch top-K similar historical traps within same expiry-week band.
- Boost score when market_state PCR/Vol regime matches current conditions.
- Surface "similar situation" count + win-rate in the Telegram alert for context.

### 6.3 Data Lifecycle
- Daily EOD job indexes new events; weekly compaction job removes low-quality / stale vectors.
- Export/backup snapshots of the collection.

---

## 8. PHASE 6 - OBSERVABILITY & ALERTING (MEDIUM)

### 8.1 Health Monitoring
- `/health` endpoint or heartbeat into Redis exposing: last tick age, buffer fill %, ws connected, queue depth, last cron success.
- Watchdog cron: if no ticks for 5 min in market hours → Telegram alert to ops channel.

### 8.2 Dashboards (optional)
- Export key metrics to Prometheus/Grafana OR simple JSON status page.
- At minimum: structured daily report to Telegram summarizing triggers, approvals, rejections, LLM cost.

### 8.3 Audit Trail
- Log every decision (Maker output, Checker verdict + reason) to Redis/DB for post-hoc review and backtest label enrichment.

---

## 9. PHASE 7 - PERFORMANCE & COST OPTIMIZATION (LOW)

- Batch Telegram sends; reuse matplotlib figures (pre-render common charts).
- Cache OpenAI calls via similar-input hashing for repeated market states.
- Right-size Qdrant collection with HNSW index params tuned for ~100k vectors.
- Redis: use smaller value types (raw binary/lists) and TTLs on volatile buffers; monitor memory.

---

## 10. DOCKER & DEPLOYMENT ENHANCEMENTS

```yaml
services:
  redis:
    image: redis:7-alpine
    container_name: agent_redis
    restart: always
    ports: ["6379:6379"]
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy noeviction
    volumes: [redis_data:/data]

  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: agent_qdrant
    restart: always
    ports: ["6333:6333"]
    volumes: [qdrant_data:/qdrant/storage]
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

  trading_agent:
    build: .
    container_name: dual_engine_agent
    restart: always
    depends_on: [redis, qdrant]
    env_file: .env
    environment:
      - TZ=Asia/Kolkata
    volumes: [.:/app]
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"]
      interval: 30s
      timeout: 5s
      retries: 3

volumes:
  redis_data:
  qdrant_data:
```

### Docker Extras
- Multi-stage Dockerfile (build deps → slim runtime) to cut image size.
- Non-root user in container.
- Separate `docker-compose.override.yml` for local dev (mounted source, hot reload).
- Add promtail/vector sidecar (optional) for log shipping.

---

## 10.5 PHASE 8 - POST-TRADE SIGNAL ANALYSIS (HIGH)

Har generated signal ka outcome track karo, analyze karo, aur wapas memory me feed karo — taaki system har trade se learn kare. Signal generation ke baad ek **Post-Trade Analysis Loop** chalta hai.

### 10.5.1 Signal State Machine (Lifecycle)
```
SIGNAL_GENERATED → APPROVED → MONITORING → EXITED (TARGET/SL/TIME) → ANALYZED → CLOSED
```

- `SIGNAL_GENERATED` → Maker-Checker approve → signal ID (uuid) milta hai.
- `MONITORING` → post_analysis worker signal ke entry/exit prices, timestamp, SL, target track karta hai.
- `EXITED` → exit reason recorded: `TARGET_HIT | SL_HIT | TIME_EXIT | DIRECTION_INVALIDATED`.
- `ANALYZED` → outcome computed (PnL, points moved, max favorable/adverse excursion).
- `CLOSED` → analysis write-back to Qdrant + stats DB.

### 10.5.2 Outcome Tracking (What to measure per signal)
- **Filled entry** vs **alert price** (slippage).
- **Actual result:** `WIN` / `LOSS` / `BE` (break-even) + points & PnL.
- **Max Favorable Excursion (MFE)** and **Max Adverse Excursion (MAE)** in points.
- **Time-to-exit** (mins) — signal duration vs expected move window.
- **Direction validity:** did price reach target before any opposite-direction level break?
- **Trap classification match:** predicted trap type vs what actually happened.

### 10.5.3 Analysis Store (SQLite/Postgres or Redis sorted-set)
```sql
CREATE TABLE signal_log (
  signal_id      TEXT PRIMARY KEY,
  generated_at   TIMESTAMP,   -- IST
  trigger_type   TEXT,        -- SCALP | INTRADAY
  direction      TEXT,        -- BULLISH | BEARISH
  entry_zone     TEXT,        -- [low, high]
  sl             REAL,
  target         REAL,
  exit_price     REAL,
  exit_reason    TEXT,        -- TARGET_HIT | SL_HIT | TIME_EXIT | INVALIDATED
  mfe            REAL,
  mae            REAL,
  pnl_points     REAL,
  outcome        TEXT,        -- WIN | LOSS | BE
  confidence     REAL,
  maker_rationale TEXT
);
```

### 10.5.4 Write-back to Memory (Qdrant)
- `post_analysis.py` EOD job: har ANALYZED signal ko `nifty_historical_traps` collection me upsert karo with **actual outcome** — PRD me sirf `historical_outcome` tha; ab `actual_outcome` bhi store hoga.
- Isse next similarity search me "similar situations ka actual win-rate" milta hai — signals better calibrated hote hain.

### 10.5.5 Post-Trade Performance Feedback to LLM
- Weekly report: per trigger-type hit-rate, avg MFE/MAE, best/worst session.
- Bias-correction loop: agar `BULL_TRAP` prediction ka win-rate < 40%, to Checker rule add karo (threshold tighten) — rule-based feedback, LLM par fully depend nahi.

### 10.5.6 Post-Trade Telegram Report
- Har trade exit ke baad: concise summary `SIGNAL #123 → TARGET HIT | +12.5 pts | MFE 18 | MAE 2 | time 6 min`.
- Weekly aggregation: hit-rate, expectancy, profit factor, per-trigger breakdown.

---

## 11. TESTING STRATEGY

| Layer | Tools | Coverage |
|-------|-------|----------|
| Unit | pytest | math_engine (PCR, velocity, level interaction), guardrail rules A-G |
| Integration | pytest-asyncio + fakeredis | websocket → redis stream → feature engine pipeline |
| Mock LLM | dependency injection | maker/checker workflow with canned LLM responses |
| Backtest | custom replay | full day replay = historical signals vs recorded outcomes |
| E2E | docker compose | smoke test boot, healthcheck, cron runs |

---

## 12. QUICK WIN CHECKLIST (implement FIRST, highest ROI)

1. [x] `constants.py` + `.env.example` — kill magic numbers.
2. [x] WebSocket auto-reconnect + backfill on restart.
3. [x] Watchdog: stale-tick alert.
4. [x] Checker Rules D, E, F, G (halt circuits).
5. [x] Structured signal JSON schema + validation.
6. [x] Backtest replay harness (rule functions already pure).
7. [x] Health endpoint + daily ops report.
8. [ ] Version-pinned deps + lint/typecheck in CI.
9. [x] Signal state machine + `signal_log` table (post-trade tracking).
10. [x] Actual-outcome write-back to Qdrant (learn from every trade).

---

## 13. RISKS & MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|------------|
| ICICI API rate limits / throttling | Data gaps | Caching, backoff, buffered replay |
| Expiry rollover breaks strike data | Wrong levels | Auto strike-sync + expiry-aware logic |
| LLM cost runaway | Cost blowout | Daily budget cap + fallback rule-only mode |
| Overfitting in backtest | False confidence | Walk-forward + slippage/cost model |
| Tick timestamp skew | Bad velocity | Clock sync (NTP) + tick sanity filters |
| Single VPS point of failure | Outage | Healthcheck + restart policies + ops alerting |

---

## 14. SUGGESTED IMPLEMENTATION ORDER (Gantt-style)

```
Week 1:  Phase 0 + Phase 1 (foundation, data reliability)
Week 2:  Phase 2 + Phase 4 (feature engine, guardrails)
Week 3:  Phase 3 (backtest + paper trade)  ← validate against live first
Week 4:  Phase 5 + 6 (memory, observability)
Week 5:  Phase 8 (post-trade analysis + Qdrant write-back) ← pairs with live paper trading
Week 6:  Phase 7 + production hardening + CI/CD
```

> ⚠️ **Golden Rule:** Do NOT go live with real capital until Phase 3 (backtest + paper trading) shows positive expectancy over ≥ 3 weeks with a conservative threshold profile. Post-trade analysis (Phase 8) zyada important hai — bina outcome tracking ke backtest validation adhoora hai.
