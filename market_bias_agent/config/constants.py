"""Centralized constants. All PRD magic numbers live here."""

# ---------------------------------------------------------------------------
# Data buffering (PRD Module 1)
# ---------------------------------------------------------------------------
SPOT_TICK_BUFFER_SIZE = 300
OI_WINDOW_INTERVALS = 30

# Redis keys
KEY_SPOT_TICKS = "spot_ticks"
KEY_CALL_OI_PREFIX = "call_oi_strike_"
KEY_PUT_OI_PREFIX = "put_oi_strike_"

# Redis TTLs on volatile buffers (Phase 7): OI windows are intraday-only
OI_BUFFER_TTL_SECONDS = 10 * 3600  # 10h covers the trading day
STRIKES_TTL_SECONDS = 24 * 3600  # daily refresh anyway

# ---------------------------------------------------------------------------
# Feature engine (PRD Module 2)
# ---------------------------------------------------------------------------
VELOCITY_1M_SECONDS = 60
VELOCITY_5M_SECONDS = 300

# Level interaction: |Spot - Level| <= 12.0 points
LEVEL_DISTANCE_TOLERANCE = 12.0

# Trigger threshold matrix (scaled by profile via TriggerProfile below)
SCALP_VELOCITY_1M_MIN = 40_000
SCALP_VOLUME_VS_20MA_MULTIPLIER = 1.5
INTRADAY_VELOCITY_5M_MIN = 150_000

# ---------------------------------------------------------------------------
# Risk guardrails (PRD Module 4)
# ---------------------------------------------------------------------------
SCALP_SL_MAX_POINTS = 4.0
SCALP_TARGET_MIN_POINTS = 6.0
PCR_LOW_BLOCK_THRESHOLD = 0.75
OI_UNWIND_BLOCK_OVERRIDE = 100_000
DUPLICATE_ALERT_COOLDOWN_SECONDS = 120

# Checker Rules D-G (Enhancement Phase 4)
# Rule D - Max daily loss circuit (hard halt on signals)
MAX_DAILY_LOSS_POINTS = 100.0
MAX_DAILY_LOSS_PCT = 2.0
# Rule E - Signal rate limiting + per-strike cooldown
MAX_SIGNALS_PER_HOUR = 5
SIGNAL_RATE_WINDOW_SECONDS = 3600
STRIKE_COOLDOWN_SECONDS = 300
# Rule F - Spread guard (reject if bid-ask spread > this)
MAX_SPREAD_POINTS = 2.0
# Rule G - ATR sanity: target distance must be reachable within `factor` bars of ATR
ATR_TARGET_REACHABLE_FACTOR = 2.0

# ---------------------------------------------------------------------------
# Structured signal schema & Maker/Checker guardrails (Enhancement Phase 4)
# ---------------------------------------------------------------------------
SIGNAL_DIRECTIONS = ("BULLISH", "BEARISH", "NEUTRAL")
SIGNAL_SIDES = ("LONG", "SHORT")
TRAP_TYPES = ("BULL_TRAP", "BEAR_TRAP", "BREAKOUT", "NONE")
MAKER_REQUIRED_FIELDS = (
    "direction",
    "confidence",
    "entry_zone",
    "sl",
    "target",
    "rationale",
    "trap_type",
)
LLM_TEMPERATURE_MIN = 0.2
LLM_TEMPERATURE_MAX = 0.4
LLM_DAILY_TOKEN_BUDGET = 200_000
LLM_MAX_RETRIES = 1

# Google Gemini OpenAI-compatible endpoint
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# ---------------------------------------------------------------------------
# Memory intelligence (Enhancement Phase 5)
# ---------------------------------------------------------------------------
QDRANT_COLLECTION_DIM = 8  # FeatureEmbedder output dimension
# Gemini text-embedding-004 / gemini-embedding-001 output dimension (3072).
GEMINI_EMBEDDING_DIM = 3072  # used when EMBEDDING_BACKEND=gemini
QDRANT_DISTANCE = "COSINE"
QDRANT_SIMILAR_LIMIT = 5
QDRANT_SIMILAR_BOOST = 0.05  # score boost when a boost condition matches
QDRANT_MAX_AGE_DAYS = 45  # weekly compaction removes older vectors
QDRANT_SNAPSHOT_DIR = "snapshots"
QDRANT_BATCH_SIZE = 128

# HNSW tuning for ~100k vectors (Phase 7)
QDRANT_HNSW_M = 16
QDRANT_HNSW_EF_CONSTRUCT = 200
QDRANT_HNSW_MAX_INDEXING_THREADS = 2
QDRANT_PAYLOAD_INDEX_FIELDS = ("expiry_week", "session_date", "historical_outcome")

# LLM response cache (Phase 7)
LLM_CACHE_SIZE = 256
LLM_CACHE_TTL_SECONDS = 15 * 60  # reuse identical market-state bias within 15 min
LLM_CACHE_FEATURE_PRECISION = 2  # round features to this many digits for stable keys

# Telegram batch limits (Phase 7)
TELEGRAM_MEDIA_GROUP_MAX = 10

# Trap event outcomes + subsequent-move descriptors
TRAP_OUTCOMES = ("BULL_TRAP_REJECTION", "BEAR_TRAP_REJECTION", "TARGET_HIT", "SL_HIT")
OUTCOME_WIN = "TARGET_HIT"
OUTCOME_LOSS = "SL_HIT"

# Reference scales for deterministic numeric embeddings (offline mode)
SPOT_REFERENCE = 24_000.0
SPOT_SCALE = 4_000.0
OI_VELOCITY_SCALE = 200_000.0

# ---------------------------------------------------------------------------
# Post-trade analysis (Enhancement Phase 8)
# ---------------------------------------------------------------------------
SIGNAL_LIFECYCLE = (
    "SIGNAL_GENERATED",
    "APPROVED",
    "MONITORING",
    "EXITED",
    "ANALYZED",
    "CLOSED",
)
EXIT_REASONS = ("TARGET_HIT", "SL_HIT", "TIME_EXIT", "DIRECTION_INVALIDATED")
OUTCOMES = ("WIN", "LOSS", "BE")

# ---------------------------------------------------------------------------
# Data quality & reliability (Enhancement Phase 1)
# ---------------------------------------------------------------------------
# Append-only source-of-truth stream for raw ticks
KEY_RAW_TICK_STREAM = "raw_ticks"
STREAM_MAXLEN = 100_000

# Decision audit trail (Phase 6)
AUDIT_TRAIL_KEY = "audit_decisions"
AUDIT_TRAIL_MAXLEN = 10_000

# Tick integrity
MAX_TICK_AGE_SECONDS = 5.0  # drop ticks older than this vs our clock
TICK_SKEW_TOLERANCE_SECONDS = 2.0  # reject ts jumping forward by more than this vs prev
# OI ticks carry `ltt` = last TRADE time; an illiquid strike can trade rarely yet
# push current OI every few seconds. Treating the trade timestamp as staleness or
# a forward jump would drop live OI data, so OI gets much wider tolerances.
OI_MAX_TICK_AGE_SECONDS = 3600.0  # drop OI ticks older than this vs our clock
OI_TICK_SKEW_TOLERANCE_SECONDS = 300.0  # allow up to 5m between same-strike OI ticks

# Watchdog / reconnect
TICK_WATCHDOG_IDLE_SECONDS = 10.0  # reconnect if no tick in this window
WS_BACKOFF_BASE_SECONDS = 2.0
WS_BACKOFF_MAX_SECONDS = 60.0
WS_BACKOFF_FACTOR = 2.0

# Ops watchdog (Phase 6): alert Telegram when ticks stall during market hours
WATCHDOG_IDLE_SECONDS = 300.0  # 5 minutes

# Strikes management
KEY_STRIKES = "nifty_strikes"
STRIKES_SYNC_INTERVAL_SECONDS = 300  # refresh every 5 min
STRIKES_RANGE_AROUND_ATM = 20  # strikes to keep each side of ATM
EXPIRY_ROLLOVER_KEY = "active_expiry_date"

# ---------------------------------------------------------------------------
# Breeze session maintenance (Enhancement: Telegram session updates)
# ---------------------------------------------------------------------------
KEY_BREEZE_SESSION_TOKEN = "breeze_session_token"
BREEZE_SESSION_TOKEN_TTL_SECONDS = 26 * 3600  # outlive the ~24h token lifetime
BREEZE_SESSION_REFRESH_INTERVAL_SECONDS = 6 * 3600  # re-login before daily expiry
BREEZE_SESSION_MAX_AGE_SECONDS = 23 * 3600  # force refresh if older than this
BREEZE_CUSTOMER_LOGIN_URL = "https://api.icicidirect.com/breezeapi/api/v1/customerlogin"
SESSION_REFRESH_CRON_HOUR_IST = 8
SESSION_REFRESH_CRON_MINUTE_IST = 0
BREEZE_HISTORY_INTERVAL = "1minute"  # historical candle interval for ingest_history

# Telegram session listener (getUpdates long-polling)
TELEGRAM_GETUPDATES_TIMEOUT = 50  # seconds, matches Bot API long-poll
TELEGRAM_POLL_INTERVAL_SECONDS = 2.0

# ---------------------------------------------------------------------------
# Advanced feature engine (Enhancement Phase 2)
# ---------------------------------------------------------------------------
# Volume delta (tick rule: up-tick volume = buying pressure)
VOLUME_DELTA_BUY_RATIO_STRONG = 1.5  # buy/sell ratio >= this -> strong buying
VOLUME_DELTA_SELL_RATIO_STRONG = 0.67  # buy/sell ratio <= this -> strong selling

# OI + price divergence labels
DIVERGENCE = {
    "LONG_BUILD": "price up + OI up (new longs)",
    "SHORT_COVER": "price up + OI down (short covering)",
    "SHORT_BUILD": "price down + OI up (new shorts)",
    "LONG_UNWIND": "price down + OI down (long unwinding)",
    "NEUTRAL": "no meaningful divergence",
}

# Momentum acceleration window
MOMENTUM_DT_SECONDS = 60.0

# ATR / volatility regime (1-minute candles)
ATR_PERIOD = 14
ATR_REFERENCE_PERIOD = 20  # ATR vs its own 20-bar average
REGIME_CALM_ATR_PCT = 0.03  # ATR% < this -> CALM
REGIME_HIGH_VOL_ATR_PCT = 0.08  # ATR% > this -> HIGH_VOL
VOLATILITY_REGIMES = ("CALM", "ACTIVE", "HIGH_VOL")

# Candle pattern names
PATTERN_BULL_ENGULFING = "BULLISH_ENGULFING"
PATTERN_BEAR_ENGULFING = "BEARISH_ENGULFING"
PATTERN_PIN_BAR = "PIN_BAR"
PATTERN_HAMMER = "HAMMER"
PATTERN_SHOOTING_STAR = "SHOOTING_STAR"
PATTERN_SWEEP_HIGH = "SWEEP_HIGH"
PATTERN_SWEEP_LOW = "SWEEP_LOW"
PIN_BAR_WICK_BODY_RATIO = 2.0
SWEEP_WICK_TOLERANCE_POINTS = 3.0

# Regime-scaled trigger thresholds (multipliers over the base profile)
# CALM markets need bigger moves to trigger (reduce noise); HIGH_VOL markets
# trigger earlier.
REGIME_THRESHOLD_SCALING = {
    "CALM": {
        "scalp_velocity_1m": 1.3,
        "intraday_velocity_5m": 1.3,
        "volume_vs_20ma": 1.8,
    },
    "ACTIVE": {
        "scalp_velocity_1m": 1.0,
        "intraday_velocity_5m": 1.0,
        "volume_vs_20ma": 1.5,
    },
    "HIGH_VOL": {
        "scalp_velocity_1m": 0.8,
        "intraday_velocity_5m": 0.8,
        "volume_vs_20ma": 1.2,
    },
}

# ---------------------------------------------------------------------------
# Schedule (IST)
# ---------------------------------------------------------------------------
EOD_CRON_HOUR_IST = 18
EOD_CRON_MINUTE_IST = 0
PREMARKET_CRON_HOUR_IST = 8
PREMARKET_CRON_MINUTE_IST = 30
MARKET_OPEN_IST = "09:15"
MARKET_CLOSE_IST = "15:30"

# ---------------------------------------------------------------------------
# Pre-market engine (PRD Module 6)
# ---------------------------------------------------------------------------
KEY_PRE_MARKET_LEVELS = "premarket_levels"
PRE_MARKET_LEVELS_TTL_SECONDS = 12 * 3600  # survives the next trading day
KEY_EOD_STRUCTURAL_BIAS = "eod_structural_bias"
# EOD 18:00 IST -> next trading day 15:30 IST close (~21.5h). 30h TTL covers a
# normal overnight gap; the signal engine also validates session_date, so a
# Friday EOD bias simply goes stale over the weekend (Monday morning premarket
# recomputes it) instead of being misapplied.
EOD_STRUCTURAL_BIAS_TTL_SECONDS = 30 * 3600
MAX_PAIN_PINNING_TOLERANCE = 12.0  # |spot - max_pain| band used by live engine
S_R_LEVEL_ROUND_BASE = 100  # psychological level spacing (24000, 24100, ...)

# ---------------------------------------------------------------------------
# Trigger threshold profiles (scaled from PRD baseline)
# ---------------------------------------------------------------------------
TRIGGER_PROFILES = {
    "AGGRESSIVE": {
        "scalp_velocity_1m": int(SCALP_VELOCITY_1M_MIN * 0.8),
        "intraday_velocity_5m": int(INTRADAY_VELOCITY_5M_MIN * 0.8),
        "volume_vs_20ma": 1.3,
    },
    "MODERATE": {
        "scalp_velocity_1m": SCALP_VELOCITY_1M_MIN,
        "intraday_velocity_5m": INTRADAY_VELOCITY_5M_MIN,
        "volume_vs_20ma": SCALP_VOLUME_VS_20MA_MULTIPLIER,
    },
    "CONSERVATIVE": {
        "scalp_velocity_1m": int(SCALP_VELOCITY_1M_MIN * 1.2),
        "intraday_velocity_5m": int(INTRADAY_VELOCITY_5M_MIN * 1.2),
        "volume_vs_20ma": 1.8,
    },
}


def get_trigger_profile(name: str) -> dict:
    """Return the scaled trigger thresholds for a profile name."""
    return TRIGGER_PROFILES.get(name.upper(), TRIGGER_PROFILES["MODERATE"])
