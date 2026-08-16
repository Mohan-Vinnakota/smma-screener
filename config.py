# ── Screening ─────────────────────────────────────────────────
LTP_MIN     = 30
LTP_MAX     = 500
MIN_BID_QTY = 1_000_000
MIN_ASK_QTY = 1_000_000

# ── SMMA Periods ──────────────────────────────────────────────
SMMA_FAST   = 20
SMMA_SLOW   = 120

# ── ETQ Windows ───────────────────────────────────────────────
ETQ_WINDOWS = [5, 20, 60]

# ── ML Model ──────────────────────────────────────────────────
ML_MIN_SAMPLES      = 50
ML_FEATURE_SHORT    = 2    # minutes for short LTQ avg
ML_FEATURE_LONG     = 5    # minutes for long LTQ avg

# ── Server ────────────────────────────────────────────────────
HTTP_PORT   = 5000
WS_PORT     = 8765
HTTP_HOST   = "127.0.0.1"
WS_HOST     = "127.0.0.1"

# ── Data ──────────────────────────────────────────────────────
TICK_STORE_MINUTES  = 120
NSE_SYMBOL_FILE     = "nse_symbols.csv"
DB_PATH             = "smma_screener.db"
LOG_FOLDER          = "logs"

# ── Reconnect ─────────────────────────────────────────────────
WS_RECONNECT_DELAY  = 5    # seconds before reconnecting
