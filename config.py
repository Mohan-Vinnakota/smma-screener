# ── NSE Equity Screening ──────────────────────────────────────
LTP_MIN     = 30
LTP_MAX     = 500
MIN_BID_QTY = 1_000_000
MIN_ASK_QTY = 1_000_000

# ── MCX Commodity Screening ───────────────────────────────────
MCX_LTP_MIN     = 0        # no lower bound (Gold = 70,000+)
MCX_LTP_MAX     = 999_999  # no upper bound
MCX_MIN_BID_QTY = 1        # commodities have lower lot depth
MCX_MIN_ASK_QTY = 1

# MCX symbols to track (name as it appears in Angel One master)
MCX_SYMBOLS = [
    "CRUDEOIL",
    "NATURALGAS",
    "GOLD",
    "GOLDM",       # Gold Mini
    "SILVER",
    "SILVERM",     # Silver Mini
    "COPPER",
    "ZINC",
    "ALUMINIUM",
    "NICKEL",
    "LEAD",
]

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
MCX_SYMBOL_FILE     = "mcx_symbols.csv"
DB_PATH             = "smma_screener.db"
LOG_FOLDER          = "logs"

# ── Reconnect ─────────────────────────────────────────────────
WS_RECONNECT_DELAY  = 5    # seconds before reconnecting

# ── Market Hours (IST) ────────────────────────────────────────
# NSE Equity:   09:15 – 15:30
# MCX:          09:00 – 23:30
NSE_OPEN_H,  NSE_OPEN_M  = 9,  15
NSE_CLOSE_H, NSE_CLOSE_M = 15, 30
MCX_OPEN_H,  MCX_OPEN_M  = 9,  0
MCX_CLOSE_H, MCX_CLOSE_M = 23, 30
