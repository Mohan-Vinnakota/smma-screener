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

# ── NSE Currency Derivatives Screening ────────────────────────
CDS_LTP_MIN     = 0        # no lower bound (JPYINR quoted small, GBPINR larger)
CDS_LTP_MAX     = 999_999  # no upper bound
CDS_MIN_BID_QTY = 1        # currency lot depth is different from equity
CDS_MIN_ASK_QTY = 1

# Currency pairs to track (name as it appears in Angel One master)
CDS_SYMBOLS = [
    "USDINR",
    "EURINR",
    "GBPINR",
    "JPYINR",
]

# ── NSE F&O Screening ──────────────────────────────────────────
FNO_LTP_MIN     = 0
FNO_LTP_MAX     = 999_999
FNO_MIN_BID_QTY = 1
FNO_MIN_ASK_QTY = 1

# Index futures to track (name as it appears in Angel One master)
FNO_SYMBOLS = [
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
]

# ── Crypto (Binance) Screening ─────────────────────────────────
CRYPTO_LTP_MIN     = 0
CRYPTO_LTP_MAX     = 999_999_999
CRYPTO_MIN_QTY_24H = 0   # optional 24h volume filter, 0 = no filter for now

# Pairs to track (Binance symbol format, USDT-quoted)
CRYPTO_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
]

BINANCE_REST_BASE = "https://api.binance.com"
BINANCE_WS_BASE    = "wss://stream.binance.com:9443"

# ── US Markets (Alpaca) Screening ───────────────────────────────
US_LTP_MIN     = 0
US_LTP_MAX     = 999_999
US_MIN_QTY     = 0   # optional volume filter, 0 = no filter for now

# US stocks to track (must match Alpaca's exact ticker, e.g. "AAPL")
US_SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "NVDA",
    "META",
]

ALPACA_DATA_WS_BASE = "wss://stream.data.alpaca.markets/v2"
ALPACA_DATA_FEED    = "iex"   # "iex" = free real-time feed; "sip" needs a paid subscription
ALPACA_TRADING_BASE = "https://api.alpaca.markets"
ALPACA_ASSETS_URL   = f"{ALPACA_TRADING_BASE}/v2/assets"

# ── SMMA Periods ──────────────────────────────────────────────
SMMA_FAST   = 20
SMMA_SLOW   = 120

# ── ETQ Windows ───────────────────────────────────────────────
ETQ_WINDOWS = [5, 20, 60]

# ── ML Model ──────────────────────────────────────────────────
ML_MIN_SAMPLES      = 50
ML_FEATURE_SHORT    = 2    # minutes for short LTQ avg
ML_FEATURE_LONG     = 5    # minutes for long LTQ avg

# Decision threshold — predict_proba() >= this counts as ACCEPT.
# 0.5 is the naive default; ml_backtest.py sweeps this against real
# historical outcomes and prints a recommended value to put here.
ML_CONFIDENCE_THRESHOLD = 0.5

# Below this many closed trades (across ALL markets combined), predict()
# returns None/"Learning" instead of a fake ACCEPT/AVOID. ML_MIN_SAMPLES
# above is only when XGBoost is numerically able to fit — this is the
# separate, higher bar for actually trusting what it says. Raise this
# further once you've seen how noisy the 100-sample verdicts still are.
ML_TRUST_SAMPLES = 100

# Fraction of each market's closed signals (chronological, most
# recent last) held out as a test set in ml_backtest.py. Never
# trained on — used only to measure real accuracy/precision/recall.
ML_BACKTEST_TEST_FRACTION = 0.2

# ── Server ────────────────────────────────────────────────────
HTTP_PORT   = 5000
WS_PORT     = 8765
HTTP_HOST   = "127.0.0.1"
WS_HOST     = "127.0.0.1"
TELEGRAM_ACCEPT_ONLY = True   # False = send all signals including AVOID/Learning
# ── Data ──────────────────────────────────────────────────────
TICK_STORE_MINUTES  = 120
NSE_SYMBOL_FILE    = "symbols/nse_symbols.csv"
MCX_SYMBOL_FILE    = "symbols/mcx_symbols.csv"
CDS_SYMBOL_FILE    = "symbols/cds_symbols.csv"
FNO_SYMBOL_FILE    = "symbols/fno_symbols.csv"
CRYPTO_SYMBOL_FILE = "symbols/crypto_symbols.csv"
US_SYMBOL_FILE     = "symbols/us_symbols.csv"
DB_PATH             = "smma_screener.db"
LOG_FOLDER          = "logs"

# ── Reconnect ─────────────────────────────────────────────────
WS_RECONNECT_DELAY  = 5    # seconds before reconnecting

# ── Market Hours (IST) ────────────────────────────────────────
# NSE Equity:   09:15 – 15:30
# MCX:          09:00 – 23:30
# NSE Currency: 09:00 – 17:00
# NSE F&O:      09:15 – 15:30
NSE_OPEN_H,  NSE_OPEN_M  = 9,  15
NSE_CLOSE_H, NSE_CLOSE_M = 15, 30
MCX_OPEN_H,  MCX_OPEN_M  = 9,  0
MCX_CLOSE_H, MCX_CLOSE_M = 23, 30
CDS_OPEN_H,  CDS_OPEN_M  = 9,  0
CDS_CLOSE_H, CDS_CLOSE_M = 17, 0
FNO_OPEN_H,  FNO_OPEN_M  = 9,  15
FNO_CLOSE_H, FNO_CLOSE_M = 15, 30

# US Equities (Alpaca): 09:30 – 16:00 US/Eastern.
# Unlike the IST markets above, this can't be a fixed IST hour pair —
# the ET→IST offset itself shifts by an hour twice a year with US DST
# (India doesn't observe DST). market_us.py computes this with
# zoneinfo("America/New_York") directly instead of a fixed-offset
# lookup, so it stays correct year-round.
US_OPEN_H,  US_OPEN_M  = 9,  30
US_CLOSE_H, US_CLOSE_M = 16, 0
