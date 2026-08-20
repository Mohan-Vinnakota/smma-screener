import time
import threading
import sys
import math
import random
import logging as _logging

from logger import logger
from core.engine import Engine
from server import start_servers, set_engine
from core.database import init_db

_logging.getLogger("werkzeug").setLevel(_logging.ERROR)

SIMULATE = "--simulate" in sys.argv
init_db()

# ─────────────────────────────────────────────────────────────
if SIMULATE:
    logger.info("=== SIMULATION MODE (NSE + MCX + CDS + FNO + CRYPTO + US) ===")

    from markets.market_nse_equity import SymbolState
    from markets.market_mcx import MCXSymbolState
    from markets.market_nse_currency import CDSSymbolState
    from markets.market_nse_fno import FNOSymbolState
    from markets.market_crypto import CryptoSymbolState
    from markets.market_us import USSymbolState
    from core.ml_model import MLModel
    from config import ML_MIN_SAMPLES

    shared_ml = MLModel(min_samples=ML_MIN_SAMPLES)
    shared_ml.bootstrap_from_history()

    def _rows_for(symbols_dict, market, decimals):
        rows = []
        for state in symbols_dict.values():
            if state.last_ltp is None:
                continue
            f = state.detector.fast.value
            s = state.detector.slow.value
            rows.append({
                "market":      market,
                "symbol":      state.symbol,
                "ltp":         round(state.last_ltp, decimals),
                "smma_fast":   round(f, decimals) if f else None,
                "smma_slow":   round(s, decimals) if s else None,
                "signal":      state.signal,
                "ml_verdict":  state.ml_verdict,
                "ml_conf":     f"{state.ml_confidence:.0%}" if state.ml_confidence else None,
                "ml_reason":   state.ml_reason,
                "bid_price":   round(state.last_bid_price, decimals),
                "bid_qty":     state.last_bid_qty,
                "ask_price":   round(state.last_ask_price, decimals),
                "ask_qty":     state.last_ask_qty,
                "etq_5m":      state.store.etq(5),
                "etq_20m":     state.store.etq(20),
                "etq_60m":     state.store.etq(60),
                "avg_ltp_20m": round(state.store.avg_ltp(20) or 0, decimals),
                "avg_ltp_60m": round(state.store.avg_ltp(60) or 0, decimals),
            })
        return rows

    # Build a fake engine that exposes get_rows()
    class SimEngine:
        def __init__(self):
            self.nse_symbols    = {}
            self.mcx_symbols    = {}
            self.cds_symbols    = {}
            self.fno_symbols    = {}
            self.crypto_symbols = {}
            self.us_symbols     = {}

        def get_rows(self):
            rows = []
            rows += _rows_for(self.nse_symbols, "NSE", 2)
            rows += _rows_for(self.mcx_symbols, "MCX", 2)
            rows += _rows_for(self.cds_symbols, "CDS", 4)
            rows += _rows_for(self.fno_symbols, "FNO", 2)
            rows += _rows_for(self.crypto_symbols, "CRYPTO", 2)
            rows += _rows_for(self.us_symbols, "US", 2)
            return rows

        def get_rows_by_market(self, market):
            return [r for r in self.get_rows() if r["market"] == market]

    engine = SimEngine()

    # ── NSE demo stocks ───────────────────────────────────────
    nse_demo = [
        ("SUZLON-EQ",  42.0),
        ("IRFC-EQ",   195.0),
        ("RVNL-EQ",   380.0),
        ("NHPC-EQ",    88.0),
        ("SAIL-EQ",   115.0),
        ("NALCO-EQ",  198.0),
        ("BEL-EQ",    295.0),
        ("BHEL-EQ",   240.0),
    ]
    for sym, price in nse_demo:
        engine.nse_symbols[sym] = SymbolState(sym, f"nse_{sym}", shared_ml)

    # ── MCX demo commodities ──────────────────────────────────
    mcx_demo = [
        ("CRUDEOIL-I",   6500.0),
        ("NATURALGAS-I",  250.0),
        ("GOLD-I",      72000.0),
        ("SILVER-I",     85000.0),
        ("COPPER-I",      800.0),
        ("ZINC-I",        260.0),
        ("ALUMINIUM-I",   220.0),
        ("NICKEL-I",     1650.0),
    ]
    for sym, price in mcx_demo:
        engine.mcx_symbols[sym] = MCXSymbolState(sym, f"mcx_{sym}", shared_ml)

    # ── CDS demo currency pairs ────────────────────────────────
    cds_demo = [
        ("USDINR-I",  87.50),
        ("EURINR-I",  94.80),
        ("GBPINR-I", 110.20),
        ("JPYINR-I",   0.57),
    ]
    for sym, price in cds_demo:
        engine.cds_symbols[sym] = CDSSymbolState(sym, f"cds_{sym}", shared_ml)

    # ── FNO demo index futures ─────────────────────────────────
    fno_demo = [
        ("NIFTY-I",     24800.0),
        ("BANKNIFTY-I", 51200.0),
        ("FINNIFTY-I",  23600.0),
    ]
    for sym, price in fno_demo:
        engine.fno_symbols[sym] = FNOSymbolState(sym, f"fno_{sym}", shared_ml)

    # ── CRYPTO demo pairs ───────────────────────────────────────
    crypto_demo = [
        ("BTCUSDT", 62000.0),
        ("ETHUSDT",  3400.0),
        ("SOLUSDT",   145.0),
        ("BNBUSDT",   580.0),
    ]
    for sym, price in crypto_demo:
        engine.crypto_symbols[sym] = CryptoSymbolState(sym, sym, shared_ml)

    # ── US demo stocks ─────────────────────────────────────────
    us_demo = [
        ("AAPL",   225.0),
        ("MSFT",   420.0),
        ("GOOGL",  175.0),
        ("AMZN",   185.0),
        ("TSLA",   250.0),
        ("NVDA",   130.0),
        ("META",   560.0),
    ]
    for sym, price in us_demo:
        engine.us_symbols[sym] = USSymbolState(sym, sym, shared_ml)

    # ── Price simulator ───────────────────────────────────────
    class StockSim:
        def __init__(self, price, vol_scale=1.0):
            self.price      = price
            self.drift      = random.uniform(-0.0002, 0.0002)
            self.vol        = random.uniform(0.0005, 0.002) * vol_scale
            self._timer     = random.randint(30, 120)

        def next_price(self):
            self._timer -= 1
            if self._timer <= 0:
                self.drift  = random.uniform(-0.001, 0.001)
                self._timer = random.randint(40, 150)
            shock       = random.gauss(0, 1)
            self.price *= math.exp(self.drift + self.vol * shock)
            return round(self.price, 2)

    nse_sims = {sym: StockSim(p)         for sym, p in nse_demo}
    mcx_sims = {sym: StockSim(p, 1.5)   for sym, p in mcx_demo}  # commodities more volatile
    cds_sims = {sym: StockSim(p, 0.3)   for sym, p in cds_demo}  # currency pairs move slower
    fno_sims = {sym: StockSim(p, 1.2)   for sym, p in fno_demo}  # index futures moderately volatile
    crypto_sims = {sym: StockSim(p, 2.0) for sym, p in crypto_demo}  # crypto: most volatile of all
    us_sims = {sym: StockSim(p, 1.0)     for sym, p in us_demo}      # US equities: baseline volatility

    def simulate():
        while True:
            # NSE ticks
            for sym, state in engine.nse_symbols.items():
                price     = nse_sims[sym].next_price()
                ltq       = random.randint(100, 5000)
                bid_price = round(price - random.uniform(0.05, 0.2), 2)
                ask_price = round(price + random.uniform(0.05, 0.2), 2)
                bid_qty   = random.randint(1_000_000, 5_000_000)
                ask_qty   = random.randint(1_000_000, 5_000_000)
                state.on_tick(price, ltq, bid_price, bid_qty, ask_price, ask_qty)

            # MCX ticks (smaller lot qty, larger price moves)
            for sym, state in engine.mcx_symbols.items():
                price     = mcx_sims[sym].next_price()
                ltq       = random.randint(1, 50)
                spread    = price * 0.001
                bid_price = round(price - spread, 2)
                ask_price = round(price + spread, 2)
                bid_qty   = random.randint(1, 200)
                ask_qty   = random.randint(1, 200)
                state.on_tick(price, ltq, bid_price, bid_qty, ask_price, ask_qty)

            # CDS ticks (currency — fine decimals, tight spread)
            for sym, state in engine.cds_symbols.items():
                price     = cds_sims[sym].next_price()
                ltq       = random.randint(1, 100)
                spread    = price * 0.0005
                bid_price = round(price - spread, 4)
                ask_price = round(price + spread, 4)
                bid_qty   = random.randint(1, 500)
                ask_qty   = random.randint(1, 500)
                state.on_tick(price, ltq, bid_price, bid_qty, ask_price, ask_qty)

            # FNO ticks (index futures — larger notional, moderate spread)
            for sym, state in engine.fno_symbols.items():
                price     = fno_sims[sym].next_price()
                ltq       = random.randint(50, 2000)
                spread    = price * 0.0005
                bid_price = round(price - spread, 2)
                ask_price = round(price + spread, 2)
                bid_qty   = random.randint(50, 5000)
                ask_qty   = random.randint(50, 5000)
                state.on_tick(price, ltq, bid_price, bid_qty, ask_price, ask_qty)

            # CRYPTO ticks (24/7, most volatile, tight spread relative to price)
            for sym, state in engine.crypto_symbols.items():
                price     = crypto_sims[sym].next_price()
                ltq       = 1
                spread    = price * 0.0003
                bid_price = round(price - spread, 2)
                ask_price = round(price + spread, 2)
                bid_qty   = round(random.uniform(0.1, 5.0), 3)
                ask_qty   = round(random.uniform(0.1, 5.0), 3)
                state.on_tick(price, ltq, bid_price, bid_qty, ask_price, ask_qty)

            # US ticks (regular US equity volatility, tight spread)
            for sym, state in engine.us_symbols.items():
                price     = us_sims[sym].next_price()
                ltq       = random.randint(1, 500)
                spread    = price * 0.0004
                bid_price = round(price - spread, 2)
                ask_price = round(price + spread, 2)
                bid_qty   = random.randint(100, 5000)
                ask_qty   = random.randint(100, 5000)
                state.on_tick(price, ltq, bid_price, bid_qty, ask_price, ask_qty)

            time.sleep(0.5)

    threading.Thread(target=simulate, daemon=True).start()
    logger.info(
        f"Simulating {len(nse_demo)} NSE stocks + {len(mcx_demo)} MCX commodities "
        f"+ {len(cds_demo)} currency pairs + {len(fno_demo)} index futures "
        f"+ {len(crypto_demo)} crypto pairs + {len(us_demo)} US stocks"
    )

# ─────────────────────────────────────────────────────────────
else:
    logger.info("=== LIVE MODE (NSE + MCX) ===")
    engine = Engine()
    threading.Thread(target=engine.run, daemon=True).start()

# ── Start dashboard servers ───────────────────────────────────
set_engine(engine)
start_servers()

logger.info("Dashboard → http://127.0.0.1:5000")
logger.info("Press Ctrl+C to stop")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    logger.info("Stopped")
