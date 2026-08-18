import time
import threading
import sys
import math
import random
import logging as _logging

from logger import logger
from engine import Engine
from server import start_servers, set_engine
from database import init_db

_logging.getLogger("werkzeug").setLevel(_logging.ERROR)

SIMULATE = "--simulate" in sys.argv
init_db()

# ─────────────────────────────────────────────────────────────
if SIMULATE:
    logger.info("=== SIMULATION MODE (NSE + MCX) ===")

    from market_nse_equity import SymbolState
    from market_mcx import MCXSymbolState
    from ml_model import MLModel
    from config import ML_MIN_SAMPLES

    shared_ml = MLModel(min_samples=ML_MIN_SAMPLES)

    # Build a fake engine that exposes get_rows()
    class SimEngine:
        def __init__(self):
            self.nse_symbols = {}
            self.mcx_symbols = {}

        def get_rows(self):
            rows = []
            for state in self.nse_symbols.values():
                if state.last_ltp is None:
                    continue
                f = state.detector.fast.value
                s = state.detector.slow.value
                rows.append({
                    "market":      "NSE",
                    "symbol":      state.symbol,
                    "ltp":         round(state.last_ltp, 2),
                    "smma_fast":   round(f, 2) if f else None,
                    "smma_slow":   round(s, 2) if s else None,
                    "signal":      state.signal,
                    "ml_verdict":  state.ml_verdict,
                    "ml_conf":     f"{state.ml_confidence:.0%}" if state.ml_confidence else None,
                    "ml_reason":   state.ml_reason,
                    "bid_price":   round(state.last_bid_price, 2),
                    "bid_qty":     state.last_bid_qty,
                    "ask_price":   round(state.last_ask_price, 2),
                    "ask_qty":     state.last_ask_qty,
                    "etq_5m":      state.store.etq(5),
                    "etq_20m":     state.store.etq(20),
                    "etq_60m":     state.store.etq(60),
                    "avg_ltp_20m": round(state.store.avg_ltp(20) or 0, 2),
                    "avg_ltp_60m": round(state.store.avg_ltp(60) or 0, 2),
                })
            for state in self.mcx_symbols.values():
                if state.last_ltp is None:
                    continue
                f = state.detector.fast.value
                s = state.detector.slow.value
                rows.append({
                    "market":      "MCX",
                    "symbol":      state.symbol,
                    "ltp":         round(state.last_ltp, 2),
                    "smma_fast":   round(f, 2) if f else None,
                    "smma_slow":   round(s, 2) if s else None,
                    "signal":      state.signal,
                    "ml_verdict":  state.ml_verdict,
                    "ml_conf":     f"{state.ml_confidence:.0%}" if state.ml_confidence else None,
                    "ml_reason":   state.ml_reason,
                    "bid_price":   round(state.last_bid_price, 2),
                    "bid_qty":     state.last_bid_qty,
                    "ask_price":   round(state.last_ask_price, 2),
                    "ask_qty":     state.last_ask_qty,
                    "etq_5m":      state.store.etq(5),
                    "etq_20m":     state.store.etq(20),
                    "etq_60m":     state.store.etq(60),
                    "avg_ltp_20m": round(state.store.avg_ltp(20) or 0, 2),
                    "avg_ltp_60m": round(state.store.avg_ltp(60) or 0, 2),
                })
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

            time.sleep(0.5)

    threading.Thread(target=simulate, daemon=True).start()
    logger.info(f"Simulating {len(nse_demo)} NSE stocks + {len(mcx_demo)} MCX commodities")

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
