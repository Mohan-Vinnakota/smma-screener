import time
import threading
import sys
import logging as _logging
from logger import logger
from engine import Engine
from server import start_servers, set_engine
from database import init_db


# Suppress Flask startup messages
_logging.getLogger("werkzeug").setLevel(_logging.ERROR)

# ── Choose mode ───────────────────────────────────────────────
SIMULATE = "--simulate" in sys.argv
init_db()

if SIMULATE:
    logger.info("=== SIMULATION MODE ===")
    from indicators import CrossoverDetector
    from engine import SymbolState

    # Create engine without broker
    engine = Engine.__new__(Engine)
    engine.symbols = {}

    # Demo stocks
    demo_stocks = [
        ("SUZLON-EQ",  "token1", 42.0),
        ("IRFC-EQ",    "token2", 195.0),
        ("RVNL-EQ",    "token3", 380.0),
        ("NHPC-EQ",    "token4", 88.0),
        ("SAIL-EQ",    "token5", 115.0),
        ("NALCO-EQ",   "token6", 198.0),
        ("BEL-EQ",     "token7", 295.0),
        ("BHEL-EQ",    "token8", 240.0),
    ]

    # Register demo stocks
    for sym, token, price in demo_stocks:
        engine.symbols[sym] = SymbolState(sym, token)

    # Simulate ticks
    import math, random

    class StockSim:
        def __init__(self, price):
            self.price = price
            self.drift = random.uniform(-0.0002, 0.0002)
            self.vol   = random.uniform(0.0005, 0.002)
            self._timer = random.randint(30, 120)

        def next_price(self):
            self._timer -= 1
            if self._timer <= 0:
                self.drift  = random.uniform(-0.001, 0.001)
                self._timer = random.randint(40, 150)
            shock = random.gauss(0, 1)
            self.price *= math.exp(self.drift + self.vol * shock)
            self.price  = max(30.5, self.price)
            return round(self.price, 2)

    sims = {sym: StockSim(price) for sym, _, price in demo_stocks}

    def simulate():
        while True:
            for sym, state in engine.symbols.items():
                price = sims[sym].next_price()
                ltq       = random.randint(100, 5000)
                bid_price = round(price - random.uniform(0.05, 0.2), 2)
                ask_price = round(price + random.uniform(0.05, 0.2), 2)
                bid_qty   = random.randint(1_000_000, 5_000_000)
                ask_qty   = random.randint(1_000_000, 5_000_000)
                state.on_tick(price, ltq, bid_price, bid_qty, ask_price, ask_qty)
            time.sleep(0.5)

    threading.Thread(target=simulate, daemon=True).start()
    logger.info(f"Simulating {len(demo_stocks)} stocks...")

else:
    logger.info("=== LIVE MODE ===")
    engine = Engine()
    threading.Thread(target=engine.run, daemon=True).start()

# ── Start dashboard servers ───────────────────────────────────
set_engine(engine)
start_servers()

logger.info("Press Ctrl+C to stop")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    logger.info("Stopped")
