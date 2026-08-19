"""
engine.py — Market Manager
Coordinates all markets: NSE Equity + MCX Commodities + NSE Currency (CDS)
+ NSE F&O + Crypto (Binance) + US Equities (Alpaca).
Each market runs in its own thread.
Shared ML model across all markets.
get_rows() merges all markets for the dashboard.
"""

import threading
import json
import pyotp
from SmartApi import SmartConnect

from ml_model import MLModel
from market_nse_equity import NSEEquityMarket
from market_mcx import MCXMarket
from market_nse_currency import CDSMarket
from market_nse_fno import FNOMarket
from market_crypto import CryptoMarket
from market_us import USMarket
from database import init_db
from logger import logger
from config import ML_MIN_SAMPLES

# ── Credentials ───────────────────────────────────────────────
with open("credentials.json") as f:
    creds = json.load(f)

API_KEY   = creds["api_key"]
CLIENT_ID = creds["client_id"]
PASSWORD  = creds["password"]
TOTP_KEY  = creds["totp_key"]


class Engine:
    """
    Top-level market manager.
    Owns login, shared ML model, and all market modules.
    """

    def __init__(self):
        self.api      = None
        self.jwt      = None
        self.feed_tok = None
        self.ml       = MLModel(min_samples=ML_MIN_SAMPLES)
        self.markets  = {}   # name → market instance
        init_db()

    # ── Login ─────────────────────────────────────────────────
    def login(self):
        self.api = SmartConnect(api_key=API_KEY)
        totp     = pyotp.TOTP(TOTP_KEY).now()
        data     = self.api.generateSession(CLIENT_ID, PASSWORD, totp)
        if data["status"]:
            self.jwt      = data["data"]["jwtToken"]
            self.feed_tok = self.api.getfeedToken()
            logger.info("Login successful!")
            return True
        logger.error(f"Login failed: {data}")
        return False

    # ── Start all markets ─────────────────────────────────────
    def run(self):
        # Crypto needs no Angel One login — public Binance data.
        # Start it independently so a crypto-only run still works
        # even if Angel One login fails.
        crypto = CryptoMarket(self.ml)
        self.markets["CRYPTO"] = crypto
        threading.Thread(target=crypto.start, daemon=True, name="CRYPTO").start()
        logger.info("Crypto (Binance) market started")

        # US Equities also need no Angel One login — Alpaca uses its
        # own API key/secret. Started independently for the same
        # reason as crypto: works even if Angel One login fails, and
        # runs happily outside NSE hours since ET and IST don't overlap much.
        us = USMarket(self.ml)
        self.markets["US"] = us
        threading.Thread(target=us.start, daemon=True, name="US").start()
        logger.info("US Equities (Alpaca) market started")

        if not self.login():
            logger.warning("Angel One login failed — running CRYPTO + US only")
            return

        # NSE Equity (09:15 – 15:30)
        nse = NSEEquityMarket(self.api, self.jwt, self.feed_tok, self.ml)
        self.markets["NSE"] = nse
        threading.Thread(target=nse.start, daemon=True, name="NSE").start()
        logger.info("NSE Equity market started")

        # MCX Commodities (09:00 – 23:30)
        mcx = MCXMarket(self.api, self.jwt, self.feed_tok, self.ml)
        self.markets["MCX"] = mcx
        threading.Thread(target=mcx.start, daemon=True, name="MCX").start()
        logger.info("MCX Commodities market started")

        # NSE Currency Derivatives (09:00 – 17:00)
        cds = CDSMarket(self.api, self.jwt, self.feed_tok, self.ml)
        self.markets["CDS"] = cds
        threading.Thread(target=cds.start, daemon=True, name="CDS").start()
        logger.info("NSE Currency market started")

        # NSE F&O index futures (09:15 – 15:30)
        fno = FNOMarket(self.api, self.jwt, self.feed_tok, self.ml)
        self.markets["FNO"] = fno
        threading.Thread(target=fno.start, daemon=True, name="FNO").start()
        logger.info("NSE F&O market started")

    # ── Dashboard data ────────────────────────────────────────
    def get_rows(self):
        """Merge rows from all active markets."""
        rows = []
        for market in self.markets.values():
            rows.extend(market.get_rows())
        return rows

    def get_rows_by_market(self, market_name):
        m = self.markets.get(market_name)
        return m.get_rows() if m else []
