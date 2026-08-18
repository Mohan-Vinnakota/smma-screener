"""
engine.py — Market Manager
Coordinates all markets: NSE Equity + MCX Commodities.
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
        if not self.login():
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
