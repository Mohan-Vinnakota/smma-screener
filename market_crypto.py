"""
market_crypto.py
Crypto market module (Binance).
Handles: symbol loading, WebSocket tick stream (public, no login needed),
SMMA signals. Runs parallel to the NSE/MCX/CDS/FNO modules — feeds into
the same ML model + dashboard.

Crypto trades 24/7 — there is no market-hours gate.
Uses Binance's public combined WebSocket ticker stream — no API key
or account needed just to watch prices and generate signals.

If you later want this to place real orders on your Binance account,
that needs an API key from Binance → Profile → API Management, added
to credentials.json as binance_api_key / binance_api_secret. Nothing
in this file uses that yet — it's pure market-data watching, same as
--simulate mode but with real live prices.
"""

import time
import threading
import json
import pandas as pd
import websocket  # websocket-client package, already in requirements.txt

from indicators import CrossoverDetector
from tick_store import TickStore
from practice import Tick
from ml_model import MLModel, CrossoverRecord
from database import save_signal, update_signal_exit, save_screened_stock
from logger import logger
from config import (
    CRYPTO_SYMBOL_FILE, BINANCE_WS_BASE,
    WS_RECONNECT_DELAY, TICK_STORE_MINUTES
)
from telegram_alert import send_alert, format_signal_alert

MARKET = "CRYPTO"


# ── Market hours check ────────────────────────────────────────
def is_crypto_open():
    """Crypto trades 24/7 — always open. Kept for interface parity
    with the other market modules (engine.py doesn't need to know
    this market never closes)."""
    return True


# ── Per-symbol state (same pattern as NSE / MCX / CDS / FNO) ──
class CryptoSymbolState:
    def __init__(self, symbol, token, ml_model):
        self.symbol         = symbol
        self.token          = token   # Binance has no numeric token; symbol doubles as its own id
        self.ml             = ml_model
        self.detector       = CrossoverDetector()
        self.store          = TickStore(max_minutes=TICK_STORE_MINUTES)
        self.last_ltp       = None
        self.last_ltq       = 0
        self.last_bid_price = 0
        self.last_bid_qty   = 0
        self.last_ask_price = 0
        self.last_ask_qty   = 0
        self.signal         = None
        self.ml_verdict     = None
        self.ml_confidence  = None
        self.ml_reason      = ""
        self.open_trade     = None

    def on_tick(self, ltp, ltq=1, bid_price=0,
                bid_qty=0, ask_price=0, ask_qty=0):
        self.last_ltp       = ltp
        self.last_ltq       = ltq
        self.last_bid_price = bid_price
        self.last_bid_qty   = bid_qty
        self.last_ask_price = ask_price
        self.last_ask_qty   = ask_qty

        tick = Tick(
            symbol=self.symbol,
            ltp=ltp, ltq=ltq,
            bid_price=bid_price, bid_qty=bid_qty,
            ask_price=ask_price, ask_qty=ask_qty
        )
        self.store.add(tick)

        signal = self.detector.update(ltp)
        if signal:
            self._handle_signal(signal, ltp, bid_price, ask_price)

    def _handle_signal(self, signal, ltp, bid_price, ask_price):
        if self.open_trade is not None:
            self._close_trade(ltp)

        avg_ltq_2m = self.store.avg_ltq(2) or 1
        avg_ltq_5m = self.store.avg_ltq(5) or 1
        etq_5m     = self.store.etq(5)
        etq_20m    = self.store.etq(20)
        f          = self.detector.fast.value or ltp
        s          = self.detector.slow.value or ltp

        record = CrossoverRecord(
            symbol=self.symbol,
            signal=signal,
            entry_ltp=ltp,
            smma_fast=f,
            smma_slow=s,
            avg_ltq_2m=avg_ltq_2m,
            avg_ltq_5m=avg_ltq_5m,
            etq_5m=etq_5m,
            etq_20m=etq_20m,
            bid_price=bid_price,
            ask_price=ask_price,
        )

        pred, conf, reason = self.ml.predict(record)
        record.predicted   = pred
        record.confidence  = round(float(conf), 2) if conf is not None else None
        record.reason      = reason

        self.signal        = signal
        # pred is None until the model has enough closed trades to be
        # trusted (see MLModel.is_trusted) — don't show a fake ACCEPT/AVOID.
        self.ml_verdict    = "Learning" if pred is None else ("ACCEPT" if pred else "AVOID")
        self.ml_confidence = round(float(conf), 2) if conf is not None else None
        self.ml_reason     = reason
        self.open_trade    = record
        conf_str            = f"{conf:.0%}" if conf is not None else "n/a"

        logger.info(
            f"🪙 CRYPTO SIGNAL {self.symbol}: {signal} @ ${ltp:,.2f} "
            f"| {self.ml_verdict} ({conf_str})"
        )

        try:
            msg = format_signal_alert(
                f"[CRYPTO] {self.symbol}", signal, ltp,
                self.ml_verdict, self.ml_confidence
            )
            send_alert(msg)
        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")

        save_signal(
            symbol=self.symbol,
            signal=signal,
            entry_ltp=ltp,
            market=MARKET,
            predicted=self.ml_verdict,
            confidence=round(float(conf), 2) if conf is not None else None,
            reason=reason,
            features=record.get_features().tolist()
        )

    def _close_trade(self, exit_ltp):
        trade = self.open_trade
        if trade.signal == "BUY":
            trade.pnl = exit_ltp - trade.entry_ltp
        else:
            trade.pnl = trade.entry_ltp - exit_ltp

        trade.exit_ltp   = exit_ltp
        trade.profitable = 1 if trade.pnl > 0 else 0
        self.ml.record_outcome(trade)

        result = "WIN ✅" if trade.profitable else "LOSS ❌"
        logger.info(
            f"📊 CRYPTO CLOSED {self.symbol} {trade.signal} | "
            f"Entry ${trade.entry_ltp:,.2f} → Exit ${exit_ltp:,.2f} | "
            f"P&L ${trade.pnl:,.2f} | {result}"
        )
        self.open_trade = None

        update_signal_exit(
            symbol=self.symbol,
            signal=trade.signal,
            exit_ltp=exit_ltp,
            pnl=trade.pnl,
            profitable=trade.profitable,
            market=MARKET
        )


# ── Crypto Market Engine ────────────────────────────────────────
class CryptoMarket:
    """
    Manages all crypto symbols via Binance's public WebSocket.
    No login required — no api/jwt_token/feed_token needed, unlike
    the Angel One markets. Call start() — it runs in its own thread.
    get_rows() returns current state for the dashboard.
    """

    def __init__(self, ml_model):
        self.ml         = ml_model
        self.symbols    = {}    # symbol_name → CryptoSymbolState
        self._ws        = None
        self._stop      = False

    # ── Symbol loading ────────────────────────────────────────
    def load_symbols(self):
        import os
        if not os.path.exists(CRYPTO_SYMBOL_FILE):
            logger.error(
                f"Crypto symbol file '{CRYPTO_SYMBOL_FILE}' not found. "
                "Run: python crypto_symbol_master.py"
            )
            return []
        df = pd.read_csv(CRYPTO_SYMBOL_FILE)
        logger.info(f"CRYPTO: loaded {len(df)} pairs from {CRYPTO_SYMBOL_FILE}")
        return df

    def screen(self):
        """Crypto majors trade continuously — no LTP/qty filter needed,
        unlike NSE equity. Just registers each configured symbol."""
        df = self.load_symbols()
        if df is None or len(df) == 0:
            return []

        symbols = df["symbol"].tolist()
        for sym in symbols:
            if sym not in self.symbols:
                self.symbols[sym] = CryptoSymbolState(sym, sym, self.ml)
        logger.info(f"CRYPTO: tracking {len(symbols)} pairs")
        return symbols

    # ── WebSocket ─────────────────────────────────────────────
    def _on_message(self, ws, message):
        try:
            payload = json.loads(message)
            data = payload.get("data", payload)  # combined stream wraps in {"stream","data"}

            symbol = data.get("s", "")
            if symbol not in self.symbols:
                return

            ltp       = float(data.get("c", 0))   # last price
            bid_price = float(data.get("b", 0))   # best bid price
            bid_qty   = float(data.get("B", 0))   # best bid qty
            ask_price = float(data.get("a", 0))   # best ask price
            ask_qty   = float(data.get("A", 0))   # best ask qty

            if ltp <= 0:
                return

            self.symbols[symbol].on_tick(
                ltp, ltq=1, bid_price=bid_price, bid_qty=bid_qty,
                ask_price=ask_price, ask_qty=ask_qty
            )

            save_screened_stock(symbol, ltp, bid_qty, ask_qty, market=MARKET)
        except Exception as e:
            logger.error(f"CRYPTO tick error: {e}")

    def _connect_websocket(self, symbols):
        streams = "/".join(f"{s.lower()}@ticker" for s in symbols)
        url = f"{BINANCE_WS_BASE}/stream?streams={streams}"

        def on_open(ws):
            logger.info(f"CRYPTO WebSocket connected! Streaming {len(symbols)} pairs")

        def on_error(ws, error):
            logger.error(f"CRYPTO WS Error: {error}")

        def on_close(ws, close_status_code, close_msg):
            if self._stop:
                return
            logger.warning(f"CRYPTO WS closed — reconnecting in {WS_RECONNECT_DELAY}s...")
            time.sleep(WS_RECONNECT_DELAY)
            try:
                self._connect_websocket(symbols)
            except Exception as e:
                logger.error(f"CRYPTO reconnect failed: {e}")

        self._ws = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=self._on_message,
            on_error=on_error,
            on_close=on_close,
        )
        self._ws.run_forever()  # blocks

    def start(self):
        """Call this in its own thread. Blocks on WebSocket. No market-hours
        wait — crypto is always open."""
        symbols = self.screen()
        if not symbols:
            logger.warning("CRYPTO: no pairs found — check crypto_symbols.csv")
            return

        threading.Thread(target=self._rescreen_loop, daemon=True).start()
        self._connect_websocket(symbols)

    def _rescreen_loop(self):
        """Periodically re-validate the symbol list (catches new/delisted
        pairs). Less critical than for MCX/CDS since Binance majors rarely
        change, but kept for consistency."""
        while True:
            time.sleep(60 * 60)  # hourly is plenty for crypto majors
            logger.info("CRYPTO: re-checking symbol list...")
            try:
                self.screen()
            except Exception as e:
                logger.error(f"CRYPTO rescreen error: {e}")

    def stop(self):
        self._stop = True
        if self._ws:
            self._ws.close()

    # ── Dashboard rows ────────────────────────────────────────
    def get_rows(self):
        rows = []
        for sym, state in self.symbols.items():
            if state.last_ltp is None:
                continue
            f = state.detector.fast.value
            s = state.detector.slow.value
            rows.append({
                "market":      MARKET,
                "symbol":      sym,
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
