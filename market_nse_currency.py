"""
market_nse_currency.py
NSE Currency Derivatives market module.
Handles: symbol loading, screening, WebSocket tick stream, SMMA signals.
Runs parallel to market_nse_equity.py / market_mcx.py — feeds into same
ML model + dashboard.

CDS Exchange hours: 09:00 – 17:00 IST
Angel One WebSocket exchangeType for CDS (currency) = 13
"""

import os
import time
import threading
import pandas as pd
from datetime import datetime
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

from indicators import CrossoverDetector
from tick_store import TickStore
from practice import Tick
from ml_model import MLModel, CrossoverRecord
from database import save_signal, update_signal_exit, save_screened_stock
from logger import logger
from config import (
    CDS_SYMBOL_FILE, CDS_MIN_BID_QTY, CDS_MIN_ASK_QTY,
    CDS_OPEN_H, CDS_OPEN_M, CDS_CLOSE_H, CDS_CLOSE_M,
    WS_RECONNECT_DELAY, TICK_STORE_MINUTES
)
from telegram_alert import send_alert, format_signal_alert

MARKET = "CDS"
EXCHANGE_TYPE = 13   # Angel One CDS (currency) exchange type for WebSocket


# ── Market hours check ────────────────────────────────────────
def is_cds_open():
    now = datetime.now()
    open_mins  = CDS_OPEN_H  * 60 + CDS_OPEN_M
    close_mins = CDS_CLOSE_H * 60 + CDS_CLOSE_M
    now_mins   = now.hour * 60 + now.minute
    return open_mins <= now_mins < close_mins


# ── Per-symbol state (same pattern as NSE / MCX) ──────────────
class CDSSymbolState:
    def __init__(self, symbol, token, ml_model):
        self.symbol         = symbol
        self.token          = token
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
        record.confidence  = round(float(conf), 2)
        record.reason      = reason

        self.signal        = signal
        self.ml_verdict    = "ACCEPT" if pred else "AVOID"
        self.ml_confidence = round(float(conf), 2)
        self.ml_reason     = reason
        self.open_trade    = record

        logger.info(
            f"💱 CDS SIGNAL {self.symbol}: {signal} @ ₹{ltp:,.4f} "
            f"| {self.ml_verdict} ({conf:.0%})"
        )

        try:
            msg = format_signal_alert(
                f"[CDS] {self.symbol}", signal, ltp,
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
            confidence=round(float(conf), 2),
            reason=reason
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
            f"📊 CDS CLOSED {self.symbol} {trade.signal} | "
            f"Entry ₹{trade.entry_ltp:,.4f} → Exit ₹{exit_ltp:,.4f} | "
            f"P&L ₹{trade.pnl:,.4f} | {result}"
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


# ── CDS Market Engine ──────────────────────────────────────────
class CDSMarket:
    """
    Manages all NSE Currency Derivatives symbols.
    Call start() — it runs in its own thread.
    get_rows() returns current state for the dashboard.
    """

    def __init__(self, api, jwt_token, feed_token, ml_model):
        self.api        = api
        self.jwt_token  = jwt_token
        self.feed_token = feed_token
        self.ml         = ml_model
        self.symbols    = {}    # symbol_name → CDSSymbolState
        self.token_map  = {}    # token_str   → symbol_name
        self._ws_tokens = []

    # ── Symbol loading ────────────────────────────────────────
    def load_symbols(self):
        if not os.path.exists(CDS_SYMBOL_FILE):
            logger.error(
                f"CDS symbol file '{CDS_SYMBOL_FILE}' not found. "
                "Run: python cds_symbol_master.py"
            )
            return []

        df = pd.read_csv(CDS_SYMBOL_FILE)
        logger.info(f"CDS: loaded {len(df)} contracts from {CDS_SYMBOL_FILE}")
        return df

    # ── Screening ─────────────────────────────────────────────
    def screen(self):
        """Fetch latest quotes for all CDS symbols. No LTP filter — pairs vary in scale."""
        df = self.load_symbols()
        if df is None or df.empty:
            return []

        token_list = df["token"].astype(str).tolist()
        passed = []

        logger.info(f"CDS: screening {len(token_list)} contracts...")
        for i in range(0, len(token_list), 50):
            batch = token_list[i:i + 50]
            try:
                resp = self.api.getMarketData(
                    mode="FULL",
                    exchangeTokens={"CDS": batch}
                )
                if resp.get("status"):
                    for q in resp["data"].get("fetched", []):
                        ltp     = q.get("ltp", 0)
                        symbol  = q.get("tradingSymbol", "")
                        token   = str(q.get("symbolToken", ""))
                        bid_qty = 0
                        ask_qty = 0
                        bid_price = 0
                        ask_price = 0
                        if q.get("depth"):
                            buy_depth  = q["depth"].get("buy",  [{}])
                            sell_depth = q["depth"].get("sell", [{}])
                            if buy_depth:
                                bid_qty   = buy_depth[0].get("quantity", 0)
                                bid_price = buy_depth[0].get("price", 0) / 100
                            if sell_depth:
                                ask_qty   = sell_depth[0].get("quantity", 0)
                                ask_price = sell_depth[0].get("price", 0) / 100

                        if ltp > 0 and bid_qty >= CDS_MIN_BID_QTY and ask_qty >= CDS_MIN_ASK_QTY:
                            passed.append({
                                "symbol":    symbol,
                                "token":     token,
                                "ltp":       ltp,
                                "bid_price": bid_price,
                                "ask_price": ask_price,
                                "bid_qty":   bid_qty,
                                "ask_qty":   ask_qty,
                            })
            except Exception as e:
                logger.error(f"CDS batch error: {e}")
            time.sleep(0.1)

        logger.info(f"CDS: {len(passed)} contracts active")
        for s in passed:
            sym = s["symbol"]
            tok = s["token"]
            if sym not in self.symbols:
                self.symbols[sym] = CDSSymbolState(sym, tok, self.ml)
            self.token_map[tok] = sym
            save_screened_stock(sym, s["ltp"], s["bid_qty"], s["ask_qty"], market=MARKET)

        return [s["token"] for s in passed]

    # ── WebSocket ─────────────────────────────────────────────
    def _on_tick(self, wsapp, message):
        try:
            ltp   = message["last_traded_price"] / 100
            ltq   = message.get("last_traded_quantity", 1)
            token = str(message.get("token", ""))

            symbol = self.token_map.get(token, "")
            if not symbol:
                return

            bid_price, bid_qty, ask_price, ask_qty = 0, 0, 0, 0
            buy_data  = message.get("best_5_buy_data",  [])
            sell_data = message.get("best_5_sell_data", [])
            if buy_data:
                bid_price = buy_data[0].get("price", 0) / 100
                bid_qty   = buy_data[0].get("quantity", 0)
            if sell_data:
                ask_price = sell_data[0].get("price", 0) / 100
                ask_qty   = sell_data[0].get("quantity", 0)

            if symbol in self.symbols:
                self.symbols[symbol].on_tick(
                    ltp, ltq, bid_price, bid_qty, ask_price, ask_qty
                )
        except Exception as e:
            logger.error(f"CDS tick error: {e}")

    def _connect_websocket(self):
        sws = SmartWebSocketV2(
            self.jwt_token, self.feed_token  # API_KEY, CLIENT_ID passed via factory
        )

        def on_open(wsapp):
            logger.info("CDS WebSocket connected!")
            sws.subscribe("cds_engine", 3,
                          [{"exchangeType": EXCHANGE_TYPE, "tokens": self._ws_tokens}])

        def on_error(wsapp, error):
            logger.error(f"CDS WS Error: {error}")

        def on_close(wsapp):
            logger.warning(f"CDS WS closed — reconnecting in {WS_RECONNECT_DELAY}s...")
            time.sleep(WS_RECONNECT_DELAY)
            try:
                self._connect_websocket()
            except Exception as e:
                logger.error(f"CDS reconnect failed: {e}")

        sws.on_open  = on_open
        sws.on_data  = self._on_tick
        sws.on_error = on_error
        sws.on_close = on_close
        sws.connect()

    def _rescreen_loop(self):
        while True:
            time.sleep(30 * 60)
            if not is_cds_open():
                logger.info("CDS: market closed — skipping rescreen")
                continue
            logger.info("CDS: re-screening (30-min cycle)...")
            try:
                new_tokens = self.screen()
                if new_tokens:
                    self._ws_tokens = new_tokens
                    logger.info(f"CDS rescreen complete — {len(new_tokens)} contracts active")
                else:
                    logger.warning("CDS rescreen: 0 contracts — keeping existing")
            except Exception as e:
                logger.error(f"CDS rescreen error: {e}")

    def start(self):
        """Call this in its own thread. Blocks on WebSocket."""
        if not is_cds_open():
            logger.info("CDS market is closed right now — will wait for open")
            while not is_cds_open():
                time.sleep(60)
            logger.info("CDS market now open — starting")

        tokens = self.screen()
        if not tokens:
            logger.warning("CDS: no contracts found — check cds_symbols.csv")
            return

        self._ws_tokens = tokens
        threading.Thread(target=self._rescreen_loop, daemon=True).start()
        self._connect_websocket()  # blocks

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
                "ltp":         round(state.last_ltp, 4),
                "smma_fast":   round(f, 4) if f else None,
                "smma_slow":   round(s, 4) if s else None,
                "signal":      state.signal,
                "ml_verdict":  state.ml_verdict,
                "ml_conf":     f"{state.ml_confidence:.0%}" if state.ml_confidence else None,
                "ml_reason":   state.ml_reason,
                "bid_price":   round(state.last_bid_price, 4),
                "bid_qty":     state.last_bid_qty,
                "ask_price":   round(state.last_ask_price, 4),
                "ask_qty":     state.last_ask_qty,
                "etq_5m":      state.store.etq(5),
                "etq_20m":     state.store.etq(20),
                "etq_60m":     state.store.etq(60),
                "avg_ltp_20m": round(state.store.avg_ltp(20) or 0, 4),
                "avg_ltp_60m": round(state.store.avg_ltp(60) or 0, 4),
            })
        return rows
