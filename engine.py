import pandas as pd
import threading
import time
import json 
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
import pyotp
from indicators import CrossoverDetector
from tick_store import TickStore
from practice import Tick
from ml_model import MLModel, CrossoverRecord
from database import init_db, save_signal, update_signal_exit, save_screened_stock
from logger import logger
from config import (
    LTP_MIN, LTP_MAX, MIN_BID_QTY, MIN_ASK_QTY,
    TICK_STORE_MINUTES, NSE_SYMBOL_FILE, WS_RECONNECT_DELAY
)

# ── Config ────────────────────────────────────────────────────
with open("credentials.json") as f:
    creds = json.load(f)

API_KEY   = creds["api_key"]
CLIENT_ID = creds["client_id"]
PASSWORD  = creds["password"]
TOTP_KEY  = creds["totp_key"]


# ── Shared ML model ───────────────────────────────────────────
ml = MLModel(min_samples=50)

# ── Per symbol state ──────────────────────────────────────────
class SymbolState:
    def __init__(self, symbol, token):
        self.symbol         = symbol
        self.token          = token
        self.detector       = CrossoverDetector()
        self.store          = TickStore()
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

        pred, conf, reason = ml.predict(record)
        record.predicted   = pred
        record.confidence  = round(float(conf), 2)
        record.reason      = reason

        self.signal        = signal
        self.ml_verdict    = "ACCEPT" if pred else "AVOID"
        self.ml_confidence = round(float(conf), 2)
        self.ml_reason     = reason
        self.open_trade    = record

        logger.info(f"🚨 SIGNAL {self.symbol}: {signal} @ ₹{ltp} | {self.ml_verdict} ({conf:.0%})")

        save_signal(
            symbol=self.symbol,
            signal=signal,
            entry_ltp=ltp,
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
        ml.record_outcome(trade)

        result = "WIN ✅" if trade.profitable else "LOSS ❌"
        logger.info(f"📊 CLOSED {self.symbol} {trade.signal} | "
            f"Entry ₹{trade.entry_ltp} → Exit ₹{exit_ltp} | "
            f"P&L ₹{trade.pnl:.2f} | {result}")
        self.open_trade = None

        update_signal_exit(
            symbol=self.symbol,
            signal=trade.signal,
            exit_ltp=exit_ltp,
            pnl=trade.pnl,
            profitable=trade.profitable
        )


# ── Engine ────────────────────────────────────────────────────
class Engine:
    def __init__(self):
        self.api        = None
        self.feed_token = None
        self.jwt_token  = None
        self.symbols    = {}
        self.token_map  = {}
        self._ws_tokens = []    # saved for reconnect
        init_db()

    def login(self):
        self.api   = SmartConnect(api_key=API_KEY)
        totp       = pyotp.TOTP(TOTP_KEY).now()
        data       = self.api.generateSession(CLIENT_ID, PASSWORD, totp)
        if data["status"]:
            self.jwt_token  = data["data"]["jwtToken"]
            self.feed_token = self.api.getfeedToken()
            logger.info("Login successful!")
            return True
        logger.error(f"Login failed: {data}")
        return False

    def screen(self):
        df         = pd.read_csv("nse_symbols.csv")
        token_list = df["token"].astype(str).tolist()
        passed     = []

        logger.info("Screening stocks...")
        for i in range(0, len(token_list), 50):
            batch = token_list[i:i+50]
            try:
                resp = self.api.getMarketData(
                    mode="FULL",
                    exchangeTokens={"NSE": batch}
                )
                if resp["status"]:
                    for q in resp["data"]["fetched"]:
                        ltp     = q.get("ltp", 0)
                        bid_qty = q["depth"]["buy"][0]["quantity"] if q.get("depth") else 0
                        ask_qty = q["depth"]["sell"][0]["quantity"] if q.get("depth") else 0
                        if LTP_MIN <= ltp <= LTP_MAX:
                            if bid_qty > MIN_BID_QTY and ask_qty > MIN_ASK_QTY:
                                passed.append({
                                    "symbol":  q["tradingSymbol"],
                                    "token":   q["symbolToken"],
                                    "ltp":     ltp,
                                    "bid_qty": bid_qty,
                                    "ask_qty": ask_qty,
                                })
            except Exception as e:
                logger.error(f"Batch error: {e}")
            time.sleep(0.1)

        logger.info(f"Passed filter: {len(passed)} stocks")
        for s in passed:
            sym = s["symbol"]
            tok = str(s["token"])
            if sym not in self.symbols:
                self.symbols[sym] = SymbolState(sym, tok)
            self.token_map[tok] = sym
            save_screened_stock(sym, s["ltp"], s.get("bid_qty", 0), s.get("ask_qty", 0))
        return [s["token"] for s in passed]

    def on_tick(self, wsapp, message):
        try:
            ltp   = message["last_traded_price"] / 100
            ltq   = message.get("last_traded_quantity", 1)
            token = str(message.get("token", ""))

            symbol = self.token_map.get(token, "")
            if not symbol:
                return

            bid_price = message["best_5_buy_data"][0]["price"] / 100
            bid_qty   = message["best_5_buy_data"][0]["quantity"]
            ask_price = message["best_5_sell_data"][0]["price"] / 100
            ask_qty   = message["best_5_sell_data"][0]["quantity"]

            if symbol in self.symbols:
                self.symbols[symbol].on_tick(
                    ltp, ltq, bid_price, bid_qty, ask_price, ask_qty
                )
        except Exception as e:
            logger.error(f"Tick error: {e}")

    def start_websocket(self, tokens):
        self._ws_tokens = tokens    # save for reconnect
        self._connect_websocket()

    def _connect_websocket(self):
        sws = SmartWebSocketV2(
            self.jwt_token, API_KEY, CLIENT_ID, self.feed_token
        )

        def on_open(wsapp):
            logger.info("WebSocket connected!")
            sws.subscribe("engine", 3,
                          [{"exchangeType": 1, "tokens": self._ws_tokens}])

        def on_error(wsapp, error):
            logger.error(f"WS Error: {error}")

        def on_close(wsapp):
            logger.warning("WS Closed — reconnecting in 5 seconds...")
            time.sleep(5)
            try:
                self._connect_websocket()
            except Exception as e:
                logger.error(f"Reconnect failed: {e}")

        sws.on_open  = on_open
        sws.on_data  = self.on_tick
        sws.on_error = on_error
        sws.on_close = on_close
        sws.connect()

    def get_rows(self):
        rows = []
        for sym, state in self.symbols.items():
            if state.last_ltp is None:
                continue
            f = state.detector.fast.value
            s = state.detector.slow.value
            rows.append({
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

    def run(self):
        if not self.login():
            return
        tokens = self.screen()
        if tokens:
            self.start_websocket(tokens)
        else:
            print("No stocks passed filter — try during market hours")


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Simulation Test ---")
    state = SymbolState("TESTSTOCK", "9999")
    prices = (
        [100] * 130 +
        [100 + i * 0.2 for i in range(100)] +
        [120] * 50 +
        [120 - i * 0.3 for i in range(100)]
    )
    for price in prices:
        state.on_tick(price)

    print("\n--- Starting Real Engine ---")
    engine = Engine()
    engine.run()