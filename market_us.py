"""
market_us.py
US Equities market module (Alpaca).
Handles: symbol loading, WebSocket auth + tick stream, SMMA signals.
Runs parallel to the NSE/MCX/CDS/FNO/Crypto modules — feeds into the
same ML model + dashboard.

US Equity hours: 09:30 - 16:00 US/Eastern (regular session only; no
pre/post-market here). Unlike crypto, this needs an Alpaca API
key/secret even just to read prices — add alpaca_api_key and
alpaca_api_secret to credentials.json (free paper-trading keys from
https://alpaca.markets work fine, nothing gets traded here).

Uses Alpaca's Market Data v2 WebSocket (IEX feed, free tier) for
trade + quote ticks.
"""

import time
import json
import threading
import pandas as pd
import websocket  # websocket-client package, already in requirements.txt
from datetime import datetime
from zoneinfo import ZoneInfo

from indicators import CrossoverDetector
from tick_store import TickStore
from practice import Tick
from ml_model import MLModel, CrossoverRecord
from database import save_signal, update_signal_exit, save_screened_stock
from logger import logger
from config import (
    US_SYMBOL_FILE, ALPACA_DATA_WS_BASE, ALPACA_DATA_FEED,
    WS_RECONNECT_DELAY, TICK_STORE_MINUTES,
    US_OPEN_H, US_OPEN_M, US_CLOSE_H, US_CLOSE_M
)
from telegram_alert import send_alert, format_signal_alert

MARKET = "US"
US_EASTERN = ZoneInfo("America/New_York")

with open("credentials.json") as f:
    _creds = json.load(f)

ALPACA_KEY    = _creds.get("alpaca_api_key")
ALPACA_SECRET = _creds.get("alpaca_api_secret")


# ── Market hours check ────────────────────────────────────────
def is_us_open():
    """Regular US equity session, 09:30-16:00, Mon-Fri, US/Eastern.
    Computed against America/New_York directly (not a fixed IST
    offset) so it stays correct across US daylight-saving changes."""
    now = datetime.now(US_EASTERN)
    if now.weekday() >= 5:   # Sat/Sun
        return False
    open_mins  = US_OPEN_H  * 60 + US_OPEN_M
    close_mins = US_CLOSE_H * 60 + US_CLOSE_M
    now_mins   = now.hour * 60 + now.minute
    return open_mins <= now_mins < close_mins


# ── Per-symbol state (same pattern as NSE / MCX / CDS / FNO / Crypto) ──
class USSymbolState:
    def __init__(self, symbol, token, ml_model):
        self.symbol         = symbol
        self.token          = token   # Alpaca has no numeric token; symbol doubles as its own id
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
            f"🇺🇸 US SIGNAL {self.symbol}: {signal} @ ${ltp:,.2f} "
            f"| {self.ml_verdict} ({conf_str})"
        )

        try:
            msg = format_signal_alert(
                f"[US] {self.symbol}", signal, ltp,
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
            f"📊 US CLOSED {self.symbol} {trade.signal} | "
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


# ── US Equities Market Engine ───────────────────────────────────
class USMarket:
    """
    Manages all US equity symbols via Alpaca's Market Data WebSocket.
    Requires alpaca_api_key / alpaca_api_secret in credentials.json —
    unlike crypto, Alpaca's feed is authenticated even for read-only
    market data. Call start() — it runs in its own thread.
    get_rows() returns current state for the dashboard.

    Outside 09:30-16:00 ET the exchange itself is closed, so no
    trade/quote ticks arrive — the WebSocket connection stays open
    (Alpaca allows this) and rows simply stop updating, same as how
    the NSE modules go quiet outside their own market hours.
    """

    def __init__(self, ml_model):
        self.ml      = ml_model
        self.symbols = {}    # symbol_name → USSymbolState
        self._ws     = None
        self._stop   = False

    # ── Symbol loading ────────────────────────────────────────
    def load_symbols(self):
        import os
        if not os.path.exists(US_SYMBOL_FILE):
            logger.error(
                f"US symbol file '{US_SYMBOL_FILE}' not found. "
                "Run: python us_symbol_master.py"
            )
            return None
        df = pd.read_csv(US_SYMBOL_FILE)
        logger.info(f"US: loaded {len(df)} tickers from {US_SYMBOL_FILE}")
        return df

    def screen(self):
        """Configured watchlist, same as crypto — no LTP/qty filter
        needed since these are a fixed set of large-cap tickers."""
        df = self.load_symbols()
        if df is None or len(df) == 0:
            return []

        symbols = df["symbol"].tolist()
        for sym in symbols:
            if sym not in self.symbols:
                self.symbols[sym] = USSymbolState(sym, sym, self.ml)
        logger.info(f"US: tracking {len(symbols)} tickers")
        return symbols

    # ── WebSocket ─────────────────────────────────────────────
    def _on_message(self, ws, message):
        try:
            payload = json.loads(message)
            msgs = payload if isinstance(payload, list) else [payload]

            for data in msgs:
                msg_type = data.get("T")

                if msg_type == "success":
                    logger.info(f"US WS: {data.get('msg')}")
                    continue
                if msg_type == "error":
                    logger.error(f"US WS error: {data.get('msg')} (code {data.get('code')})")
                    continue

                symbol = data.get("S", "")
                if symbol not in self.symbols:
                    continue

                if msg_type == "t":       # trade tick
                    ltp = float(data.get("p", 0))
                    ltq = float(data.get("s", 1))
                    if ltp <= 0:
                        continue
                    state = self.symbols[symbol]
                    state.on_tick(
                        ltp, ltq=ltq,
                        bid_price=state.last_bid_price, bid_qty=state.last_bid_qty,
                        ask_price=state.last_ask_price, ask_qty=state.last_ask_qty,
                    )
                    save_screened_stock(symbol, ltp, state.last_bid_qty, state.last_ask_qty, market=MARKET)

                elif msg_type == "q":     # quote tick — updates bid/ask only
                    state = self.symbols[symbol]
                    state.last_bid_price = float(data.get("bp", 0))
                    state.last_bid_qty   = float(data.get("bs", 0))
                    state.last_ask_price = float(data.get("ap", 0))
                    state.last_ask_qty   = float(data.get("as", 0))

        except Exception as e:
            logger.error(f"US tick error: {e}")

    def _connect_websocket(self, symbols):
        if not ALPACA_KEY or not ALPACA_SECRET:
            logger.error(
                "US: alpaca_api_key / alpaca_api_secret missing from "
                "credentials.json — skipping US market"
            )
            return

        url = f"{ALPACA_DATA_WS_BASE}/{ALPACA_DATA_FEED}"

        def on_open(ws):
            logger.info("US WebSocket connected — authenticating...")
            ws.send(json.dumps({
                "action": "auth",
                "key": ALPACA_KEY,
                "secret": ALPACA_SECRET,
            }))
            ws.send(json.dumps({
                "action": "subscribe",
                "trades": symbols,
                "quotes": symbols,
            }))
            logger.info(f"US: subscribed to {len(symbols)} tickers")

        def on_error(ws, error):
            logger.error(f"US WS Error: {error}")

        def on_close(ws, close_status_code, close_msg):
            if self._stop:
                return
            logger.warning(f"US WS closed — reconnecting in {WS_RECONNECT_DELAY}s...")
            time.sleep(WS_RECONNECT_DELAY)
            try:
                self._connect_websocket(symbols)
            except Exception as e:
                logger.error(f"US reconnect failed: {e}")

        self._ws = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=self._on_message,
            on_error=on_error,
            on_close=on_close,
        )
        self._ws.run_forever()  # blocks

    def start(self):
        """Call this in its own thread. Blocks on WebSocket."""
        symbols = self.screen()
        if not symbols:
            logger.warning("US: no tickers found — check us_symbols.csv")
            return

        if not is_us_open():
            logger.info(
                "US: market currently closed (regular session is "
                "09:30-16:00 US/Eastern, Mon-Fri) — connecting anyway, "
                "rows will populate once trading resumes"
            )

        threading.Thread(target=self._rescreen_loop, daemon=True).start()
        self._connect_websocket(symbols)

    def _rescreen_loop(self):
        """Periodically re-validate the symbol list (catches new/delisted
        tickers). Less critical than for MCX/CDS since this is a fixed
        large-cap watchlist, kept for consistency."""
        while True:
            time.sleep(60 * 60)  # hourly is plenty for a fixed watchlist
            logger.info("US: re-checking symbol list...")
            try:
                self.screen()
            except Exception as e:
                logger.error(f"US rescreen error: {e}")

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
