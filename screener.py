import pandas as pd
from SmartApi import SmartConnect
import pyotp
import time
import json
# ── Config ────────────────────────────────────────────────────
with open("credentials.json") as f:
    creds = json.load(f)

API_KEY   = creds["api_key"]
CLIENT_ID = creds["client_id"]
PASSWORD  = creds["password"]
TOTP_KEY  = creds["totp_key"]

LTP_MIN    = 30
LTP_MAX    = 500
MIN_BID_QTY = 1_000_000
MIN_ASK_QTY = 1_000_000

# ── Login ─────────────────────────────────────────────────────
api = SmartConnect(api_key=API_KEY)
totp = pyotp.TOTP(TOTP_KEY).now()
data = api.generateSession(CLIENT_ID, PASSWORD, totp)

if data["status"]:
    print("Login successful!")
else:
    print(f"Login failed: {data}")
    exit()

# ── Load symbols ──────────────────────────────────────────────
symbols = pd.read_csv("nse_symbols.csv")
print(f"Loaded {len(symbols)} symbols")

# ── Batch fetch ───────────────────────────────────────────────
def fetch_batch(tokens):
    try:
        resp = api.getMarketData(
            mode="FULL",
            exchangeTokens={"NSE": tokens}
        )
        if resp["status"]:
            return resp["data"]["fetched"]
        return []
    except Exception as e:
        print(f"Batch error: {e}")
        return []

# ── Screen ────────────────────────────────────────────────────
passed = []
token_list = symbols["token"].astype(str).tolist()
batch_size = 50

print("Screening stocks...")
for i in range(0, len(token_list), batch_size):
    batch = token_list[i:i+batch_size]
    quotes = fetch_batch(batch)

    for q in quotes:
        ltp     = q.get("ltp", 0)
        bid_qty = q["depth"]["buy"][0]["quantity"] if q.get("depth") else 0
        ask_qty = q["depth"]["sell"][0]["quantity"] if q.get("depth") else 0

        if LTP_MIN <= ltp <= LTP_MAX:
            if bid_qty > MIN_BID_QTY and ask_qty > MIN_ASK_QTY:
                passed.append({
                    "symbol": q["tradingSymbol"],
                    "token":  q["symbolToken"],
                    "ltp":    ltp,
                    "bid_qty": bid_qty,
                    "ask_qty": ask_qty,
                })

    time.sleep(0.1)   # avoid rate limiting
    print(f"Processed {min(i+batch_size, len(token_list))}/{len(token_list)}", end="\r")

print(f"\nStocks passing filter: {len(passed)}")
for s in passed:
    print(f"{s['symbol']:20} LTP: {s['ltp']:7.2f} | Bid Qty: {s['bid_qty']:>12,} | Ask Qty: {s['ask_qty']:>12,}")