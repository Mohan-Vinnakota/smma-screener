"""
us_symbol_master.py
Validates the configured US_SYMBOLS list against Alpaca's live
/v2/assets endpoint and saves the confirmed, currently-tradable
tickers to us_symbols.csv.

Like Binance, Alpaca symbols are plain strings (e.g. "AAPL") — no
token lookup or expiry logic needed. This script exists mainly to
(a) confirm each ticker is real and currently tradable on Alpaca,
and (b) keep the same "run once, generates a CSV" pattern as the
other markets so market_us.py can load symbols the same way.

Requires alpaca_api_key / alpaca_api_secret in credentials.json —
the same keypair works for both the Trading API (used here) and the
Market Data WebSocket (used by market_us.py). Free paper-trading
keys from https://alpaca.markets are enough; no funding required
just to watch prices and generate signals.

Usage:
    python us_symbol_master.py
"""

import json
import requests
import pandas as pd
from config import US_SYMBOLS, US_SYMBOL_FILE, ALPACA_ASSETS_URL

with open("credentials.json") as f:
    creds = json.load(f)

HEADERS = {
    "APCA-API-KEY-ID":     creds["alpaca_api_key"],
    "APCA-API-SECRET-KEY": creds["alpaca_api_secret"],
}


def download_us_symbols():
    print("Fetching Alpaca tradable assets...")
    resp = requests.get(
        ALPACA_ASSETS_URL,
        headers=HEADERS,
        params={"status": "active", "asset_class": "us_equity"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    all_assets = {a["symbol"]: a for a in data}
    print(f"Total active US equity assets on Alpaca: {len(all_assets)}")

    rows = []
    missing = []
    for sym in US_SYMBOLS:
        info = all_assets.get(sym.upper())
        if info is None:
            missing.append(sym)
            continue
        if not info.get("tradable"):
            print(f"⚠️  {sym} found but not currently tradable — skipping")
            continue
        rows.append({
            "symbol":     info["symbol"],
            "name":       info.get("name", ""),
            "exchange":   info.get("exchange", ""),
            "tradable":   info["tradable"],
        })

    if missing:
        print(f"\n⚠️  Not found on Alpaca: {missing}")
        print("Check spelling in config.py US_SYMBOLS (must match Alpaca's exact ticker, e.g. 'AAPL').")

    if not rows:
        print("\n❌ No valid symbols found — nothing saved.")
        return

    result = pd.DataFrame(rows)
    result.to_csv(US_SYMBOL_FILE, index=False)
    print(f"\n✅ Saved {len(result)} US tickers to {US_SYMBOL_FILE}")
    print()
    print(result.to_string(index=False))


if __name__ == "__main__":
    download_us_symbols()
