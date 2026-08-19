"""
crypto_symbol_master.py
Validates the configured CRYPTO_SYMBOLS list against Binance's live
exchangeInfo and saves the confirmed, currently-tradable pairs to
crypto_symbols.csv.

Unlike NSE/MCX/CDS/FNO, Binance symbols are plain strings (e.g. "BTCUSDT")
— no token lookup or expiry/front-month logic needed. This script exists
mainly to (a) confirm each pair is actually live and trading right now,
and (b) keep the same "run once, generates a CSV" pattern as the other
markets so market_crypto.py can load symbols the same way.

Usage:
    python crypto_symbol_master.py
"""

import requests
import pandas as pd
from config import CRYPTO_SYMBOLS, CRYPTO_SYMBOL_FILE, BINANCE_REST_BASE

URL = f"{BINANCE_REST_BASE}/api/v3/exchangeInfo"

def download_crypto_symbols():
    print("Fetching Binance exchange info...")
    resp = requests.get(URL, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    all_symbols = {s["symbol"]: s for s in data.get("symbols", [])}
    print(f"Total symbols on Binance: {len(all_symbols)}")

    rows = []
    missing = []
    for sym in CRYPTO_SYMBOLS:
        info = all_symbols.get(sym.upper())
        if info is None:
            missing.append(sym)
            continue
        if info.get("status") != "TRADING":
            print(f"⚠️  {sym} found but not currently TRADING (status={info.get('status')}) — skipping")
            continue
        rows.append({
            "symbol":     info["symbol"],
            "baseAsset":  info["baseAsset"],
            "quoteAsset": info["quoteAsset"],
            "status":     info["status"],
        })

    if missing:
        print(f"\n⚠️  Not found on Binance: {missing}")
        print("Check spelling in config.py CRYPTO_SYMBOLS (must match Binance's exact symbol, e.g. 'BTCUSDT').")

    if not rows:
        print("\n❌ No valid symbols found — nothing saved.")
        return

    result = pd.DataFrame(rows)
    result.to_csv(CRYPTO_SYMBOL_FILE, index=False)
    print(f"\n✅ Saved {len(result)} crypto pairs to {CRYPTO_SYMBOL_FILE}")
    print()
    print(result.to_string(index=False))

if __name__ == "__main__":
    download_crypto_symbols()
