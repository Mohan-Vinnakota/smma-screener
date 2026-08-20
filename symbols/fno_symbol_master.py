"""
fno_symbol_master.py
Downloads the Angel One symbol master and extracts active NSE F&O
(NFO) index futures — NIFTY, BANKNIFTY, FINNIFTY. Run this once to
generate fno_symbols.csv. It automatically picks the nearest expiry
contract for each index.

Usage:
    python fno_symbol_master.py
"""

import requests
import pandas as pd
from datetime import datetime
from config import FNO_SYMBOLS, FNO_SYMBOL_FILE

URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

def download_fno_symbols():
    print("Downloading Angel One symbol master...")
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    print(f"Total symbols in master: {len(df)}")

    # Filter NFO index futures only
    fno = df[
        (df["exch_seg"] == "NFO") &
        (df["instrumenttype"] == "FUTIDX")
    ].copy()

    print(f"NFO FUTIDX contracts found: {len(fno)}")

    # Keep only our target indices
    fno["base_name"] = fno["name"].str.upper().str.strip()
    fno = fno[fno["base_name"].isin([s.upper() for s in FNO_SYMBOLS])].copy()
    print(f"Matching our FNO_SYMBOLS list: {len(fno)}")

    if fno.empty:
        print("\n⚠️  No matching symbols found.")
        print("Names available in NFO FUTIDX:")
        all_fno = df[(df["exch_seg"] == "NFO") & (df["instrumenttype"] == "FUTIDX")]
        print(all_fno["name"].unique()[:30])
        return

    # Parse expiry date
    fno["expiry_dt"] = pd.to_datetime(fno["expiry"], format="%d%b%Y", errors="coerce")

    today = datetime.now()
    fno = fno[fno["expiry_dt"] >= today].copy()

    # For each index pick nearest expiry (front month)
    nearest = (
        fno.sort_values("expiry_dt")
           .groupby("base_name")
           .first()
           .reset_index()
    )

    result = nearest[["token", "symbol", "name", "expiry", "lotsize", "tick_size"]].copy()
    result.columns = ["token", "symbol", "name", "expiry", "lotsize", "tick_size"]

    result.to_csv(FNO_SYMBOL_FILE, index=False)
    print(f"\n✅ Saved {len(result)} F&O contracts to {FNO_SYMBOL_FILE}")
    print()
    print(result[["symbol", "name", "expiry", "lotsize"]].to_string(index=False))

if __name__ == "__main__":
    download_fno_symbols()
