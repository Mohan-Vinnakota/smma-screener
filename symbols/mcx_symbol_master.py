"""
mcx_symbol_master.py
Downloads the Angel One symbol master and extracts active MCX futures.
Run this once to generate mcx_symbols.csv.
It automatically picks the nearest expiry contract for each commodity.

Usage:
    python mcx_symbol_master.py
"""

import requests
import pandas as pd
from datetime import datetime
from config import MCX_SYMBOLS, MCX_SYMBOL_FILE

URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

def download_mcx_symbols():
    print("Downloading Angel One symbol master...")
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    print(f"Total symbols in master: {len(df)}")

    # Filter MCX futures only
    mcx = df[
        (df["exch_seg"] == "MCX") &
        (df["instrumenttype"] == "FUTCOM")
    ].copy()

    print(f"MCX FUTCOM contracts found: {len(mcx)}")

    # Keep only our target commodities
    mcx["base_name"] = mcx["name"].str.upper().str.strip()
    mcx = mcx[mcx["base_name"].isin([s.upper() for s in MCX_SYMBOLS])].copy()
    print(f"Matching our MCX_SYMBOLS list: {len(mcx)}")

    if mcx.empty:
        print("\n⚠️  No matching symbols found.")
        print("Names available in MCX FUTCOM:")
        all_mcx = df[(df["exch_seg"] == "MCX") & (df["instrumenttype"] == "FUTCOM")]
        print(all_mcx["name"].unique()[:30])
        return

    # Parse expiry date
    mcx["expiry_dt"] = pd.to_datetime(mcx["expiry"], format="%d%b%Y", errors="coerce")

    today = datetime.now()
    mcx = mcx[mcx["expiry_dt"] >= today].copy()

    # For each base commodity pick nearest expiry (front month)
    nearest = (
        mcx.sort_values("expiry_dt")
           .groupby("base_name")
           .first()
           .reset_index()
    )

    result = nearest[["token", "symbol", "name", "expiry", "lotsize", "tick_size"]].copy()
    result.columns = ["token", "symbol", "name", "expiry", "lotsize", "tick_size"]

    result.to_csv(MCX_SYMBOL_FILE, index=False)
    print(f"\n✅ Saved {len(result)} MCX contracts to {MCX_SYMBOL_FILE}")
    print()
    print(result[["symbol", "name", "expiry", "lotsize"]].to_string(index=False))

if __name__ == "__main__":
    download_mcx_symbols()
