"""
cds_symbol_master.py
Downloads the Angel One symbol master and extracts active NSE Currency
Derivatives (CDS) futures. Run this once to generate cds_symbols.csv.
It automatically picks the nearest expiry contract for each currency pair.

Usage:
    python cds_symbol_master.py
"""

import requests
import pandas as pd
from datetime import datetime
from config import CDS_SYMBOLS, CDS_SYMBOL_FILE

URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

def download_cds_symbols():
    print("Downloading Angel One symbol master...")
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    print(f"Total symbols in master: {len(df)}")

    # Filter CDS futures only
    cds = df[
        (df["exch_seg"] == "CDS") &
        (df["instrumenttype"] == "FUTCUR")
    ].copy()

    print(f"CDS FUTCUR contracts found: {len(cds)}")

    # Keep only our target currency pairs
    cds["base_name"] = cds["name"].str.upper().str.strip()
    cds = cds[cds["base_name"].isin([s.upper() for s in CDS_SYMBOLS])].copy()
    print(f"Matching our CDS_SYMBOLS list: {len(cds)}")

    if cds.empty:
        print("\n⚠️  No matching symbols found.")
        print("Names available in CDS FUTCUR:")
        all_cds = df[(df["exch_seg"] == "CDS") & (df["instrumenttype"] == "FUTCUR")]
        print(all_cds["name"].unique()[:30])
        return

    # Parse expiry date
    cds["expiry_dt"] = pd.to_datetime(cds["expiry"], format="%d%b%Y", errors="coerce")

    today = datetime.now()
    cds = cds[cds["expiry_dt"] >= today].copy()

    # For each currency pair pick nearest expiry (front month)
    nearest = (
        cds.sort_values("expiry_dt")
           .groupby("base_name")
           .first()
           .reset_index()
    )

    result = nearest[["token", "symbol", "name", "expiry", "lotsize", "tick_size"]].copy()
    result.columns = ["token", "symbol", "name", "expiry", "lotsize", "tick_size"]

    result.to_csv(CDS_SYMBOL_FILE, index=False)
    print(f"\n✅ Saved {len(result)} CDS contracts to {CDS_SYMBOL_FILE}")
    print()
    print(result[["symbol", "name", "expiry", "lotsize"]].to_string(index=False))

if __name__ == "__main__":
    download_cds_symbols()
