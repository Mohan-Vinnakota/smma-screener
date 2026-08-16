import requests
import pandas as pd

url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

print("Downloading symbol master...")
response = requests.get(url)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data)
print(f"Total symbols downloaded: {len(df)}")

# Filter only NSE equity stocks
nse_eq = df[
    (df["exch_seg"] == "NSE") &
    (df["instrumenttype"] == "") &
    (df["symbol"].str.endswith("-EQ"))
].copy()

print(f"NSE equity stocks: {len(nse_eq)}")
print(nse_eq[["token", "symbol", "name"]].head(10))

# Save to CSV so we don't download every time
nse_eq.to_csv("nse_symbols.csv", index=False)
print("\nSaved to nse_symbols.csv")