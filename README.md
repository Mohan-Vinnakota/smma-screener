# SMMA Screener

Live multi-market screener (NSE Equity, MCX Commodities, NSE Currency,
NSE F&O, Crypto, US Equities) with SMMA crossover detection and
XGBoost ML predictions.

## Features
- Real-time NSE equity screening (2,488 stocks), LTP filter 30-500 rupees, bid/ask qty filter above 10 lakhs
- MCX commodities (Crude, Gold, Silver, Natural Gas, Copper, Zinc, Aluminium, Nickel, Lead)
- NSE Currency Derivatives (USDINR/EURINR/GBPINR/JPYINR) and F&O index futures (NIFTY/BANKNIFTY/FINNIFTY)
- Crypto via Binance public WebSocket (BTCUSDT/ETHUSDT/SOLUSDT/BNBUSDT), 24/7, no API key needed
- US Equities via Alpaca Market Data WebSocket (AAPL/MSFT/GOOGL/AMZN/TSLA/NVDA/META), 09:30-16:00 US/Eastern
- SMMA 20 and SMMA 120 crossover detection, shared across all markets
- XGBoost ML model predicts trade profitability
- ETQ windows for 5 minutes, 20 minutes and 60 minutes
- Live auto-refreshing dashboard with per-market tabs
- Angel One SmartAPI integration (NSE/MCX/CDS/FNO) — Crypto and US Equities need no Angel One login
- Simulation mode for testing without any broker/API keys

## How to Run

Step 1 - Install dependencies

    pip install -r requirements.txt

Step 2 - Add your credentials to credentials.json file (see below)

Step 3 - Generate symbol lists once per market you'll run live

    python crypto_symbol_master.py
    python us_symbol_master.py
    # NSE/MCX/CDS/FNO symbol files ship pre-generated (nse_symbols.csv etc.)

Step 4 - Run in simulation mode

    python main.py --simulate

Step 5 - Run in live mode during market hours

    python main.py

Step 6 - Open browser and go to http://127.0.0.1:5000

## Tech Stack
- Python
- Flask and WebSockets
- Angel One SmartAPI (NSE/MCX/CDS/FNO)
- Binance public WebSocket (Crypto)
- Alpaca Market Data WebSocket (US Equities)
- XGBoost and Scikit-learn
- HTML CSS and JavaScript

## credentials.json fields
    {
      "api_key": "...",          // Angel One SmartAPI
      "client_id": "...",
      "password": "...",
      "totp_key": "...",
      "telegram_token": "...",
      "telegram_chat": "...",
      "alpaca_api_key": "...",   // Alpaca (free paper-trading key works)
      "alpaca_api_secret": "..."
    }

Crypto needs none of these — Binance's public ticker stream is unauthenticated.

## Important
Never commit credentials.json to GitHub — it's already in .gitignore.