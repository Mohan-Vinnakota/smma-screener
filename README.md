# SMMA Screener

Live NSE stock screener with SMMA crossover detection and XGBoost ML predictions.

## Features
- Real-time NSE stock screening (2,488 stocks)
- LTP filter between 30 and 500 rupees
- Bid and Ask quantity filter above 10 lakhs
- SMMA 20 and SMMA 120 crossover detection
- XGBoost ML model predicts trade profitability
- ETQ windows for 5 minutes, 20 minutes and 60 minutes
- Live auto-refreshing dashboard
- Angel One SmartAPI integration
- Simulation mode for testing without broker

## How to Run

Step 1 - Install dependencies

    pip install -r requirements.txt

Step 2 - Add your credentials to credentials.json file

Step 3 - Run in simulation mode

    python main.py --simulate

Step 4 - Run in live mode during market hours

    python main.py

Step 5 - Open browser and go to http://127.0.0.1:5000

## Tech Stack
- Python
- Flask and WebSockets
- Angel One SmartAPI
- XGBoost and Scikit-learn
- HTML CSS and JavaScript

## Important
Add your Angel One credentials to credentials.json before running.
Never commit credentials.json to GitHub.