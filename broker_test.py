from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
import pyotp
import json

with open("credentials.json") as f:
    creds = json.load(f)

API_KEY   = creds["api_key"]
CLIENT_ID = creds["client_id"]
PASSWORD  = creds["password"]
TOTP_KEY  = creds["totp_key"]

api = SmartConnect(api_key=API_KEY)
totp = pyotp.TOTP(TOTP_KEY).now()

data = api.generateSession(CLIENT_ID, PASSWORD, totp)

if data["status"]:
    print("Login successful!")
else:
    print(f"Login failed: {data}")
    exit()

feed_token = api.getfeedToken()


# ── WebSocket ──────────────────────────────────────────────────
def on_data(wsapp, message):
    ltp       = message['last_traded_price'] / 100
    ltq       = message['last_traded_quantity']
    bid_price = message['best_5_buy_data'][0]['price'] / 100
    bid_qty   = message['best_5_buy_data'][0]['quantity']
    ask_price = message['best_5_sell_data'][0]['price'] / 100
    ask_qty   = message['best_5_sell_data'][0]['quantity']
    print(f"LTP: ₹{ltp} | LTQ: {ltq} | Bid: ₹{bid_price} ({bid_qty}) | Ask: ₹{ask_price} ({ask_qty})")


def on_open(wsapp):
    print("WebSocket connected!")
    token_list = [{"exchangeType": 1, "tokens": ["3045"]}]
    sws.subscribe("test_session", 3, token_list)

def on_error(wsapp, error):
    print(f"Error: {error}")

def on_close(wsapp):
    print("WebSocket closed")

sws = SmartWebSocketV2(
    data["data"]["jwtToken"],
    API_KEY,
    CLIENT_ID,
    feed_token
)

sws.on_open  = on_open
sws.on_data  = on_data
sws.on_error = on_error
sws.on_close = on_close

print("Connecting to WebSocket...")
sws.connect()
