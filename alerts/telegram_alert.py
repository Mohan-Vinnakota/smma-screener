import requests
import json
from config import TELEGRAM_ACCEPT_ONLY

with open("credentials.json") as f:
    creds = json.load(f)

TOKEN   = creds["telegram_token"]
CHAT_ID = creds["telegram_chat"]
BASE_URL = f"https://api.telegram.org/bot{TOKEN}"

def send_alert(message: str, verdict: str = None):
    if TELEGRAM_ACCEPT_ONLY and verdict != "ACCEPT":
        return
    try:
        url     = f"{BASE_URL}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        resp    = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            print("Telegram sent ✅")
        else:
            print(f"Telegram failed: {resp.text}")
    except Exception as e:
        print(f"Telegram error: {e}")

def format_signal_alert(symbol, signal, ltp, verdict, confidence):
    emoji = "🟢" if signal == "BUY" else "🔴"
    conf_str = f"{confidence:.0%}" if confidence is not None else "n/a"
    return (
        f"{emoji} <b>SMMA SIGNAL</b>\n"
        f"Stock   : <b>{symbol}</b>\n"
        f"Signal  : <b>{signal}</b>\n"
        f"Price   : ₹{ltp}\n"
        f"ML      : {verdict} ({conf_str})\n"
    )

if __name__ == "__main__":
    msg = format_signal_alert("SUZLON-EQ", "BUY", 42.5, "ACCEPT", 0.75)
    send_alert(msg)