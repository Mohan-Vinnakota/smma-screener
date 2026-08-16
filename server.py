import json
import asyncio
import threading
import websockets
from flask import Flask, jsonify, send_from_directory
import os

app = Flask(__name__)

_engine = None   # will be set by main.py

def set_engine(engine):
    global _engine
    _engine = engine

# ── HTTP Routes ───────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("dashboard", "index.html")

@app.route("/api/rows")
def api_rows():
    if _engine is None:
        return jsonify([])
    return jsonify(_engine.get_rows())

# ── Flask thread ──────────────────────────────────────────────
def run_flask():
    print("Dashboard → http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

# ── WebSocket broadcast ───────────────────────────────────────
_clients = set()

async def ws_handler(websocket):
    global _clients         
    _clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        _clients.discard(websocket)
async def broadcast_loop():
    global _clients         
    while True:
        await asyncio.sleep(2)
        if not _clients or _engine is None:
            continue
        payload = json.dumps(_engine.get_rows())
        dead = set()
        for ws in list(_clients):
            try:
                await ws.send(payload)
            except:
                dead.add(ws)
        _clients -= dead
async def ws_main():
    async with websockets.serve(ws_handler, "127.0.0.1", 8765):
        print("WebSocket → ws://127.0.0.1:8765")
        await broadcast_loop()

def run_websocket():
    asyncio.run(ws_main())

# ── Start both servers ────────────────────────────────────────
def start_servers():
    threading.Thread(target=run_flask,     daemon=True).start()
    threading.Thread(target=run_websocket, daemon=True).start()

