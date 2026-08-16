import json
import asyncio
import threading
import websockets
from flask import Flask, jsonify, send_from_directory
from config import HTTP_HOST, HTTP_PORT, WS_HOST, WS_PORT
from logger import logger
import logging as _logging
_logging.getLogger("werkzeug").setLevel(_logging.ERROR)

app = Flask(__name__)
_engine = None

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
    logger.info(f"Dashboard → http://{HTTP_HOST}:{HTTP_PORT}")
    import logging as _logging
    log = _logging.getLogger("werkzeug")
    log.setLevel(_logging.ERROR)
    log.disabled = True
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, use_reloader=False)

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
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        logger.info(f"WebSocket → ws://{WS_HOST}:{WS_PORT}")
        await broadcast_loop()

def run_websocket():
    asyncio.run(ws_main())

# ── Start both servers ────────────────────────────────────────
def start_servers():
    threading.Thread(target=run_flask,     daemon=True).start()
    threading.Thread(target=run_websocket, daemon=True).start()

