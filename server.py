import json
import threading
import time
from flask import Flask, jsonify, send_from_directory, request
from flask_sock import Sock
from config import HTTP_HOST, HTTP_PORT
from logger import logger
from core.database import get_all_signals
import logging as _logging
_logging.getLogger("werkzeug").setLevel(_logging.ERROR)

app = Flask(__name__)
sock = Sock(app)
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
    """All markets combined."""
    if _engine is None:
        return jsonify([])
    return jsonify(_engine.get_rows())

@app.route("/api/rows/<market>")
def api_rows_market(market):
    """Rows for a specific market: /api/rows/NSE or /api/rows/MCX"""
    if _engine is None:
        return jsonify([])
    return jsonify(_engine.get_rows_by_market(market.upper()))

@app.route("/api/signals")
def api_signals():
    """All signals. Optional ?market=NSE or ?market=MCX filter."""
    market = request.args.get("market", None)
    return jsonify(get_all_signals(market=market))

# ── WebSocket (same port as HTTP) ────────────────────────────
_clients = {}          # ws -> queue.Queue
_clients_lock = threading.Lock()

@sock.route("/ws")
def ws_handler(ws):
    import queue
    q = queue.Queue()
    with _clients_lock:
        _clients[ws] = q
    try:
        while True:
            payload = q.get()          # blocks until broadcaster pushes something
            if payload is None:        # sentinel for shutdown/disconnect
                break
            ws.send(payload)
    except Exception:
        pass
    finally:
        with _clients_lock:
            _clients.pop(ws, None)

def broadcast_loop():
    while True:
        time.sleep(2)
        if _engine is None:
            continue
        with _clients_lock:
            if not _clients:
                continue
            payload = json.dumps(_engine.get_rows())
            for q in list(_clients.values()):
                q.put(payload)

# ── Flask thread ──────────────────────────────────────────────
def run_flask():
    logger.info(f"Dashboard + WS → http://{HTTP_HOST}:{HTTP_PORT}")
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, use_reloader=False, threaded=True)

# ── Start servers ────────────────────────────────────────────
def start_servers():
    threading.Thread(target=run_flask,       daemon=True).start()
    threading.Thread(target=broadcast_loop,  daemon=True).start()