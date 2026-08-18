import sqlite3
import os
from datetime import datetime
from config import DB_PATH
from logger import logger

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # signals table — market column added (NSE / MCX)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            market      TEXT DEFAULT 'NSE',
            symbol      TEXT NOT NULL,
            signal      TEXT NOT NULL,
            entry_ltp   REAL,
            exit_ltp    REAL,
            pnl         REAL,
            profitable  INTEGER,
            predicted   TEXT,
            confidence  REAL,
            reason      TEXT,
            timestamp   TEXT
        )
    """)

    # screened_stocks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS screened_stocks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            market      TEXT DEFAULT 'NSE',
            symbol      TEXT NOT NULL,
            ltp         REAL,
            bid_qty     INTEGER,
            ask_qty     INTEGER,
            screened_at TEXT
        )
    """)

    # Migrate existing DB — add market column if missing (backward compat)
    _add_column_if_missing(cursor, "signals",          "market", "TEXT DEFAULT 'NSE'")
    _add_column_if_missing(cursor, "screened_stocks",  "market", "TEXT DEFAULT 'NSE'")

    conn.commit()
    conn.close()
    logger.info("Database ready")

def _add_column_if_missing(cursor, table, column, col_def):
    cursor.execute(f"PRAGMA table_info({table})")
    cols = [row["name"] for row in cursor.fetchall()]
    if column not in cols:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")
        logger.info(f"Migrated DB: added '{column}' to {table}")

def save_signal(symbol, signal, entry_ltp, market="NSE",
                exit_ltp=None, pnl=None, profitable=None,
                predicted=None, confidence=None, reason=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO signals
        (market, symbol, signal, entry_ltp, exit_ltp, pnl,
         profitable, predicted, confidence, reason, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        market, symbol, signal,
        float(entry_ltp) if entry_ltp is not None else None,
        float(exit_ltp)  if exit_ltp  is not None else None,
        float(pnl)       if pnl       is not None else None,
        profitable,
        predicted,
        float(confidence) if confidence is not None else None,
        reason,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def update_signal_exit(symbol, signal, exit_ltp, pnl, profitable, market="NSE"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id FROM signals
        WHERE symbol = ? AND signal = ? AND market = ? AND exit_ltp IS NULL
        ORDER BY id DESC
        LIMIT 1
    """, (symbol, signal, market))

    row = cursor.fetchone()
    if row:
        cursor.execute("""
            UPDATE signals
            SET exit_ltp = ?, pnl = ?, profitable = ?
            WHERE id = ?
        """, (exit_ltp, pnl, profitable, row["id"]))
        conn.commit()
    conn.close()

def save_screened_stock(symbol, ltp, bid_qty, ask_qty, market="NSE"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO screened_stocks
        (market, symbol, ltp, bid_qty, ask_qty, screened_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        market, symbol, ltp, bid_qty, ask_qty,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def get_all_signals(market=None):
    conn = get_connection()
    cursor = conn.cursor()
    if market:
        cursor.execute("""
            SELECT * FROM signals
            WHERE market = ?
            ORDER BY id DESC
            LIMIT 200
        """, (market,))
    else:
        cursor.execute("""
            SELECT * FROM signals
            ORDER BY id DESC
            LIMIT 200
        """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

if __name__ == "__main__":
    init_db()
    save_signal("CRUDEOIL-I", "BUY", 6500.0, market="MCX",
                predicted="ACCEPT", confidence=0.72,
                reason="Rule-based: ACCEPT (score=0.72)")
    print("MCX signal saved")
    sigs = get_all_signals()
    for s in sigs:
        print(s)
