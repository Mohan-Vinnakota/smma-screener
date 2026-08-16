import sqlite3
import os
from datetime import datetime

DB_PATH = "smma_screener.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # signals table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
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

    # screened stocks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS screened_stocks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT NOT NULL,
            ltp         REAL,
            bid_qty     INTEGER,
            ask_qty     INTEGER,
            screened_at TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("Database ready")

def save_signal(symbol, signal, entry_ltp, exit_ltp=None,
                pnl=None, profitable=None, predicted=None,
                confidence=None, reason=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO signals
        (symbol, signal, entry_ltp, exit_ltp, pnl,
         profitable, predicted, confidence, reason, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        symbol, signal,
        float(entry_ltp) if entry_ltp else None,
        float(exit_ltp) if exit_ltp else None,
        float(pnl) if pnl else None,
        profitable,
        predicted,
        float(confidence) if confidence else None,   # ← fix here
        reason,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()
def update_signal_exit(symbol, signal, exit_ltp, pnl, profitable):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE signals
        SET exit_ltp = ?, pnl = ?, profitable = ?
        WHERE symbol = ? AND signal = ?
        AND exit_ltp IS NULL
        ORDER BY id DESC
        LIMIT 1
    """, (exit_ltp, pnl, profitable, symbol, signal))
    conn.commit()
    conn.close()
def update_signal_exit(symbol, signal, exit_ltp, pnl, profitable):
    conn = get_connection()
    cursor = conn.cursor()

    # First find the latest open signal id
    cursor.execute("""
        SELECT id FROM signals
        WHERE symbol = ? AND signal = ? AND exit_ltp IS NULL
        ORDER BY id DESC
        LIMIT 1
    """, (symbol, signal))

    row = cursor.fetchone()
    if row:
        cursor.execute("""
            UPDATE signals
            SET exit_ltp = ?, pnl = ?, profitable = ?
            WHERE id = ?
        """, (exit_ltp, pnl, profitable, row["id"]))
        conn.commit()

    conn.close()
def save_screened_stock(symbol, ltp, bid_qty, ask_qty):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO screened_stocks
        (symbol, ltp, bid_qty, ask_qty, screened_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        symbol, ltp, bid_qty, ask_qty,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def get_all_signals():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM signals
        ORDER BY id DESC
        LIMIT 100
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

if __name__ == "__main__":
    init_db()

    # Test save signal
    save_signal(
        symbol="SUZLON-EQ",
        signal="BUY",
        entry_ltp=42.5,
        predicted="ACCEPT",
        confidence=0.75,
        reason="XGBoost: ACCEPT (75% confidence)"
    )
    print("Signal saved")

    # Test update exit
    update_signal_exit("SUZLON-EQ", "BUY", 45.0, 2.5, 1)
    print("Exit updated")

    # Test fetch
    signals = get_all_signals()
    for s in signals:
        print(s)