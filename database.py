"""
database.py
─────────────────────────────────────────────
SQLite persistence layer for MindCare.

Adds three tables:
  - users             (login accounts)
  - messages          (chat history, per user)
  - risk_assessments  (risk-assessment submissions, per user)

Uses only the Python standard library (sqlite3), so no new
dependency needs to be added to requirements.txt.
"""

import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mindcare.db")


@contextmanager
def get_connection():
    """Yield a SQLite connection with foreign keys enabled, and always close it."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't already exist. Safe to call every app start."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,             -- 'user' or 'assistant'
                content TEXT NOT NULL,
                sentiment TEXT,
                risk REAL,
                crisis INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                sadness INTEGER, anxiety INTEGER, sleep INTEGER,
                energy INTEGER, selfharm INTEGER,
                total_score INTEGER NOT NULL,
                level TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)


# ─────────────────────────────────────────────
# USERS
# ─────────────────────────────────────────────
def create_user(username: str, password_hash: str, salt: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, salt, datetime.now().isoformat()),
        )
        return cur.lastrowid


def get_user_by_username(username: str):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


# ─────────────────────────────────────────────
# MESSAGES (chat history)
# ─────────────────────────────────────────────
def save_message(user_id: int, role: str, content: str, sentiment: str = None,
                  risk: float = None, crisis: bool = False, timestamp: str = None):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO messages (user_id, role, content, sentiment, risk, crisis, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, role, content, sentiment, risk, int(crisis),
             timestamp or datetime.now().strftime("%I:%M %p")),
        )


def get_messages(user_id: int, limit: int = 100):
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM messages WHERE user_id = ?
               ORDER BY id ASC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def clear_messages(user_id: int):
    with get_connection() as conn:
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))


# ─────────────────────────────────────────────
# RISK ASSESSMENTS
# ─────────────────────────────────────────────
def save_risk_assessment(user_id: int, sadness: int, anxiety: int, sleep: int,
                          energy: int, selfharm: int, total_score: int, level: str):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO risk_assessments
               (user_id, sadness, anxiety, sleep, energy, selfharm, total_score, level, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, sadness, anxiety, sleep, energy, selfharm, total_score, level,
             datetime.now().strftime("%Y-%m-%d %I:%M %p")),
        )


def get_risk_history(user_id: int, limit: int = 20):
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM risk_assessments WHERE user_id = ?
               ORDER BY id DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
