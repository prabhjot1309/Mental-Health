"""
database.py
─────────────────────────────────────────────
Persistence layer for MindCare, with TWO possible backends:

  1. Postgres (production) — used automatically if a DATABASE_URL secret
     is configured. This is REQUIRED for Streamlit Community Cloud,
     because that platform does not give apps persistent disk storage:
     the local filesystem (and therefore a SQLite .db file) gets wiped
     on every restart/redeploy/sleep cycle, which is why accounts were
     disappearing and users had to keep re-signing up.

  2. SQLite (local dev fallback) — used automatically if no DATABASE_URL
     is set, so you can still run this locally with zero setup.

All the functions below (create_user, save_message, etc.) work the same
regardless of which backend is active — the rest of the app never needs
to know which one is in use.

Tables:
  - users              (login accounts)
  - conversations      (one row per chat session, multiple per user)
  - messages           (chat messages, linked to a conversation)
  - risk_assessments   (risk-assessment submissions, per user)
  - sessions           ("remember me" persistent login tokens)
"""

import os
from datetime import datetime
from contextlib import contextmanager

try:
    import streamlit as st
except ImportError:
    st = None


def _get_database_url():
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    if st is not None:
        try:
            return st.secrets["DATABASE_URL"]
        except (KeyError, FileNotFoundError):
            return None
    return None


DATABASE_URL = _get_database_url()
BACKEND = "postgres" if DATABASE_URL else "sqlite"

if BACKEND == "postgres":
    import psycopg2
    import psycopg2.extras
else:
    import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mindcare.db")

# SQLite (3.35+) and Postgres both support "INSERT ... RETURNING id",
# which we rely on everywhere instead of driver-specific lastrowid logic.
PK_DEFINITION = "SERIAL PRIMARY KEY" if BACKEND == "postgres" else "INTEGER PRIMARY KEY AUTOINCREMENT"


def _adapt_sql(sql: str) -> str:
    """Translate '?' placeholders to Postgres-style '%s'."""
    return sql.replace("?", "%s") if BACKEND == "postgres" else sql


class _CursorWrapper:
    def __init__(self, cursor):
        self._cursor = cursor

    def fetchone(self):
        row = self._cursor.fetchone()
        return dict(row) if row is not None else None

    def fetchall(self):
        return [dict(r) for r in self._cursor.fetchall()]


class _ConnWrapper:
    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=()):
        cur = self._conn.cursor()
        cur.execute(_adapt_sql(sql), params)
        return _CursorWrapper(cur)


@contextmanager
def get_connection():
    if BACKEND == "postgres":
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
    wrapped = _ConnWrapper(conn)
    try:
        yield wrapped
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS users (
                id {PK_DEFINITION},
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                mobile TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS conversations (
                id {PK_DEFINITION},
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS messages (
                id {PK_DEFINITION},
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sentiment TEXT,
                risk REAL,
                crisis INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS risk_assessments (
                id {PK_DEFINITION},
                user_id INTEGER NOT NULL,
                sadness INTEGER, anxiety INTEGER, sleep INTEGER,
                energy INTEGER, selfharm INTEGER,
                total_score INTEGER NOT NULL,
                level TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)

        # --- Migration: add email/mobile columns to a pre-existing users table ---
        if BACKEND == "postgres":
            existing_cols = {r["column_name"] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'users'"
            ).fetchall()}
        else:
            existing_cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "email" not in existing_cols:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
        if "mobile" not in existing_cols:
            conn.execute("ALTER TABLE users ADD COLUMN mobile TEXT")


# ─────────────────────────────────────────────
# USERS
# ─────────────────────────────────────────────
def create_user(username: str, password_hash: str, salt: str,
                 email: str = None, mobile: str = None) -> int:
    with get_connection() as conn:
        row = conn.execute(
            """INSERT INTO users (username, email, mobile, password_hash, salt, created_at)
               VALUES (?, ?, ?, ?, ?, ?) RETURNING id""",
            (username, email or None, mobile or None, password_hash, salt, datetime.now().isoformat()),
        ).fetchone()
        return row["id"]


def get_user_by_username(username: str):
    with get_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def get_user_by_identifier(identifier: str):
    """Look up a user by username, email, OR mobile number — whichever matches."""
    identifier = identifier.strip()
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ? OR mobile = ?",
            (identifier, identifier, identifier),
        ).fetchone()


def get_user_by_id(user_id: int):
    with get_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def is_email_taken(email: str) -> bool:
    with get_connection() as conn:
        return conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone() is not None


def is_mobile_taken(mobile: str) -> bool:
    with get_connection() as conn:
        return conn.execute("SELECT 1 FROM users WHERE mobile = ?", (mobile,)).fetchone() is not None


def get_or_create_google_user(email: str, display_name: str = None):
    """
    Used for 'Continue with Google' sign-in. If an account with this email
    already exists, return it. Otherwise auto-create one (no password needed
    since Google already verified identity) and return the new record.
    """
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row:
            return row

        base_username = (display_name or email.split("@")[0]).strip().replace(" ", "_")
        username = base_username
        suffix = 1
        while conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            suffix += 1
            username = f"{base_username}{suffix}"

        # Google-authenticated accounts don't need a local password; store an
        # unusable random hash so the column constraints are still satisfied.
        import secrets as _secrets
        salt = _secrets.token_hex(16)
        password_hash = _secrets.token_hex(32)
        created_at = datetime.now().isoformat()

        new_row = conn.execute(
            """INSERT INTO users (username, email, mobile, password_hash, salt, created_at)
               VALUES (?, ?, ?, ?, ?, ?) RETURNING id""",
            (username, email, None, password_hash, salt, created_at),
        ).fetchone()
        return {
            "id": new_row["id"], "username": username, "email": email, "mobile": None,
            "password_hash": password_hash, "salt": salt, "created_at": created_at,
        }


# ─────────────────────────────────────────────
# SESSIONS ("remember me" persistent login)
# ─────────────────────────────────────────────
def create_session(token: str, user_id: int, expires_at: str):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, datetime.now().isoformat(), expires_at),
        )


def get_session(token: str):
    with get_connection() as conn:
        return conn.execute("SELECT * FROM sessions WHERE token = ?", (token,)).fetchone()


def delete_session(token: str):
    with get_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# ─────────────────────────────────────────────
# CONVERSATIONS
# ─────────────────────────────────────────────
def create_conversation(user_id: int, title: str = "New Chat") -> int:
    now = datetime.now().isoformat()
    with get_connection() as conn:
        row = conn.execute(
            "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?) RETURNING id",
            (user_id, title, now, now),
        ).fetchone()
        return row["id"]


def get_conversations(user_id: int):
    """Most recently updated first."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()


def rename_conversation(conversation_id: int, title: str):
    with get_connection() as conn:
        conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))


def touch_conversation(conversation_id: int):
    """Bump updated_at so recently-active chats sort to the top."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (datetime.now().isoformat(), conversation_id),
        )


def delete_conversation(conversation_id: int):
    with get_connection() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def get_conversation(conversation_id: int):
    with get_connection() as conn:
        return conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()


# ─────────────────────────────────────────────
# MESSAGES
# ─────────────────────────────────────────────
def save_message(conversation_id: int, role: str, content: str, sentiment: str = None,
                  risk: float = None, crisis: bool = False, timestamp: str = None) -> int:
    with get_connection() as conn:
        row = conn.execute(
            """INSERT INTO messages (conversation_id, role, content, sentiment, risk, crisis, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id""",
            (conversation_id, role, content, sentiment, risk, int(crisis),
             timestamp or datetime.now().strftime("%I:%M %p")),
        ).fetchone()
        new_id = row["id"]
    touch_conversation(conversation_id)
    return new_id


def truncate_messages_from(conversation_id: int, message_id: int):
    """Delete this message and everything after it (used when editing a message)."""
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM messages WHERE conversation_id = ? AND id >= ?",
            (conversation_id, message_id),
        )


def get_messages(conversation_id: int, limit: int = 500):
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()


def clear_messages(conversation_id: int):
    with get_connection() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))


def search_conversations(user_id: int, query: str):
    """Search conversation titles AND message content for this user."""
    like = f"%{query}%"
    with get_connection() as conn:
        return conn.execute(
            """SELECT DISTINCT c.* FROM conversations c
               LEFT JOIN messages m ON m.conversation_id = c.id
               WHERE c.user_id = ? AND (c.title LIKE ? OR m.content LIKE ?)
               ORDER BY c.updated_at DESC""",
            (user_id, like, like),
        ).fetchall()


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
        return conn.execute(
            "SELECT * FROM risk_assessments WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
