"""
database.py
─────────────────────────────────────────────
SQLite persistence layer for MindCare.

Tables:
  - users              (login accounts)
  - conversations      (one row per chat session, so users can have
                         multiple named conversations, like ChatGPT)
  - messages           (chat messages, linked to a conversation)
  - risk_assessments   (risk-assessment submissions, per user)

Standard-library only (sqlite3) — no new dependency needed.
"""

import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mindcare.db")


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                mobile TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,             -- 'user' or 'assistant'
                content TEXT NOT NULL,
                sentiment TEXT,
                risk REAL,
                crisis INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
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
        # Persistent "remember me" login sessions, referenced by a token stored
        # in a browser cookie so users don't need to log in every visit.
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
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
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
        cur = conn.execute(
            """INSERT INTO users (username, email, mobile, password_hash, salt, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (username, email or None, mobile or None, password_hash, salt, datetime.now().isoformat()),
        )
        return cur.lastrowid


def get_user_by_username(username: str):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def get_user_by_identifier(identifier: str):
    """Look up a user by username, email, OR mobile number — whichever matches."""
    identifier = identifier.strip()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ? OR mobile = ?",
            (identifier, identifier, identifier),
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def is_email_taken(email: str) -> bool:
    with get_connection() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
        return row is not None


def is_mobile_taken(mobile: str) -> bool:
    with get_connection() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE mobile = ?", (mobile,)).fetchone()
        return row is not None


def get_or_create_google_user(email: str, display_name: str = None):
    """
    Used for 'Continue with Google' sign-in. If an account with this email
    already exists, return it. Otherwise auto-create one (no password needed
    since Google already verified identity) and return the new record.
    """
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row:
            return dict(row)

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

        cur = conn.execute(
            """INSERT INTO users (username, email, mobile, password_hash, salt, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (username, email, None, password_hash, salt, datetime.now().isoformat()),
        )
        new_id = cur.lastrowid
        return {
            "id": new_id, "username": username, "email": email, "mobile": None,
            "password_hash": password_hash, "salt": salt, "created_at": datetime.now().isoformat(),
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
        row = conn.execute("SELECT * FROM sessions WHERE token = ?", (token,)).fetchone()
        return dict(row) if row else None


def delete_session(token: str):
    with get_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# ─────────────────────────────────────────────
# CONVERSATIONS
# ─────────────────────────────────────────────
def create_conversation(user_id: int, title: str = "New Chat") -> int:
    now = datetime.now().isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, title, now, now),
        )
        return cur.lastrowid


def get_conversations(user_id: int):
    """Most recently updated first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


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
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return dict(row) if row else None


# ─────────────────────────────────────────────
# MESSAGES
# ─────────────────────────────────────────────
def save_message(conversation_id: int, role: str, content: str, sentiment: str = None,
                  risk: float = None, crisis: bool = False, timestamp: str = None) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO messages (conversation_id, role, content, sentiment, risk, crisis, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (conversation_id, role, content, sentiment, risk, int(crisis),
             timestamp or datetime.now().strftime("%I:%M %p")),
        )
        new_id = cur.lastrowid
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
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def clear_messages(conversation_id: int):
    with get_connection() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))


def search_conversations(user_id: int, query: str):
    """Search conversation titles AND message content for this user."""
    like = f"%{query}%"
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT DISTINCT c.* FROM conversations c
               LEFT JOIN messages m ON m.conversation_id = c.id
               WHERE c.user_id = ? AND (c.title LIKE ? OR m.content LIKE ?)
               ORDER BY c.updated_at DESC""",
            (user_id, like, like),
        ).fetchall()
        return [dict(r) for r in rows]


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
            "SELECT * FROM risk_assessments WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
