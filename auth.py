"""
auth.py
─────────────────────────────────────────────
Username/email/mobile + password auth for MindCare, plus
long-lived "remember me" session tokens.

Passwords are salted and hashed with PBKDF2-HMAC-SHA256 — never stored
in plain text. Uses only the Python standard library.
"""

import hashlib
import re
import secrets
from datetime import datetime, timedelta

import database as db

SESSION_LIFETIME_DAYS = 30


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()


def _is_valid_email(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value))


def _is_valid_mobile(value: str) -> bool:
    digits = re.sub(r"[\s\-()+]", "", value)
    return digits.isdigit() and 7 <= len(digits) <= 15


def signup(username: str, password: str, email: str = "", mobile: str = ""):
    """
    Create a new account. Email and mobile are both optional, but at
    least one is required so the account can be recovered/logged into
    without the exact username.
    Returns (success: bool, message: str).
    """
    username = username.strip()
    email = email.strip()
    mobile = mobile.strip()

    if not username or not password:
        return False, "Username and password can't be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if not email and not mobile:
        return False, "Please provide an email address or a mobile number."
    if email and not _is_valid_email(email):
        return False, "That doesn't look like a valid email address."
    if mobile and not _is_valid_mobile(mobile):
        return False, "That doesn't look like a valid mobile number."
    if db.get_user_by_username(username):
        return False, "That username is already taken."
    if email and db.is_email_taken(email):
        return False, "An account with that email already exists."
    if mobile and db.is_mobile_taken(mobile):
        return False, "An account with that mobile number already exists."

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    db.create_user(username, password_hash, salt, email=email or None, mobile=mobile or None)
    return True, "Account created! You can now log in."


def login(identifier: str, password: str):
    """
    Verify credentials. `identifier` can be a username, email, or mobile number.
    Returns (success: bool, user_dict_or_None, message: str).
    """
    user = db.get_user_by_identifier(identifier)
    if not user:
        return False, None, "No account found with that username, email, or mobile number."

    expected_hash = _hash_password(password, user["salt"])
    if secrets.compare_digest(expected_hash, user["password_hash"]):
        return True, user, "Welcome back!"
    return False, None, "Incorrect password."


# ─────────────────────────────────────────────
# GOOGLE SIGN-IN
# ─────────────────────────────────────────────
def login_with_google(email: str, display_name: str = None):
    """
    Find-or-create an account for a Google-authenticated email and return it.
    No password check needed — Google already verified the person's identity.
    Returns the user dict.
    """
    return db.get_or_create_google_user(email.strip().lower(), display_name)


# ─────────────────────────────────────────────
# "REMEMBER ME" SESSION TOKENS
# ─────────────────────────────────────────────
def create_remember_me_token(user_id: int) -> str:
    """Create a long-lived session token so the user stays logged in across visits."""
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now() + timedelta(days=SESSION_LIFETIME_DAYS)).isoformat()
    db.create_session(token, user_id, expires_at)
    return token


def resolve_session_token(token: str):
    """
    Given a token pulled from the browser cookie, return the logged-in
    user dict if the token is valid and not expired, else None.
    """
    if not token:
        return None
    session = db.get_session(token)
    if not session:
        return None
    if datetime.fromisoformat(session["expires_at"]) < datetime.now():
        db.delete_session(token)
        return None
    return db.get_user_by_id(session["user_id"])


def revoke_session_token(token: str):
    if token:
        db.delete_session(token)
