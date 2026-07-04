"""
auth.py
─────────────────────────────────────────────
Minimal username/password auth for MindCare.

Passwords are never stored in plain text: each user gets a random
salt, and we store PBKDF2-HMAC-SHA256(password, salt) instead.
Uses only the Python standard library (hashlib, secrets).
"""

import hashlib
import secrets

import database as db


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()


def signup(username: str, password: str):
    """
    Create a new account.
    Returns (success: bool, message: str).
    """
    username = username.strip()
    if not username or not password:
        return False, "Username and password can't be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if db.get_user_by_username(username):
        return False, "That username is already taken."

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    db.create_user(username, password_hash, salt)
    return True, "Account created! You can now log in."


def login(username: str, password: str):
    """
    Verify credentials.
    Returns (success: bool, user_dict_or_None, message: str).
    """
    user = db.get_user_by_username(username.strip())
    if not user:
        return False, None, "No account found with that username."

    expected_hash = _hash_password(password, user["salt"])
    if secrets.compare_digest(expected_hash, user["password_hash"]):
        return True, user, "Welcome back!"
    return False, None, "Incorrect password."
