"""
auth_db.py — User authentication & analysis history SQLite module
"""
import sqlite3
import hashlib
import os
import secrets
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "ab_platform.db")


# ── Connection ────────────────────────────────────────────────────
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ── Initialize tables ─────────────────────────────────────────────
def init_db() -> None:
    conn = _get_conn()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL UNIQUE,
            salt        TEXT    NOT NULL,
            pwd_hash    TEXT    NOT NULL,
            created_at  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS analysis_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            created_at      TEXT    NOT NULL,
            -- Experiment config
            data_source     TEXT,
            group_col       TEXT,
            control_label   TEXT,
            treatment_label TEXT,
            metric_col      TEXT,
            metric_type     TEXT,
            method          TEXT,
            alpha           REAL,
            mde             REAL,
            loss_threshold  REAL,
            n_control       INTEGER,
            n_treatment     INTEGER,
            -- Frequentist results
            freq_mean_a      REAL,
            freq_mean_b      REAL,
            freq_p_value     REAL,
            freq_effect_size REAL,
            freq_delta       REAL,
            freq_ci_low      REAL,
            freq_ci_high     REAL,
            freq_significant INTEGER,
            freq_decision    TEXT,
            -- Bayesian results
            bayes_mean_a         REAL,
            bayes_mean_b         REAL,
            bayes_prob_b_better  REAL,
            bayes_prob_practical REAL,
            bayes_loss_a         REAL,
            bayes_loss_b         REAL,
            bayes_decision       TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()

    # Migrate older databases: add missing columns if they don't exist
    existing = {row[1] for row in conn.execute("PRAGMA table_info(analysis_history)")}
    for col_name, col_type in [
        ("group_col",       "TEXT"),
        ("control_label",   "TEXT"),
        ("treatment_label", "TEXT"),
    ]:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE analysis_history ADD COLUMN {col_name} {col_type}")
    conn.commit()
    conn.close()


# ── User management ───────────────────────────────────────────────
def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode()).hexdigest()


def register_user(username: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    if not username or not password:
        return False, "Username and password are required"
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    conn = _get_conn()
    try:
        salt = secrets.token_hex(16)
        pwd_hash = _hash_password(password, salt)
        conn.execute(
            "INSERT INTO users (username, salt, pwd_hash, created_at) VALUES (?, ?, ?, ?)",
            (username.strip(), salt, pwd_hash, datetime.now().isoformat()),
        )
        conn.commit()
        return True, "Registration successful"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Verify credentials. Returns user dict on success, None on failure."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id, username, salt, pwd_hash FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
        if row and _hash_password(password, row["salt"]) == row["pwd_hash"]:
            return {"id": row["id"], "username": row["username"]}
        return None
    finally:
        conn.close()


def change_password(user_id: int, old_password: str, new_password: str) -> tuple[bool, str]:
    """Change password. Returns (success, message)."""
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters"
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT salt, pwd_hash FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not row:
            return False, "User not found"
        if _hash_password(old_password, row["salt"]) != row["pwd_hash"]:
            return False, "Current password is incorrect"
        new_salt = secrets.token_hex(16)
        new_hash = _hash_password(new_password, new_salt)
        conn.execute(
            "UPDATE users SET salt = ?, pwd_hash = ? WHERE id = ?",
            (new_salt, new_hash, user_id),
        )
        conn.commit()
        return True, "Password changed successfully"
    finally:
        conn.close()


# ── Analysis history ──────────────────────────────────────────────
def save_analysis(user_id: int, record: dict) -> int:
    """Save one analysis run. Returns the new record id."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO analysis_history (
                user_id, created_at,
                data_source, group_col, control_label, treatment_label,
                metric_col, metric_type, method,
                alpha, mde, loss_threshold, n_control, n_treatment,
                freq_mean_a, freq_mean_b, freq_p_value, freq_effect_size,
                freq_delta, freq_ci_low, freq_ci_high, freq_significant, freq_decision,
                bayes_mean_a, bayes_mean_b, bayes_prob_b_better, bayes_prob_practical,
                bayes_loss_a, bayes_loss_b, bayes_decision
            ) VALUES (
                :user_id, :created_at,
                :data_source, :group_col, :control_label, :treatment_label,
                :metric_col, :metric_type, :method,
                :alpha, :mde, :loss_threshold, :n_control, :n_treatment,
                :freq_mean_a, :freq_mean_b, :freq_p_value, :freq_effect_size,
                :freq_delta, :freq_ci_low, :freq_ci_high, :freq_significant, :freq_decision,
                :bayes_mean_a, :bayes_mean_b, :bayes_prob_b_better, :bayes_prob_practical,
                :bayes_loss_a, :bayes_loss_b, :bayes_decision
            )""",
            {**record, "user_id": user_id},
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_user_history(user_id: int, limit: int = 50) -> list[dict]:
    """Return the most recent `limit` analysis records for a user."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT * FROM analysis_history
               WHERE user_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_analysis(record_id: int, user_id: int) -> bool:
    """Delete a record (must belong to the given user)."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            "DELETE FROM analysis_history WHERE id = ? AND user_id = ?",
            (record_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
