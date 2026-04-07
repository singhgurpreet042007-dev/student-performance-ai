import sqlite3
import hashlib

DB_NAME = "users.db"


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            study_hours REAL,
            attendance REAL,
            predicted_score REAL,
            risk TEXT
        )
    """)

    conn.commit()
    conn.close()


def create_user(username: str, password: str):
    username = username.strip()

    if not username or not password:
        return False, "Username and password are required."

    if len(username) < 3:
        return False, "Username must be at least 3 characters."

    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, _hash_password(password)),
        )
        conn.commit()
        conn.close()
        return True, "User created successfully!"
    except sqlite3.IntegrityError:
        return False, "Username already exists!"


def verify_user(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (username.strip(),))
    row = cur.fetchone()
    conn.close()

    if not row:
        return False

    return row[0] == _hash_password(password)


def save_prediction(username, study_hours, attendance, score, risk):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO predictions (username, study_hours, attendance, predicted_score, risk)
        VALUES (?, ?, ?, ?, ?)
    """, (username, study_hours, attendance, score, risk))

    conn.commit()
    conn.close()


def get_user_history(username):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        SELECT study_hours, attendance, predicted_score, risk
        FROM predictions
        WHERE username = ?
        ORDER BY id DESC
        LIMIT 10
    """, (username,))

    data = cur.fetchall()
    conn.close()
    return data