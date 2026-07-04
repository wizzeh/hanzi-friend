"""SQLite storage for Hanzi Friend.

One connection per thread (sqlite3 objects can't cross threads), WAL mode
so readers never block the writer. The lexicon and audio cache are shared;
everything else is scoped to a user.

Set HANZI_DATA_DIR to point the database (and TTS cache) somewhere
writable in production; it defaults to the working directory for dev.
"""

import json
import os
import sqlite3
import threading
import time

DATA_DIR = os.environ.get("HANZI_DATA_DIR", ".")
DB_PATH = os.path.join(DATA_DIR, "hanzi.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    characters_seen INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    word TEXT NOT NULL,
    quiz_type TEXT NOT NULL,
    reading TEXT NOT NULL DEFAULT '',
    fsrs TEXT NOT NULL,
    due_ts REAL NOT NULL,
    UNIQUE (user_id, word, quiz_type, reading)
);
CREATE INDEX IF NOT EXISTS cards_due ON cards (user_id, due_ts);

CREATE TABLE IF NOT EXISTS review_logs (
    id INTEGER PRIMARY KEY,
    card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL,
    reviewed_at REAL NOT NULL,
    log TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS review_logs_card ON review_logs (card_id);

CREATE TABLE IF NOT EXISTS pending_pairs (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    word TEXT NOT NULL,
    reading TEXT NOT NULL,
    UNIQUE (user_id, word, reading)
);

CREATE TABLE IF NOT EXISTS prelearn (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    word TEXT NOT NULL,
    UNIQUE (user_id, word)
);

CREATE TABLE IF NOT EXISTS definitions (
    user_id INTEGER NOT NULL REFERENCES users(id),
    word TEXT NOT NULL,
    definition TEXT NOT NULL,
    PRIMARY KEY (user_id, word)
);

CREATE TABLE IF NOT EXISTS lexicon (
    word TEXT PRIMARY KEY,
    senses TEXT NOT NULL,
    components TEXT NOT NULL,
    version INTEGER NOT NULL
);
"""

_local = threading.local()


MIGRATIONS = [
    "ALTER TABLE users ADD COLUMN openai_api_key TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE users ADD COLUMN speech_key TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE users ADD COLUMN speech_region TEXT NOT NULL DEFAULT 'eastus'",
]


def connect() -> sqlite3.Connection:
    if getattr(_local, "conn", None) is None:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA)
        for migration in MIGRATIONS:
            try:
                conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # Column already exists.
        _local.conn = conn
    return _local.conn


# --- users -----------------------------------------------------------------


def create_user(username: str, password_hash: str) -> int:
    conn = connect()
    with conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, time.time()),
        )
    return cur.lastrowid


def get_user(username: str):
    return connect().execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()


def user_keys(user_id: int):
    """The user's own API keys. AI features run on these -- there is no
    fallback to the server's keys."""
    return connect().execute(
        "SELECT openai_api_key, speech_key, speech_region FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()


def set_user_keys(user_id: int, openai_api_key: str, speech_key: str, speech_region: str):
    conn = connect()
    with conn:
        conn.execute(
            """UPDATE users SET openai_api_key = ?, speech_key = ?, speech_region = ?
               WHERE id = ?""",
            (openai_api_key, speech_key, speech_region or "eastus", user_id),
        )


def characters_seen(user_id: int) -> int:
    row = connect().execute(
        "SELECT characters_seen FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    return row["characters_seen"]


def set_characters_seen(user_id: int, number: int):
    conn = connect()
    with conn:
        conn.execute(
            "UPDATE users SET characters_seen = ? WHERE id = ?", (number, user_id)
        )


def increment_characters_seen(user_id: int):
    conn = connect()
    with conn:
        conn.execute(
            "UPDATE users SET characters_seen = characters_seen + 1 WHERE id = ?",
            (user_id,),
        )


# --- cards -----------------------------------------------------------------


def insert_card(user_id: int, word: str, quiz_type: str, reading, fsrs_dict: dict):
    conn = connect()
    with conn:
        conn.execute(
            """INSERT OR IGNORE INTO cards (user_id, word, quiz_type, reading, fsrs, due_ts)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                word,
                quiz_type,
                reading or "",
                json.dumps(fsrs_dict),
                due_timestamp(fsrs_dict),
            ),
        )


def due_timestamp(fsrs_dict: dict) -> float:
    from datetime import datetime

    return datetime.fromisoformat(fsrs_dict["due"]).timestamp()


def due_cards(user_id: int):
    return connect().execute(
        "SELECT * FROM cards WHERE user_id = ? AND due_ts <= ?",
        (user_id, time.time()),
    ).fetchall()


def get_card(card_id: int):
    return connect().execute(
        "SELECT * FROM cards WHERE id = ?", (card_id,)
    ).fetchone()


def has_cards_for_word(user_id: int, word: str) -> bool:
    return (
        connect().execute(
            "SELECT 1 FROM cards WHERE user_id = ? AND word = ? LIMIT 1",
            (user_id, word),
        ).fetchone()
        is not None
    )


def update_card(card_id: int, fsrs_dict: dict, review_log: dict, rating: int):
    conn = connect()
    with conn:
        conn.execute(
            "UPDATE cards SET fsrs = ?, due_ts = ? WHERE id = ?",
            (json.dumps(fsrs_dict), due_timestamp(fsrs_dict), card_id),
        )
        conn.execute(
            "INSERT INTO review_logs (card_id, rating, reviewed_at, log) VALUES (?, ?, ?, ?)",
            (card_id, rating, time.time(), json.dumps(review_log)),
        )


def user_words(user_id: int):
    return [
        row["word"]
        for row in connect().execute(
            "SELECT DISTINCT word FROM cards WHERE user_id = ?", (user_id,)
        )
    ]


def known_words(user_id: int, quiz_type: str):
    return [
        row["word"]
        for row in connect().execute(
            "SELECT DISTINCT word FROM cards WHERE user_id = ? AND quiz_type = ? ORDER BY word",
            (user_id, quiz_type),
        )
    ]


def hardest_words(user_id: int, quiz_type: str):
    rows = connect().execute(
        "SELECT word, fsrs FROM cards WHERE user_id = ? AND quiz_type = ?",
        (user_id, quiz_type),
    ).fetchall()
    rows.sort(key=lambda row: json.loads(row["fsrs"])["difficulty"], reverse=True)
    return [row["word"] for row in rows]


# --- pending secondary readings ---------------------------------------------


def queue_pending_pair(user_id: int, word: str, reading: str):
    conn = connect()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO pending_pairs (user_id, word, reading) VALUES (?, ?, ?)",
            (user_id, word, reading),
        )


def next_pending_pair(user_id: int):
    return connect().execute(
        "SELECT * FROM pending_pairs WHERE user_id = ? ORDER BY id LIMIT 1",
        (user_id,),
    ).fetchone()


def remove_pending_pair(user_id: int, word: str, reading: str):
    conn = connect()
    with conn:
        conn.execute(
            "DELETE FROM pending_pairs WHERE user_id = ? AND word = ? AND reading = ?",
            (user_id, word, reading),
        )


# --- pre-learn queue ---------------------------------------------------------


def queue_prelearn(user_id: int, word: str):
    conn = connect()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO prelearn (user_id, word) VALUES (?, ?)",
            (user_id, word),
        )


def next_prelearn(user_id: int):
    row = connect().execute(
        "SELECT word FROM prelearn WHERE user_id = ? ORDER BY id LIMIT 1",
        (user_id,),
    ).fetchone()
    return row["word"] if row else None


def prelearn_queue(user_id: int):
    return [
        row["word"]
        for row in connect().execute(
            "SELECT word FROM prelearn WHERE user_id = ? ORDER BY id", (user_id,)
        )
    ]


def remove_prelearn(user_id: int, word: str) -> bool:
    conn = connect()
    with conn:
        cur = conn.execute(
            "DELETE FROM prelearn WHERE user_id = ? AND word = ?", (user_id, word)
        )
    return cur.rowcount > 0


# --- user definitions --------------------------------------------------------


def user_definition(user_id: int, word: str) -> str:
    row = connect().execute(
        "SELECT definition FROM definitions WHERE user_id = ? AND word = ?",
        (user_id, word),
    ).fetchone()
    return row["definition"] if row else ""


def set_user_definition(user_id: int, word: str, definition: str):
    conn = connect()
    with conn:
        conn.execute(
            """INSERT INTO definitions (user_id, word, definition) VALUES (?, ?, ?)
               ON CONFLICT (user_id, word) DO UPDATE SET definition = excluded.definition""",
            (user_id, word, definition),
        )


# --- shared lexicon ----------------------------------------------------------


def get_lexicon(word: str):
    row = connect().execute(
        "SELECT * FROM lexicon WHERE word = ?", (word,)
    ).fetchone()
    if row is None:
        return None
    return {
        "word": row["word"],
        "senses": json.loads(row["senses"]),
        "components": row["components"],
        "version": row["version"],
    }


def put_lexicon(word: str, entry: dict):
    conn = connect()
    with conn:
        conn.execute(
            """INSERT INTO lexicon (word, senses, components, version) VALUES (?, ?, ?, ?)
               ON CONFLICT (word) DO UPDATE SET
                   senses = excluded.senses,
                   components = excluded.components,
                   version = excluded.version""",
            (word, json.dumps(entry["senses"]), entry["components"], entry["version"]),
        )


def all_lexicon_words():
    return [row["word"] for row in connect().execute("SELECT word FROM lexicon")]


def all_card_words():
    return [
        row["word"]
        for row in connect().execute("SELECT DISTINCT word FROM cards ORDER BY word")
    ]
