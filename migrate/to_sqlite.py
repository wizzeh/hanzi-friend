"""Import the old TinyDB db.json into the SQLite database.

    python -m migrate.to_sqlite <username> [db.json]

Creates the user (prompts for a password), then imports their cards,
review logs, progress, definitions, and the shared lexicon into the
database at HANZI_DATA_DIR/hanzi.db.
"""

import getpass
import json
import sys
from datetime import datetime

from werkzeug.security import generate_password_hash

import db as store


def migrate(path: str, username: str, password: str):
    with open(path) as f:
        data = json.load(f)

    conn = store.connect()
    user_id = store.create_user(username, generate_password_hash(password))

    cards = 0
    logs = 0
    duplicates = 0

    with conn:
        for doc in data["_default"].values():
            type_ = doc["type_"]

            if type_ == "card":
                cur = conn.execute(
                    """INSERT OR IGNORE INTO cards
                       (user_id, word, quiz_type, reading, fsrs, due_ts)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        doc["word"],
                        doc["quiz_type"],
                        doc.get("pinyin") or "",
                        json.dumps(doc["card"]),
                        store.due_timestamp(doc["card"]),
                    ),
                )
                if cur.rowcount == 0:
                    duplicates += 1
                    continue
                cards += 1
                for log in doc.get("review_logs", []):
                    conn.execute(
                        """INSERT INTO review_logs (card_id, rating, reviewed_at, log)
                           VALUES (?, ?, ?, ?)""",
                        (
                            cur.lastrowid,
                            log["rating"],
                            datetime.fromisoformat(
                                log["review_datetime"]
                            ).timestamp(),
                            json.dumps(log),
                        ),
                    )
                    logs += 1
            elif type_ == "characters_seen":
                store.set_characters_seen(user_id, doc["number_seen"])
            elif type_ == "definition":
                store.set_user_definition(user_id, doc["hanzi"], doc["definition"])
            elif type_ == "pending_pair":
                store.queue_pending_pair(user_id, doc["word"], doc["pinyin"])

        for entry in data.get("lexicon", {}).values():
            store.put_lexicon(entry["word"], entry)

    print(
        "user {} (id {}): {} cards, {} review logs, {} lexicon entries".format(
            username, user_id, cards, logs, len(data.get("lexicon", {}))
        )
    )
    if duplicates:
        print(
            "{} duplicate cards skipped (same word/quiz/reading)".format(duplicates)
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    username = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "db.json"
    password = getpass.getpass("Password for {}: ".format(username))
    if password != getpass.getpass("Again: "):
        print("Passwords don't match.")
        sys.exit(1)
    migrate(path, username, password)
