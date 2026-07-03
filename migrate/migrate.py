import json
from fsrs import Card, State
from datetime import datetime

with open("db.json", "r") as f:
    db = json.load(f)

    id = 0
    for key, value in db["_default"].items():
        if value["type_"] == "card":
            card = Card(
                card_id = id,
                state = State.Review,
                step = None,
                stability = value["card"]["stability"],
                difficulty = value["card"]["difficulty"],
                due = datetime.fromisoformat(value["card"]["due"]),
                last_review = datetime.fromisoformat(value["card"]["last_review"]),
            )
            db["_default"][key]["card"] = card.to_dict()
            id += 1

    with open("db-migrated.json", "w") as f:
        json.dump(db, f)

