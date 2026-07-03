"""Migrate existing cards to character-pronunciation pairs.

Tags every non-component card with its word's primary reading, and for
each other learnable reading clones the primary card's FSRS state into a
new card -- so secondary readings come due on the primary's schedule and
reschedule honestly once they're actually reviewed.

Words the lexicon marks recognition-only (pure components, archaic
non-words) are pruned down to just their meaning card instead.

Run the lexicon backfill first (python enrich.py), then:

    python -m migrate.pairs           # dry run, prints what would happen
    python -m migrate.pairs --apply   # do it

Idempotent: cards that already have a pinyin field are left alone.
"""

import sys

from tinydb import Query
import tinydb.operations as dbops

from db import db
import hanzi


def migrate(apply: bool):
    Cards = Query()
    docs = db.search(Cards.type_ == "card")

    tagged = 0
    cloned = 0
    pruned = 0
    no_reading = []

    for doc in docs:
        # Components and non-words keep only their meaning card.
        if hanzi.recognition_only(doc["word"]):
            if doc["quiz_type"] != "meaning":
                pruned += 1
                if apply:
                    db.remove(doc_ids=[doc.doc_id])
            continue

        if doc["quiz_type"] == "component" or "pinyin" in doc:
            continue

        readings = hanzi.readings(doc["word"])
        if not readings:
            no_reading.append(doc["word"])
            continue

        primary = hanzi.primary_reading(doc["word"], readings)

        tagged += 1
        if apply:
            db.update(dbops.set("pinyin", primary), doc_ids=[doc.doc_id])

        for secondary in readings:
            if secondary == primary:
                continue
            cloned += 1
            if apply:
                db.insert(
                    {
                        "type_": "card",
                        "card": dict(doc["card"]),
                        "word": doc["word"],
                        "quiz_type": doc["quiz_type"],
                        "pinyin": secondary,
                    }
                )

    mode = "applied" if apply else "dry run"
    print(
        "{}: tagged {} cards with their primary reading, "
        "cloned {} secondary-reading cards, "
        "pruned {} non-meaning cards from components".format(
            mode, tagged, cloned, pruned
        )
    )
    if no_reading:
        print(
            "skipped {} cards with no known reading: {}".format(
                len(no_reading), " ".join(sorted(set(no_reading)))
            )
        )


if __name__ == "__main__":
    migrate(apply="--apply" in sys.argv)
