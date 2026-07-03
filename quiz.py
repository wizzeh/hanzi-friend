from typing import NamedTuple, List, Optional
from random import choice
from datetime import datetime

import fsrs
from tinydb import Query
import tinydb.operations as dbops

from db import db
import db as store
import hanzi
from loach_word_order import word_order

scheduler = fsrs.Scheduler(
    desired_retention=0.7,
    enable_fuzzing=True,
)

# One card per (word, reading) for these; component cards are per word,
# since decomposition doesn't depend on pronunciation.
PER_READING_QUIZZES = [
    "meaning",
    "pronunciation",
    "translation-chinese",
    "translation-english",
]

RATINGS = {
    "easy": fsrs.Rating.Easy,
    "good": fsrs.Rating.Good,
    "hard": fsrs.Rating.Hard,
    "again": fsrs.Rating.Again,
}


class QuizPick(NamedTuple):
    word: str
    quiz_type: str
    reading: Optional[str]
    card_id: Optional[int]
    remaining: int


def serialize_card(card: fsrs.Card, word: str, quiz_type: str, reading):
    return {
        "type_": "card",
        "card": card.to_dict(),
        "word": word,
        "quiz_type": quiz_type,
        "pinyin": reading,
    }


def try_enrich(word: str):
    try:
        import enrich

        enrich.ensure_lexicon(word)
    except Exception as e:
        print("enrichment failed for {}: {}".format(word, e))


def learn_word(word: str, reading: Optional[str] = None):
    """Create cards for our word.

    Without a reading: the word is newly introduced. Cards are created
    for its primary reading (plus the component card), and any other
    readings are queued to be introduced later.

    With a reading: a queued secondary reading is being introduced."""
    Pending = Query()

    if reading is not None:
        for quiz_type in PER_READING_QUIZZES:
            db.insert(serialize_card(fsrs.Card(), word, quiz_type, reading))
        db.remove(
            (Pending.type_ == "pending_pair")
            & (Pending.word == word)
            & (Pending.pinyin == reading)
        )
        return

    if not hanzi.is_known_word(word):
        return

    try_enrich(word)

    # Components and non-words just get a single recognition card.
    if hanzi.recognition_only(word):
        db.insert(serialize_card(fsrs.Card(), word, "meaning", None))
        return

    readings = hanzi.readings(word)
    primary = hanzi.primary_reading(word, readings)

    db.insert(serialize_card(fsrs.Card(), word, "component", None))
    for quiz_type in PER_READING_QUIZZES:
        db.insert(serialize_card(fsrs.Card(), word, quiz_type, primary))

    for secondary in readings:
        if secondary != primary:
            db.insert({"type_": "pending_pair", "word": word, "pinyin": secondary})


def card_is_due(val):
    due = datetime.fromisoformat(val["due"])

    return datetime.now().timestamp() > due.timestamp()


def next_quiz() -> QuizPick:
    """Pick a due card at random; else introduce a queued secondary
    reading; else introduce the next new word."""
    Cards = Query()
    cards = db.search((Cards.type_ == "card") & (Cards.card.test(card_is_due)))

    if len(cards) > 0:
        card = choice(cards)
        return QuizPick(
            word=card["word"],
            quiz_type=card["quiz_type"],
            reading=card.get("pinyin"),
            card_id=card.doc_id,
            remaining=len(cards),
        )

    pending = db.search(Cards.type_ == "pending_pair")
    if pending:
        pair = pending[0]
        return QuizPick(
            word=pair["word"],
            quiz_type="intro",
            reading=pair["pinyin"],
            card_id=None,
            remaining=0,
        )

    num_characters = store.characters_seen()
    while db.search(Cards.word == word_order[num_characters]):
        num_characters = num_characters + 1
    store.set_characters_seen(num_characters)

    return QuizPick(
        word=word_order[num_characters],
        quiz_type="intro",
        reading=None,
        card_id=None,
        remaining=0,
    )


def review(card_id: int, difficulty: str):
    doc = db.get(doc_id=card_id)

    # Logs live at the doc level; older cards kept them inside the card dict.
    review_logs = doc.get("review_logs", doc["card"].get("review_logs", []))

    card = fsrs.Card.from_dict(doc["card"])
    rating = RATINGS.get(difficulty, fsrs.Rating.Again)

    new_card, review_log = scheduler.review_card(card, rating)
    review_logs.append(review_log.to_dict())

    db.update(dbops.set("card", new_card.to_dict()), doc_ids=[card_id])
    db.update(dbops.set("review_logs", review_logs), doc_ids=[card_id])
    db.update(dbops.set("last_migration", 1), doc_ids=[card_id])


def most_difficult_words() -> List[str]:
    Cards = Query()
    cards = list(
        db.search((Cards.type_ == "card") & (Cards.quiz_type == "translation-chinese"))
    )

    cards.sort(key=lambda card: card["card"]["difficulty"], reverse=True)

    return [c["word"] for c in cards]
