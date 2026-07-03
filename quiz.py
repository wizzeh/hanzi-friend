from typing import NamedTuple, List, Optional, Tuple
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

QUIZ_TYPES = [
    "component",
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


class AnnotatedCard(NamedTuple):
    card: fsrs.Card
    word: str
    quiz_type: str

    def serialize(self):
        return {
            "type_": "card",
            "card": self.card.to_dict(),
            "word": self.word,
            "quiz_type": self.quiz_type,
        }


def generate_cards_for_hanzi(word: str) -> List[AnnotatedCard]:
    if not hanzi.is_known_word(word):
        return []

    return [
        AnnotatedCard(card=fsrs.Card(), word=word, quiz_type=quiz_type)
        for quiz_type in QUIZ_TYPES
    ]


def learn_word(word: str):
    for card in generate_cards_for_hanzi(word):
        db.insert(card.serialize())


def card_is_due(val):
    due = datetime.fromisoformat(val["due"])

    return datetime.now().timestamp() > due.timestamp()


def next_quiz() -> Tuple[str, str, int]:
    """Pick a due card at random, or the next new word if nothing is due.

    Returns (word, quiz_type, number of due cards)."""
    Cards = Query()
    cards = db.search((Cards.type_ == "card") & (Cards.card.test(card_is_due)))

    if len(cards) > 0:
        quiz_card = choice(cards)
        return (quiz_card["word"], quiz_card["quiz_type"], len(cards))

    num_characters = store.characters_seen()
    while db.search(Cards.word == word_order[num_characters]):
        num_characters = num_characters + 1
    store.set_characters_seen(num_characters)

    return (word_order[num_characters], "intro", 0)


def review(word: str, quiz_type: str, difficulty: str):
    Cards = Query()
    doc = db.search(
        (Cards.type_ == "card") & (Cards.word == word) & (Cards.quiz_type == quiz_type)
    )[0]

    # Logs live at the doc level; older cards kept them inside the card dict.
    review_logs = doc.get("review_logs", doc["card"].get("review_logs", []))

    card = fsrs.Card.from_dict(doc["card"])
    rating = RATINGS.get(difficulty, fsrs.Rating.Again)

    new_card, review_log = scheduler.review_card(card, rating)
    review_logs.append(review_log.to_dict())

    db.update(dbops.set("card", new_card.to_dict()), doc_ids=[doc.doc_id])
    db.update(dbops.set("review_logs", review_logs), doc_ids=[doc.doc_id])
    db.update(dbops.set("last_migration", 1), doc_ids=[doc.doc_id])


def most_difficult_words() -> List[str]:
    Cards = Query()
    cards = list(
        db.search((Cards.type_ == "card") & (Cards.quiz_type == "translation-chinese"))
    )

    cards.sort(key=lambda card: card["card"]["difficulty"], reverse=True)

    return [c["word"] for c in cards]
