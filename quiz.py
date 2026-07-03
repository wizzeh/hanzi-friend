from typing import NamedTuple, List, Optional
from random import choice
import json

import fsrs

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


def try_enrich(word: str):
    try:
        import enrich

        enrich.ensure_lexicon(word)
    except Exception as e:
        print("enrichment failed for {}: {}".format(word, e))


def learn_word(user_id: int, word: str, reading: Optional[str] = None):
    """Create cards for our word.

    Without a reading: the word is newly introduced. Cards are created
    for its primary reading (plus the component card), and any other
    readings are queued to be introduced later.

    With a reading: a queued secondary reading is being introduced."""
    if reading is not None:
        for quiz_type in PER_READING_QUIZZES:
            store.insert_card(user_id, word, quiz_type, reading, fsrs.Card().to_dict())
        store.remove_pending_pair(user_id, word, reading)
        return

    if not hanzi.is_known_word(word):
        return

    try_enrich(word)

    # Components and non-words just get a single recognition card.
    if hanzi.recognition_only(word):
        store.insert_card(user_id, word, "meaning", None, fsrs.Card().to_dict())
        return

    readings = hanzi.readings(word)
    primary = hanzi.primary_reading(word, readings)

    store.insert_card(user_id, word, "component", None, fsrs.Card().to_dict())
    for quiz_type in PER_READING_QUIZZES:
        store.insert_card(user_id, word, quiz_type, primary, fsrs.Card().to_dict())

    for secondary in readings:
        if secondary != primary:
            store.queue_pending_pair(user_id, word, secondary)


def next_quiz(user_id: int) -> QuizPick:
    """Pick a due card at random; else introduce a queued secondary
    reading; else introduce the next new word."""
    cards = store.due_cards(user_id)

    if cards:
        card = choice(cards)
        return QuizPick(
            word=card["word"],
            quiz_type=card["quiz_type"],
            reading=card["reading"] or None,
            card_id=card["id"],
            remaining=len(cards),
        )

    pending = store.next_pending_pair(user_id)
    if pending:
        return QuizPick(
            word=pending["word"],
            quiz_type="intro",
            reading=pending["reading"],
            card_id=None,
            remaining=0,
        )

    num_characters = store.characters_seen(user_id)
    while store.has_cards_for_word(user_id, word_order[num_characters]):
        num_characters = num_characters + 1
    store.set_characters_seen(user_id, num_characters)

    return QuizPick(
        word=word_order[num_characters],
        quiz_type="intro",
        reading=None,
        card_id=None,
        remaining=0,
    )


def review(user_id: int, card_id: int, difficulty: str):
    doc = store.get_card(card_id)
    if doc is None or doc["user_id"] != user_id:
        return

    card = fsrs.Card.from_dict(json.loads(doc["fsrs"]))
    rating = RATINGS.get(difficulty, fsrs.Rating.Again)

    new_card, review_log = scheduler.review_card(card, rating)
    store.update_card(card_id, new_card.to_dict(), review_log.to_dict(), int(rating))


def most_difficult_words(user_id: int) -> List[str]:
    return store.hardest_words(user_id, "translation-chinese")
