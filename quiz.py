from typing import NamedTuple, List, Optional, Dict
from random import choice
from datetime import datetime, timezone
import json

import fsrs

import db as store
import grammar
import hanzi
import similarity
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


def try_enrich(user_id: int, word: str):
    try:
        import enrich

        api_key = store.user_keys(user_id)["openai_api_key"] or None
        enrich.ensure_lexicon(word, api_key=api_key)
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

    store.clear_last_intro_grammar(user_id)
    try_enrich(user_id, word)

    # Components and non-words just get a single recognition card.
    if hanzi.recognition_only(word):
        store.insert_card(user_id, word, "meaning", None, fsrs.Card().to_dict())
        return

    readings = hanzi.readings(word)
    primary = hanzi.primary_reading(word, readings)

    store.insert_card(user_id, word, "component", None, fsrs.Card().to_dict())
    for quiz_type in PER_READING_QUIZZES:
        store.insert_card(user_id, word, quiz_type, primary, fsrs.Card().to_dict())

    # If this word has a known lookalike, quiz them against each other.
    if len(word) == 1:
        learned = [w for w in store.user_words(user_id) if w != word]
        lookalikes = similarity.top_visual(word, learned, k=1)
        if lookalikes and lookalikes[0][1] >= similarity.CONTRAST_THRESHOLD:
            store.insert_card(user_id, word, "contrast", None, fsrs.Card().to_dict())

    for secondary in readings:
        if secondary != primary:
            store.queue_pending_pair(user_id, word, secondary)


def learn_grammar(user_id: int, point_id: str, known: bool = False):
    """Create the review card for our grammar point. A point the user
    already commands still gets a card -- seeded with an Easy review so
    it schedules weeks out and the claim gets verified eventually."""
    store.insert_card(user_id, point_id, "grammar", None, fsrs.Card().to_dict())
    store.advance_grammar_seen(user_id)

    if known:
        card_id = store.card_id_for(user_id, point_id, "grammar")
        if card_id is not None:
            review(user_id, card_id, "easy")


def next_quiz(user_id: int) -> QuizPick:
    """Pick a due card at random; else introduce, in order of priority:
    a pre-learn word from the wild, a queued secondary reading, or --
    alternating so a grammar backlog never freezes vocabulary intake --
    a due grammar point or the next new word from the frequency order."""
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

    prelearn = store.next_prelearn(user_id)
    if prelearn:
        return QuizPick(
            word=prelearn,
            quiz_type="intro",
            reading=None,
            card_id=None,
            remaining=0,
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

    grammar_seen, last_was_grammar = store.grammar_state(user_id)
    point = grammar.next_point(grammar_seen)
    if point and point["position"] <= num_characters and not last_was_grammar:
        return QuizPick(
            word=point["id"],
            quiz_type="grammar-intro",
            reading=None,
            card_id=None,
            remaining=0,
        )
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


def _humanize_interval(delta) -> str:
    minutes = delta.total_seconds() / 60
    if minutes < 60:
        return "{}m".format(max(1, round(minutes)))
    if minutes < 60 * 24:
        return "{}h".format(round(minutes / 60))
    days = minutes / (60 * 24)
    if days < 32:
        return "{}d".format(round(days))
    return "{}mo".format(round(days / 30.4))


def preview_intervals(card_id: int) -> Optional[Dict[str, str]]:
    """What each rating would schedule, for the feedback buttons.
    With fuzzing enabled the real review may land a little off these,
    which is fine at the granularity we display."""
    doc = store.get_card(card_id)
    if doc is None:
        return None

    now = datetime.now(timezone.utc)
    out = {}
    for name, rating in RATINGS.items():
        card = fsrs.Card.from_dict(json.loads(doc["fsrs"]))
        new_card, _ = scheduler.review_card(card, rating, review_datetime=now)
        out[name] = _humanize_interval(new_card.due - now)
    return out


def most_difficult_words(user_id: int) -> List[str]:
    return store.hardest_words(user_id, "translation-chinese")
