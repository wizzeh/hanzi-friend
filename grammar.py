"""The grammar curriculum: Chinese Grammar Wiki patterns, ordered by
insertion position into the word order (see data/grammar/build_grammar.py).

A user's progress through the curriculum is a single counter,
grammar_seen: how many points have been introduced. Points are always
introduced in curriculum order, so the counter identifies the next one.
"""

import json
import os
from typing import Optional

ARTICLE_BASE = "https://resources.allsetlearning.com/chinese/grammar/"

with open(os.path.join(os.path.dirname(__file__), "data", "grammar", "grammar.json")) as f:
    CURRICULUM = json.load(f)

BY_ID = {p["id"]: p for p in CURRICULUM}


def next_point(grammar_seen: int) -> Optional[dict]:
    if grammar_seen < len(CURRICULUM):
        return CURRICULUM[grammar_seen]
    return None


def point(point_id: str) -> dict:
    return BY_ID[point_id]


def is_point_id(word: str) -> bool:
    return word in BY_ID


def article_url(point_id: str) -> str:
    return ARTICLE_BASE + point_id


def backlog(grammar_seen: int, characters_seen: int):
    """Points due for introduction: reached by our word progress but not
    yet introduced."""
    return [
        p
        for p in CURRICULUM[grammar_seen:]
        if p["position"] <= characters_seen
    ]
