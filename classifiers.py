"""Measure-word pairings mined from CC-CEDICT annotations.

CC-CEDICT marks a noun's classifiers inline -- 狗 is "dog/CL:隻|只[zhi1],
條|条[tiao2]". We parse those into simplified-character + pinyin pairs,
dropping 个: it pairs with nearly every noun, so quizzing it teaches
nothing. Words whose only classifier is 个 get no card at all.

Run directly to add classifier cards for words already being learned:

    python classifiers.py
"""

import re
from typing import List, NamedTuple, Optional, Tuple
from random import sample, shuffle

from hanzi import try_define, filter_definitions, fixed_tone_convert


class Classifier(NamedTuple):
    hanzi: str  # simplified form
    pinyin: str  # numbered, e.g. zhi1


class Choice(NamedTuple):
    hanzi: str
    pinyin_display: str
    is_answer: bool


# A CL item is 隻|只[zhi1] (traditional|simplified[pinyin]) or 本[ben3]
# when both scripts share a form.
CL_ITEM = re.compile(r"([\u3400-\u9fff]+)(?:\|([\u3400-\u9fff]+))?\[([a-z0-9:]+)\]")

GE = "个"

# Plausible wrong answers: the classifiers a learner actually meets,
# each with its reading for display. Their pairings are what makes
# them good discrimination practice.
COMMON: List[Tuple[str, str]] = [
    ("只", "zhi1"),
    ("条", "tiao2"),
    ("本", "ben3"),
    ("张", "zhang1"),
    ("把", "ba3"),
    ("件", "jian4"),
    ("双", "shuang1"),
    ("杯", "bei1"),
    ("瓶", "ping2"),
    ("位", "wei4"),
    ("辆", "liang4"),
    ("台", "tai2"),
    ("座", "zuo4"),
    ("棵", "ke1"),
    ("颗", "ke1"),
    ("封", "feng1"),
    ("幅", "fu2"),
    ("门", "men2"),
    ("首", "shou3"),
    ("艘", "sou1"),
    ("架", "jia4"),
    ("块", "kuai4"),
    ("支", "zhi1"),
    ("匹", "pi3"),
    ("头", "tou2"),
    ("朵", "duo3"),
    ("片", "pian4"),
    ("篇", "pian1"),
    ("家", "jia1"),
    ("间", "jian1"),
    ("场", "chang3"),
    ("套", "tao4"),
    ("份", "fen4"),
    ("副", "fu4"),
    ("批", "pi1"),
    ("项", "xiang4"),
    ("滴", "di1"),
    ("遍", "bian4"),
    ("次", "ci4"),
    ("句", "ju4"),
]


def _cl_items(definition: str):
    for chunk in re.findall(r"CL:([^/]*)", definition):
        for match in CL_ITEM.finditer(chunk):
            traditional, simplified, pinyin = match.groups()
            yield simplified or traditional, pinyin


def classifiers_for(word: str) -> List[Classifier]:
    """The measure words CC-CEDICT pairs with our noun, in listed order,
    minus the generic 个."""
    out: List[Classifier] = []
    seen = set()
    for entry in filter_definitions(try_define(word)):
        for char, pinyin in _cl_items(entry["definition"]):
            if char != GE and char not in seen:
                seen.add(char)
                out.append(Classifier(char, pinyin))
    return out


def speak_phrase(word: str) -> Optional[str]:
    """The noun phrase our audio should say for this word: 一 plus its
    first classifier when it has one. None means say the bare word."""
    found = classifiers_for(word)
    if not found:
        return None
    return "一{}{}".format(found[0].hanzi, word)


def quiz_choices(word: str, count: int = 4) -> List[Choice]:
    """The answer buttons: every real classifier plus `count` plausible
    distractors, shuffled."""
    answers = classifiers_for(word)
    if not answers:
        return []

    taken = {c.hanzi for c in answers}
    pool = [(c, p) for c, p in COMMON if c not in taken]

    choices = [
        Choice(c.hanzi, fixed_tone_convert(c.pinyin), True) for c in answers
    ]
    for char, pinyin in sample(pool, min(count, len(pool))):
        choices.append(Choice(char, fixed_tone_convert(pinyin), False))
    shuffle(choices)
    return choices


def backfill_cards():
    """Insert classifier cards for words already being learned, for the
    students whose vocabulary predates the quiz."""
    import fsrs

    import db as store

    made = 0
    for word in store.all_card_words():
        if not classifiers_for(word):
            continue
        for user_id in store.card_users(word):
            if store.card_id_for(user_id, word, "classifier") is None:
                store.insert_card(
                    user_id, word, "classifier", None, fsrs.Card().to_dict()
                )
                made += 1
    print("{} classifier cards added".format(made))


if __name__ == "__main__":
    backfill_cards()
