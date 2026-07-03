from typing import NamedTuple, List
from random import sample, shuffle

from hanzipy.decomposer import HanziDecomposer
from hanzipy.dictionary import HanziDictionary
from hanzipy.exceptions import NotAHanziCharacter
from pypinyin.contrib.tone_convert import to_tone3, to_tone

from radicals import radicals as all_radicals
from filter_defs import filter_definitions
import db

decomposer = HanziDecomposer()
dictionary = HanziDictionary()


class Component(NamedTuple):
    hanzi: str
    meaning: str
    is_real: bool


class Decomposition(NamedTuple):
    radicals: List[Component]
    true_length: int


class HanziInfo(NamedTuple):
    hanzi: str
    pinyin: str
    pinyin_variants: List[str]
    meaning: List[str]
    decomposition: Decomposition
    user_definition: str


def try_define(hanzi):
    try:
        return dictionary.definition_lookup(hanzi)
    except NotAHanziCharacter:
        return [{"definition": "No definition found"}]
    except KeyError:
        return [{"definition": "No definition found"}]


def is_known_word(hanzi: str) -> bool:
    try:
        dictionary.definition_lookup(hanzi)
        return True
    except (NotAHanziCharacter, KeyError):
        return False


def normalize_cedict_pinyin(pinyin):
    # CC-CEDICT writes ü as "u:", which pypinyin doesn't understand.
    return pinyin.lower().replace("u:", "v")


def fixed_tone_convert(tone):
    return " ".join(
        to_tone(part) for part in normalize_cedict_pinyin(tone).split(" ")
    )


def numbered_pinyin(pinyin):
    return " ".join(
        to_tone3(syllable, neutral_tone_with_five=True)
        for syllable in normalize_cedict_pinyin(pinyin).split(" ")
    )


def hanzi_info(hanzi: str) -> HanziInfo:
    entries = filter_definitions(try_define(hanzi))

    pinyin_result = "/".join(
        sorted(
            set(
                fixed_tone_convert(entry["pinyin"])
                if "pinyin" in entry
                else "no pinyin"
                for entry in entries
            )
        )
    )

    pinyin_variants = sorted(
        set(numbered_pinyin(entry["pinyin"]) for entry in entries if "pinyin" in entry)
    )

    decomposition = decomposer.decompose(hanzi, 2)

    decomp = [
        Component(
            hanzi=component,
            meaning=try_define(component)[0]["definition"],
            is_real=True,
        )
        for component in decomposition["components"]
        if component != decomposer.noglyph
    ]

    meaning = [
        (entry["pinyin"] if "pinyin" in entry else "?") + ": " + entry["definition"]
        for entry in entries
    ]

    return HanziInfo(
        hanzi=hanzi,
        pinyin=pinyin_result,
        pinyin_variants=pinyin_variants,
        meaning=meaning,
        decomposition=Decomposition(radicals=decomp, true_length=len(decomp)),
        user_definition=db.user_definition(hanzi),
    )


def generate_component_test(decomposition: Decomposition) -> Decomposition:
    component_hanzi = [component.hanzi for component in decomposition.radicals]

    false_radicals = list(filter(lambda x: x not in component_hanzi, all_radicals))

    number_false_radicals = max(3, 8 - len(decomposition.radicals))
    included_false_radicals = sample(false_radicals, number_false_radicals)

    false_components = [
        Component(hanzi=radical, meaning="", is_real=False)
        for radical in included_false_radicals
    ]
    ret = decomposition.radicals + false_components
    shuffle(ret)

    return Decomposition(radicals=ret, true_length=decomposition.true_length)
