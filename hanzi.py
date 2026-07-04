from typing import NamedTuple, List
from random import sample, shuffle

from hanzipy.decomposer import HanziDecomposer
from hanzipy.dictionary import HanziDictionary
from hanzipy.exceptions import NotAHanziCharacter
from pypinyin.contrib.tone_convert import to_tone3, to_tone

from radicals import radicals as all_radicals
from filter_defs import filter_definitions
import db
import similarity

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
    glosses: List[str]
    decomposition: Decomposition
    user_definition: str
    story: str


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


def format_sense(sense) -> str:
    formatted = fixed_tone_convert(sense["pinyin"]) + ": " + sense["gloss"]
    if sense["note"]:
        formatted += " ({})".format(sense["note"])
    return formatted


def learnable_senses(lexicon_entry):
    # Fall back to everything if no sense is marked learnable.
    senses = [s for s in lexicon_entry["senses"] if s["learn"]]
    return senses or lexicon_entry["senses"]


def recognition_only(word: str) -> bool:
    """True for components and non-words: nothing here is worth studying
    as a standalone word, we just want to recognize the shape's meaning."""
    lexicon_entry = db.get_lexicon(word)
    return bool(lexicon_entry) and not any(
        s["learn"] for s in lexicon_entry["senses"]
    )


def readings(word: str) -> List[str]:
    """The pronunciations of our word worth making cards for, curated
    when we have a lexicon entry, else everything CC-CEDICT lists."""
    lexicon_entry = db.get_lexicon(word)
    if lexicon_entry:
        return sorted(set(s["pinyin"] for s in learnable_senses(lexicon_entry)))
    return sorted(
        set(
            numbered_pinyin(entry["pinyin"])
            for entry in filter_definitions(try_define(word))
            if "pinyin" in entry
        )
    )


def primary_reading(word: str, readings_list: List[str]) -> str:
    """The most common reading, per pypinyin's frequency-based default."""
    from pypinyin import lazy_pinyin, Style

    default = " ".join(
        lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)
    )
    if default in readings_list:
        return default
    return readings_list[0] if readings_list else default


def hanzi_info(hanzi: str, reading: str = None, user_id: int = None) -> HanziInfo:
    """Info for our word, restricted to one pronunciation when a pair
    card names it."""
    lexicon_entry = db.get_lexicon(hanzi)
    story = ""

    if lexicon_entry:
        senses = learnable_senses(lexicon_entry)
        if reading:
            senses = [s for s in senses if s["pinyin"] == reading] or senses

        pinyin_result = "/".join(
            sorted(set(fixed_tone_convert(s["pinyin"]) for s in senses))
        )
        pinyin_variants = sorted(set(s["pinyin"] for s in senses))
        meaning = [format_sense(s) for s in senses]
        glosses = [s["gloss"] for s in senses]
        story = lexicon_entry["components"]
    else:
        # Raw CC-CEDICT for words we haven't enriched yet.
        entries = [
            entry
            for entry in filter_definitions(try_define(hanzi))
            if not (
                reading
                and "pinyin" in entry
                and numbered_pinyin(entry["pinyin"]) != reading
            )
        ]

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
            set(
                numbered_pinyin(entry["pinyin"])
                for entry in entries
                if "pinyin" in entry
            )
        )

        meaning = [
            (entry["pinyin"] if "pinyin" in entry else "?") + ": " + entry["definition"]
            for entry in entries
        ]
        glosses = [entry["definition"] for entry in entries]

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

    return HanziInfo(
        hanzi=hanzi,
        pinyin=pinyin_result,
        pinyin_variants=pinyin_variants,
        meaning=meaning,
        glosses=glosses,
        decomposition=Decomposition(radicals=decomp, true_length=len(decomp)),
        user_definition=db.user_definition(user_id, hanzi) if user_id else "",
        story=story,
    )


def generate_component_test(decomposition: Decomposition) -> Decomposition:
    component_hanzi = [component.hanzi for component in decomposition.radicals]

    false_radicals = list(filter(lambda x: x not in component_hanzi, all_radicals))

    number_false_radicals = max(3, 8 - len(decomposition.radicals))

    # Adversarial distractors first: radicals that look like the real
    # components; pad out with random ones.
    included_false_radicals = []
    for component in component_hanzi:
        for radical in similarity.similar_radicals(component, false_radicals, 2):
            if radical not in included_false_radicals:
                included_false_radicals.append(radical)
    included_false_radicals = included_false_radicals[:number_false_radicals]
    remaining = [r for r in false_radicals if r not in included_false_radicals]
    included_false_radicals += sample(
        remaining, number_false_radicals - len(included_false_radicals)
    )

    false_components = [
        Component(hanzi=radical, meaning="", is_real=False)
        for radical in included_false_radicals
    ]
    ret = decomposition.radicals + false_components
    shuffle(ret)

    return Decomposition(radicals=ret, true_length=decomposition.true_length)
