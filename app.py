from flask import Flask, render_template, request
from pypinyin.contrib.tone_convert import to_tone3, to_tone
from hanzipy.decomposer import HanziDecomposer
from hanzipy.dictionary import HanziDictionary
from hanzipy.exceptions import NotAHanziCharacter
from typing import NamedTuple, List
from radicals import radicals as all_radicals
from random import sample, shuffle, choice
import fsrs
from tinydb import TinyDB, Query
import tinydb.operations as dbops
from tinydb.table import Document
from openai import OpenAI
from filter_defs import filter_definitions


from dotenv import load_dotenv
import os

from loach_word_order import word_order

from datetime import datetime, timezone

import requests

load_dotenv()

decomposer = HanziDecomposer()
dictionary = HanziDictionary()
fsrs_system = fsrs.FSRS()
db = TinyDB("db.json")
ai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


try:
    db.insert(Document({"type_": "characters_seen", "number_seen": 0}, doc_id=1))
except ValueError:
    pass  # We already inserted it, that's fine.


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET")


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
    meaning: List[str]
    decomposition: Decomposition
    user_definition: str

    def pinyin_numbers(self):
        return to_tone3(self.pinyin, neutral_tone_with_five=True)


def try_define(hanzi):
    try:
        return dictionary.definition_lookup(hanzi)
    except NotAHanziCharacter:
        return [{"definition": "No definition found"}]
    except KeyError:
        return [{"definition": "No definition found"}]


def fixed_tone_convert(tone):
    return " ".join([to_tone(part.lower()) for part in tone.split(" ")])


def hanzi_info(hanzi: str) -> HanziInfo:
    pinyin_result = "/".join(
        list(
            set(
                [
                    fixed_tone_convert(entry["pinyin"])
                    if "pinyin" in entry
                    else "no pinyin"
                    for entry in filter_definitions(try_define(hanzi))
                ]
            )
        )
    )
    print(pinyin_result)

    decomposition = decomposer.decompose(hanzi, 2)
    decomp = decomposition["components"]

    decomp = [
        Component(
            hanzi=component,
            meaning=try_define(component)[0]["definition"],
            is_real=True,
        )
        for component in decomp
    ]

    # decomp = [
    #     component
    #     + " ("
    #     + dictionary.definition_lookup(component)[0]["definition"]
    #     + ")"
    #     for component in decomp
    # ]

    meaning = [
        (entry["pinyin"] if "pinyin" in entry else "?") + ": " + entry["definition"]
        for entry in filter_definitions(try_define(hanzi))
    ]

    Definitions = Query()
    user_definition = db.search(
        (Definitions.type_ == "definition") & (Definitions.hanzi == hanzi)
    )

    if user_definition:
        user_definition = user_definition[0]["definition"]
    else:
        user_definition = ""

    return HanziInfo(
        hanzi=hanzi,
        pinyin=pinyin_result,
        meaning=meaning,
        decomposition=Decomposition(radicals=decomp, true_length=len(decomp)),
        user_definition=user_definition,
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


def card_is_due(val):
    due = datetime.fromisoformat(val["due"])
    return datetime.now(timezone.utc) > due


@app.route("/")
def index():
    return render_template(
        "hanzi.html",
    )


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


def generate_cards_for_hanzi(hanzi: str) -> List[AnnotatedCard]:
    try:
        dictionary.definition_lookup(hanzi)
    except (NotAHanziCharacter, KeyError):
        return []

    def card(quiz_type: str) -> AnnotatedCard:
        return AnnotatedCard(card=fsrs.Card(), word=hanzi, quiz_type=quiz_type)

    return [
        card("component"),
        card("meaning"),
        card("pronunciation"),
        card("translation-chinese"),
        card("translation-english"),
    ]


class Translation(NamedTuple):
    english: str
    chinese: str

    @staticmethod
    def from_response(response: str):
        chinese, english = response.split("\n")
        chinese = chinese.strip()
        english = english.strip()
        return Translation(english=english, chinese=chinese)

    @staticmethod
    def for_hanzi(word: str):
        num_characters = db.search(Query().type_ == "characters_seen")[0]["number_seen"]
        avail_characters = word_order[:num_characters]

        word_chars = ""

        for char in avail_characters:
            word_chars += "- {}\n".format(char)

        message = """
You are an astute and culturally aware AI working as part of a Chinese language learning application. You will be given a list of characters which the student is expected to know. Your job is to generate a sentence containing only those words, which the student will be expected to translate.

An additional goal of your role in this program is to teach Chinese grammar, so make sentences grammatically interesting within the vocabulary you are given.

Your list of available characters is:
{}

The student is being quizzed on the character {}, so make sure to include that in your sentence.

Your response is being parsed by an API, so make sure to respond in the following two line format, and do not include any other text in your response:

Chinese Sentence
English Sentence""".format(word_chars, word)

        content = ""
        for _ in range(5):
            chat_completion = ai_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
                model="gpt-4o",
            )
            content = chat_completion.choices[0].message.content

            if word in content:
                break

        return Translation.from_response(content)


def next_quiz_query():
    Cards = Query()
    cards = db.search((Cards.type_ == "card") & (Cards.card.test(card_is_due)))

    Cards = Query()
    num_characters = db.search(Cards.type_ == "characters_seen")[0]["number_seen"]

    quiz = None
    next_hanzi = None

    if len(cards) > 0:
        quiz_card = choice(cards)
        quiz = quiz_card["quiz_type"]
        next_hanzi = quiz_card["word"]
    else:
        next_hanzi = word_order[num_characters]
        quiz = "intro"

    info = hanzi_info(next_hanzi)

    return (info, quiz)


def render_quiz(info, quiz):
    if quiz.startswith("translation"):
        translation = Translation.for_hanzi(info.hanzi)
        return render_template(
            "quizzes/translation.html",
            hanzi=info.hanzi,
            pinyin=info.pinyin,
            meaning=info.meaning,
            decomposition=generate_component_test(info.decomposition),
            pinyin_numbers=info.pinyin_numbers(),
            quiz_type=quiz,
            to_translate=translation.english
            if quiz.endswith("english")
            else translation.chinese,
            translation=translation.chinese
            if quiz.endswith("english")
            else translation.english,
            is_english=quiz.endswith("english"),
            user_definition=info.user_definition,
        )
    else:
        return render_template(
            "quizzes/" + quiz + ".html",
            hanzi=info.hanzi,
            pinyin=info.pinyin,
            meaning=info.meaning,
            decomposition=generate_component_test(info.decomposition),
            pinyin_numbers=info.pinyin_numbers(),
            quiz_type=quiz,
            words_known=db.search(Query().type_ == "characters_seen")[0]["number_seen"],
            user_definition=info.user_definition,
        )


def render_next():
    db.all()
    info, quiz = next_quiz_query()
    return render_quiz(info, quiz)


@app.route("/learn/<hanzi>", methods=["POST"])
def learn(hanzi):
    for card in generate_cards_for_hanzi(hanzi):
        db.insert(card.serialize())

    Cards = Query()
    db.update(dbops.increment("number_seen"), Cards.type_ == "characters_seen")
    return render_next()


@app.route("/learn", methods=["POST"])
def start():
    return render_next()


@app.route("/<hanzi>/<quiz_type>/<difficulty>", methods=["POST"])
def next_quiz(hanzi, quiz_type, difficulty):
    Cards = Query()
    card = db.search(
        (Cards.type_ == "card") & (Cards.word == hanzi) & (Cards.quiz_type == quiz_type)
    )[0]
    card_id = card.doc_id
    card = fsrs.Card.from_dict(card["card"])

    rating = None
    if difficulty == "easy":
        rating = fsrs.Rating.Easy
    elif difficulty == "good":
        rating = fsrs.Rating.Good
    elif difficulty == "hard":
        rating = fsrs.Rating.Hard
    else:
        rating = fsrs.Rating.Again

    new_card, _ = fsrs_system.review_card(card, rating)

    db.update(dbops.set("card", new_card.to_dict()), doc_ids=[card_id])

    db.all()

    return render_next()


@app.route("/table")
def table():
    return render_template("chart.html")


@app.route("/difficult")
def difficult():
    Cards = Query()
    cards = list(
        db.search((Cards.type_ == "card") & (Cards.quiz_type == "pronunciation"))
    )

    cards.sort(key=lambda card: card["card"]["difficulty"], reverse=True)

    return [c["word"] for c in cards]


@app.route("/<hanzi>/update-definition", methods=["POST"])
def set_definition(hanzi):
    Definitions = Query()

    definition = request.form["definition"]

    db.upsert(
        {"type_": "definition", "definition": definition, "hanzi": hanzi},
        (Definitions.type_ == "definition") & (Definitions.hanzi == hanzi),
    )

    return ("", 200)


@app.route("/pronounce/<text>", methods=["GET"])
def pronounce(text):
    cache = False
    if text in word_order:
        try:
            with open("static/audio/{}.mp3".format(text), "rb") as f:
                return f.read()
        except FileNotFoundError:
            cache = not text.strip().endswith("。")

    def cache_as_stream(content, do_cache):
        if not do_cache:
            for item in content:
                yield item

        with open("static/audio/{}.mp3".format(text), "wb") as f:
            for item in content:
                f.write(item)
                yield item

    key = os.environ.get("SPEECH_KEY")

    url = "https://eastus.tts.speech.microsoft.com/cognitiveservices/v1"

    message = """
    <speak version='1.0' xml:lang='zh-CN'><voice xml:lang='en-US' xml:gender='Female'
        name='zh-CN-XiaoxiaoNeural'>
            {}
    </voice></speak>
    """.format(text)

    headers = {
        "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
        "Content-Type": "application/ssml+xml",
        "Host": "eastus.tts.speech.microsoft.com",
        "Ocp-Apim-Subscription-Key": key,
        "User-Agent": "hanzi-tts",
    }

    r = requests.post(url, headers=headers, data=message)

    return cache_as_stream(r.iter_content(chunk_size=128), cache)


if __name__ == "__main__":
    app.run(debug=True, port=8089)
