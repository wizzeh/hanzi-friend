from typing import NamedTuple
import os

from dotenv import load_dotenv
from openai import OpenAI
from tinydb import Query

from db import db
from hanzi import try_define
from filter_defs import filter_definitions

load_dotenv()

ai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROMPT = """
You are an astute and culturally aware AI working as part of a Chinese language learning application. You will be given a list of characters which the student is expected to know. Your job is to generate a sentence containing only those words, which the student will be expected to translate.

An additional goal of your role in this program is to teach Chinese grammar, so make sentences grammatically interesting within the vocabulary you are given.

Your list of available characters is:
{}

The student is being quizzed on the character {}, so make sure to include that in your sentence.{}

Your response is being parsed by an API, so make sure to respond in the following two line format, and do not include any other text in your response:

Chinese Sentence
English Sentence"""


class Translation(NamedTuple):
    english: str
    chinese: str

    @staticmethod
    def from_response(response: str):
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        if len(lines) != 2:
            return None
        chinese, english = lines
        return Translation(english=english, chinese=chinese)

    @staticmethod
    def fallback(word: str):
        # If the AI keeps misbehaving, quiz the word on its own.
        definitions = filter_definitions(try_define(word)) or try_define(word)
        return Translation(english=definitions[0]["definition"], chinese=word)

    @staticmethod
    def for_hanzi(word: str, reading=None, glosses=None):
        cards = Query()
        avail_characters = sorted(
            set(
                card["word"]
                for card in db.search(
                    (cards.type_ == "card") & (cards.quiz_type == "meaning")
                )
            )
        )

        sense_hint = ""
        if reading and glosses:
            sense_hint = (
                " Use it specifically in the sense \"{}\" (pronounced {})."
            ).format("; ".join(glosses), reading)

        word_chars = "".join("- {}\n".format(char) for char in avail_characters)
        message = PROMPT.format(word_chars, word, sense_hint)

        for _ in range(5):
            chat_completion = ai_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
                model="gpt-5.2",
            )
            content = chat_completion.choices[0].message.content

            translation = Translation.from_response(content)
            if translation is not None and word in translation.chinese:
                return translation

        return Translation.fallback(word)
