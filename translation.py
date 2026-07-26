from typing import NamedTuple

from dotenv import load_dotenv
from openai import OpenAI

import db as store
from hanzi import try_define
from filter_defs import filter_definitions

load_dotenv()

PROMPT = """
You are an astute and culturally aware AI working as part of a Chinese language learning application. You will be given a list of characters which the student is expected to know. Your job is to generate a sentence containing only those words, which the student will be expected to translate.

An additional goal of your role in this program is to teach Chinese grammar, so make sentences grammatically interesting within the vocabulary you are given.

Your list of available characters is:
{}

The student is being quizzed on the character {}, so make sure to include that in your sentence.{}

Your response is being parsed by an API, so make sure to respond in the following two line format, and do not include any other text in your response:

Chinese Sentence
English Sentence"""

GRAMMAR_PROMPT = """
You are an astute and culturally aware AI working as part of a Chinese language learning application. You will be given a list of characters which the student is expected to know. Your job is to generate a sentence containing only those words, which the student will be expected to translate.

Your list of available characters is:
{}

The student is being quizzed on the grammar pattern "{}" ({}), so build your sentence around that pattern. The pattern must genuinely appear in the sentence -- not just its words used incidentally. For reference, an example of the pattern in use: {}

Your response is being parsed by an API, so make sure to respond in the following two line format, and do not include any other text in your response:

Chinese Sentence
English Sentence"""


class Translation(NamedTuple):
    english: str
    chinese: str

    @staticmethod
    def from_response(response: str):
        # Curly apostrophes render in the CJK font, which looks off in English.
        response = response.replace("’", "'")
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        if len(lines) != 2:
            return None
        chinese, english = lines
        return Translation(english=english, chinese=chinese)

    @staticmethod
    def fallback(word: str):
        # If the AI keeps misbehaving, quiz the word on its own.
        definitions = filter_definitions(try_define(word)) or try_define(word)
        english = definitions[0]["definition"].replace("’", "'")
        return Translation(english=english, chinese=word)

    @staticmethod
    def for_hanzi(user_id: int, word: str, reading=None, glosses=None):
        # BYOK: sentences are generated on the user's own key. Routes
        # that get here are gated on the key existing.
        ai_client = OpenAI(api_key=store.user_keys(user_id)["openai_api_key"])

        avail_characters = store.known_words(user_id, "meaning")

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

    @staticmethod
    def for_grammar(user_id: int, point: dict):
        """A sentence exercising our grammar point, built from known
        vocabulary. Verification is necessarily loose -- a pattern isn't
        a substring -- so we check that some trigger word made it in and
        otherwise trust the model."""
        ai_client = OpenAI(api_key=store.user_keys(user_id)["openai_api_key"])

        avail_characters = store.known_words(user_id, "meaning")
        word_chars = "".join("- {}\n".format(char) for char in avail_characters)
        message = GRAMMAR_PROMPT.format(
            word_chars, point["name"], point["pattern"], point["example"]
        )

        for _ in range(5):
            chat_completion = ai_client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model="gpt-5.2",
            )
            translation = Translation.from_response(
                chat_completion.choices[0].message.content
            )
            if translation is not None and (
                not point["triggers"]
                or any(t in translation.chinese for t in point["triggers"])
            ):
                return translation

        # The wiki's own example, untranslated, beats no quiz at all.
        return Translation(english=point["pattern"], chinese=point["example"])
