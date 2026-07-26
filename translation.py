import json
from typing import NamedTuple

from dotenv import load_dotenv
from openai import OpenAI

import db as store
from hanzi import try_define
from filter_defs import filter_definitions

load_dotenv()

MODEL = "gpt-5.6-luna"

PROMPT = """You are generating a translation exercise for a Chinese spaced-repetition app. The student is an adult English learner of simplified Chinese.

Write one natural Chinese sentence for the student to translate into English, plus its English translation.

Rules:
- Use ONLY characters from the student's vocabulary below. Every character in the sentence must appear there; punctuation is fine.
- The sentence must contain {word}.{sense_hint}
- One or two clauses, roughly 6-20 characters.
- Prefer sentences that exercise real grammar (aspect, comparison, complements) over bare subject-verb-object, but never at the cost of sounding natural. Write idiomatic Chinese, not English translated word-by-word.

Vocabulary the student knows:
{vocab}"""

GRAMMAR_PROMPT = """You are generating a translation exercise for a Chinese spaced-repetition app. The student is an adult English learner of simplified Chinese.

Write one natural Chinese sentence for the student to translate into English, plus its English translation.

The student is being quizzed on the grammar pattern "{name}" ({pattern}), so build the sentence around that pattern. The pattern must genuinely appear in the sentence -- not just its words used incidentally. For reference, an example of the pattern in use: {example} -- do not copy it; write your own sentence.

Rules:
- Use ONLY characters from the student's vocabulary below. Every character in the sentence must appear there; punctuation is fine.
- One or two clauses, roughly 6-20 characters.
- Write idiomatic Chinese, not English translated word-by-word.

Vocabulary the student knows:
{vocab}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "chinese": {"type": "string"},
        "english": {"type": "string"},
    },
    "required": ["chinese", "english"],
    "additionalProperties": False,
}

GRAMMAR_SCHEMA = {
    "type": "object",
    "properties": {
        "chinese": {"type": "string"},
        "english": {"type": "string"},
        "pattern_usage": {
            "type": "string",
            "description": "The words in your sentence that instantiate the pattern",
        },
    },
    "required": ["chinese", "english", "pattern_usage"],
    "additionalProperties": False,
}


def _is_hanzi(char: str) -> bool:
    # Main CJK block plus Extension A covers our vocabulary sources.
    code = ord(char)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF


def _unknown_hanzi(sentence: str, allowed: set) -> set:
    """Hanzi in our sentence outside the allowed set. Punctuation,
    Latin, and digits are always fine."""
    return set(c for c in sentence if _is_hanzi(c) and c not in allowed)


def _generate(ai_client, message: str, schema_name: str, schema: dict, validate):
    """Up to 5 attempts, feeding each validation failure back to the
    model rather than retrying blind. validate returns a complaint
    string, or None to accept. Returns the reply dict, or None."""
    messages = [{"role": "user", "content": message}]
    for _ in range(5):
        completion = ai_client.chat.completions.create(
            messages=messages,
            model=MODEL,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        content = completion.choices[0].message.content
        if not content:
            continue
        reply = json.loads(content)

        problem = validate(reply)
        if problem is None:
            return reply
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": problem + " Try again."})
    return None


class Translation(NamedTuple):
    english: str
    chinese: str

    @staticmethod
    def _from_reply(reply: dict):
        # Curly apostrophes render in the CJK font, which looks off in English.
        return Translation(
            english=reply["english"].replace("’", "'"),
            chinese=reply["chinese"],
        )

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

        vocab = store.known_words(user_id, "meaning")
        allowed = set(c for w in vocab for c in w) | set(word)

        sense_hint = ""
        if reading and glosses:
            sense_hint = (
                " Use it specifically in the sense \"{}\" (pronounced {})."
            ).format("; ".join(glosses), reading)

        message = PROMPT.format(
            word=word, sense_hint=sense_hint, vocab=" ".join(vocab)
        )

        def validate(reply):
            if word not in reply["chinese"]:
                return "Your sentence did not contain {}.".format(word)
            unknown = _unknown_hanzi(reply["chinese"], allowed)
            if unknown:
                return "Your sentence used {}, which the student has not learned.".format(
                    " ".join(sorted(unknown))
                )
            return None

        reply = _generate(ai_client, message, "translation", SCHEMA, validate)
        if reply is None:
            return Translation.fallback(word)
        return Translation._from_reply(reply)

    @staticmethod
    def for_grammar(user_id: int, point: dict):
        """A sentence exercising our grammar point, built from known
        vocabulary. Verification is necessarily loose -- a pattern isn't
        a substring -- so we check that some trigger word made it in,
        make the model name its own instantiation, and otherwise trust
        it."""
        ai_client = OpenAI(api_key=store.user_keys(user_id)["openai_api_key"])

        vocab = store.known_words(user_id, "meaning")
        # The pattern's own trigger words are fair game even when the
        # student hasn't carded them individually.
        allowed = set(c for w in vocab for c in w)
        allowed |= set(c for t in point["triggers"] for c in t)

        message = GRAMMAR_PROMPT.format(
            name=point["name"],
            pattern=point["pattern"],
            example=point["example"],
            vocab=" ".join(vocab),
        )

        def validate(reply):
            if point["triggers"] and not any(
                t in reply["chinese"] for t in point["triggers"]
            ):
                return 'Your sentence does not use the pattern "{}".'.format(
                    point["pattern"]
                )
            unknown = _unknown_hanzi(reply["chinese"], allowed)
            if unknown:
                return "Your sentence used {}, which the student has not learned.".format(
                    " ".join(sorted(unknown))
                )
            return None

        reply = _generate(
            ai_client, message, "grammar_translation", GRAMMAR_SCHEMA, validate
        )
        if reply is None:
            # The wiki's own example, untranslated, beats no quiz at all.
            return Translation(english=point["pattern"], chinese=point["example"])
        return Translation._from_reply(reply)
