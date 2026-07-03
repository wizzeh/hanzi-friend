"""AI-curated lexicon entries built on top of the raw CC-CEDICT data.

CC-CEDICT glosses are noisy for learning: archaic variants, cross-reference
stubs, and encyclopedia asides mixed in with the senses we actually want to
study. We hand the raw entries to a strong model to group them by
pronunciation, rewrite them as learner-friendly glosses, and flag which
senses are worth learning. The model can only rework what we give it -- any
pinyin outside the CC-CEDICT reading set is rejected and retried.

Run directly to backfill every word that has cards:

    python enrich.py
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

import db as store
from hanzi import try_define, numbered_pinyin, decomposer

load_dotenv()

ai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

LEXICON_VERSION = 1
MODEL = "gpt-5.5"

PROMPT = """You are a lexicographer for a Chinese spaced-repetition app. The student is an adult English speaker somewhere around HSK 3. Below are the raw CC-CEDICT entries for {word}, plus its graphical components.

Rework the entries into study-ready senses:

- Group the entries by pronunciation and meaning: one sense per pronunciation-meaning cluster. Distinct meanings that share a pronunciation stay one sense unless they are genuinely unrelated words.
- Write each gloss as a short, plain-English definition (a few words to one line). Drop encyclopedia asides, usage trivia, and cross-references.
- Order senses from most to least worth studying.
- Set "learn" to false for senses a learner should not be quizzed on: archaic or literary-only usage, unofficial variants, cross-reference stubs, surnames and other proper nouns, and readings that only occur in rare compounds. When in doubt about whether a sense is current, common usage, set "learn" to false.
- "note" is one short clause of context when helpful (register, what it contrasts with, common word it appears in), else "".
- Use the pinyin exactly as given in the entries (tone numbers). Never introduce a pronunciation that is not in the entries below.
- "components": one memorable line explaining how the graphical components build the character, as a memory hook. If the components are unhelpful fragments, give the best honest hook you can or "".

CC-CEDICT entries for {word}:
{entries}

Graphical components: {components}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "senses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pinyin": {"type": "string"},
                    "gloss": {"type": "string"},
                    "learn": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                "required": ["pinyin", "gloss", "learn", "note"],
                "additionalProperties": False,
            },
        },
        "components": {"type": "string"},
    },
    "required": ["senses", "components"],
    "additionalProperties": False,
}


def cedict_entries(word):
    return [entry for entry in try_define(word) if "pinyin" in entry]


def valid_entry(entry, allowed_pinyin):
    return len(entry["senses"]) > 0 and all(
        numbered_pinyin(sense["pinyin"]) in allowed_pinyin
        for sense in entry["senses"]
    )


def enrich_word(word: str):
    """Build a lexicon entry for our word. Returns None if the model can't
    produce a valid one."""
    entries = cedict_entries(word)
    if not entries:
        return None

    allowed_pinyin = set(numbered_pinyin(entry["pinyin"]) for entry in entries)
    components = [
        component
        for component in decomposer.decompose(word, 2)["components"]
        if component != decomposer.noglyph
    ]

    message = PROMPT.format(
        word=word,
        entries="\n".join(
            "- {}: {}".format(entry["pinyin"], entry["definition"])
            for entry in entries
        ),
        components=" ".join(components) if components else "(none)",
    )

    for _ in range(3):
        completion = ai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": message}],
            # Flex processing: batch pricing on the live API, slower is fine.
            extra_body={"service_tier": "flex"},
            timeout=900,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "lexicon_entry",
                    "strict": True,
                    "schema": SCHEMA,
                },
            },
        )
        entry = json.loads(completion.choices[0].message.content)

        if valid_entry(entry, allowed_pinyin):
            for sense in entry["senses"]:
                sense["pinyin"] = numbered_pinyin(sense["pinyin"])
            entry["version"] = LEXICON_VERSION
            return entry

    return None


def ensure_lexicon(word: str):
    """Fetch our cached lexicon entry, enriching and caching it if new."""
    cached = store.get_lexicon(word)
    if cached is not None and cached["version"] == LEXICON_VERSION:
        return cached

    entry = enrich_word(word)
    if entry is not None:
        store.put_lexicon(word, entry)
    return entry


def backfill():
    from concurrent.futures import ThreadPoolExecutor
    from tinydb import Query

    words = sorted(set(card["word"] for card in store.db.search(Query().type_ == "card")))
    todo = [
        word
        for word in words
        if (cached := store.get_lexicon(word)) is None
        or cached["version"] != LEXICON_VERSION
    ]
    print("{} words, {} to enrich".format(len(words), len(todo)))

    # Enrich in parallel but write from this thread; TinyDB isn't thread-safe.
    with ThreadPoolExecutor(max_workers=8) as pool:
        done = 0
        failed = []
        for word, entry in zip(todo, pool.map(enrich_word, todo)):
            done += 1
            if entry is None:
                failed.append(word)
            else:
                store.put_lexicon(word, entry)
            if done % 25 == 0 or done == len(todo):
                print("{}/{} ({} failed)".format(done, len(todo), len(failed)))

    if failed:
        print("failed:", " ".join(failed))


if __name__ == "__main__":
    backfill()
