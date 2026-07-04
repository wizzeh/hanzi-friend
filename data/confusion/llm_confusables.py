"""Generate llm_confusables.txt: LLM-curated visually-confusable pairs.

The component-overlap scorer in similarity.py can't see atomic
near-twins (已/己/巳, 人/入, 土/士) -- there are no shared components
to count. This asks GPT-5.5 for the famous learner confusions per
character, validates the answers against our character universe, and
writes one pair per line. similarity.py loads it alongside
same_stroke.txt as curated (full-score) pairs.

Run from the project root:

    python -m data.confusion.llm_confusables           # sample run
    python -m data.confusion.llm_confusables --full    # every relevant char
"""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
OUT_PATH = HERE / "llm_confusables.txt"
DONE_PATH = HERE / "llm_confusables.done"

PROMPT = """You are helping a Chinese-learning app detect confusable characters.

Which simplified Chinese characters do learners commonly confuse with {char} because they LOOK similar? Think of near-twin glyphs: same silhouette with a stroke added, lengthened, or hooked differently (like 未/末, 已/己/巳, 人/入, 土/士, 戊/戌/戍).

Only list visual confusions -- not characters that merely sound alike or mean something similar. Only single characters. If nothing is genuinely confusable with {char}, return an empty list."""

SCHEMA = {
    "type": "object",
    "properties": {
        "confusables": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["confusables"],
    "additionalProperties": False,
}


def query_chars(full: bool):
    sys.path.insert(0, str(HERE.parent.parent))
    import db as store
    from loach_word_order import word_order

    learned = [w for w in store.all_card_words() if len(w) == 1]
    if not full:
        return learned[:0] + list("已己巳未末干千于土士人入日曰戊天"), set(
            w for w in word_order if len(w) == 1
        )
    upcoming = [w for w in word_order if len(w) == 1][:2000]
    universe = set(w for w in word_order if len(w) == 1)
    return sorted(set(learned) | set(upcoming)), universe


def confusables_for(char: str, universe, service_tier: str):
    import enrich

    completion = enrich.env_client().chat.completions.create(
        model=enrich.MODEL,
        messages=[{"role": "user", "content": PROMPT.format(char=char)}],
        extra_body={"service_tier": service_tier},
        timeout=900 if service_tier == "flex" else 120,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "confusables", "strict": True, "schema": SCHEMA},
        },
    )
    answer = json.loads(completion.choices[0].message.content)["confusables"]
    return [c for c in answer if len(c) == 1 and c != char and c in universe]


def main():
    full = "--full" in sys.argv
    chars, universe = query_chars(full)
    tier = "flex" if full else "default"

    # Resume support: don't re-query chars from previous runs, and merge
    # into the existing pair file rather than replacing it.
    already = set()
    if full and DONE_PATH.exists():
        already = set(DONE_PATH.read_text(encoding="utf-8").split())
        chars = [c for c in chars if c not in already]
    pairs = set()
    if full and OUT_PATH.exists():
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                pairs.add(tuple(parts))
    print("{} chars to query ({} tier, {} already done)".format(
        len(chars), tier, len(already)))

    # On a terminal quota error, stop queuing new calls instead of
    # burning through the rest of the list.
    stop = threading.Event()

    def worker(char):
        if stop.is_set():
            return char, None
        try:
            return char, confusables_for(char, universe, tier)
        except Exception as e:
            if "insufficient_quota" in str(e):
                stop.set()
            print("  {} failed: {}".format(char, str(e)[:120]))
            return char, None

    done = 0
    queried = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for char, found in pool.map(worker, chars):
            done += 1
            if found is None:
                continue
            queried.append(char)
            for other in found:
                pairs.add(tuple(sorted((char, other))))
            if done % 50 == 0 or done == len(chars):
                print("{}/{} ({} pairs)".format(done, len(chars), len(pairs)))
            if not full:
                print("  {} -> {}".format(char, " ".join(found) or "(none)"))

    if full:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            for a, b in sorted(pairs):
                f.write("{}\t{}\n".format(a, b))
        with open(DONE_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(already | set(queried))))
        print("wrote {} ({} pairs, {} chars queried total)".format(
            OUT_PATH, len(pairs), len(already) + len(queried)))
        if stop.is_set():
            print("stopped early on quota exhaustion; rerun --full to resume")


if __name__ == "__main__":
    main()
