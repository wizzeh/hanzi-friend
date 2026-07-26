"""Build grammar.json: the grammar pattern inventory for grammar quizzes.

Patterns come from the Chinese Grammar Wiki's level lists (A1 through C1,
roughly HSK 1-5). Each level page is a set of tables mapping a grammar
point to its structural pattern and an example; we scrape those rows and
keep the wiki's permalink id, so the full article is always reachable at

    https://resources.allsetlearning.com/chinese/grammar/<id>

Each point's "triggers" are the words its pattern hinges on: the CJK runs
in the pattern column, split against our word order with longest match
when a run isn't itself a word we teach (不只 becomes 不 + 只).

Each point also gets a "position": the index into the word order at which
it should be introduced. Points whose triggers arrive on schedule sit
right after their last trigger word. Trigger order alone front-loads too
much, though: a pattern's function word can arrive long before a learner
can handle the construction (白 is our first word, but its B2 "wasted
effort" sense leans on verbs and aspect), and fourteen points are purely
structural (reduplication, basic word order) with no triggers at all.
Both get level-gated instead: each level opens at half its median trigger
position, and its early or triggerless points spread evenly from there
to the next level's opening, in page order. A final pass keeps any two
intros a couple of words apart. Positions are unique, so the output is
sorted by position and reads as the grammar curriculum.

Run from anywhere: python data/grammar/build_grammar.py
Fetches the five level pages from the wiki (~250 KB total).
Output grammar.json is committed (force-add: .gitignore has *.json).

Content: Chinese Grammar Wiki (https://resources.allsetlearning.com/chinese/),
by AllSet Learning, under CC BY-NC-SA 3.0.
"""

import html
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

from loach_word_order import word_order  # noqa: E402

LEVELS = ["A1", "A2", "B1", "B2", "C1"]
PAGE = "https://resources.allsetlearning.com/chinese/grammar/{}_grammar_points"

WORDS = set(word_order)
CJK = re.compile(r"[㐀-鿿]+")

# Rows: permalink id, English name, pattern, example.
ROW = re.compile(
    r'<tr>\s*<td><a href="/chinese/grammar/([^"]+)"[^>]*>([^<]+)</a></td>\s*'
    r"<td>(.*?)</td>\s*<td>(.*?)</td>",
    re.S,
)
HEADLINE = re.compile(r'<span class="mw-headline" id="[^"]*">([^<]+)</span>')


def fetch(level: str) -> str:
    req = urllib.request.Request(
        PAGE.format(level), headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8")


def strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).replace(" ", " ").strip()


def triggers(pattern: str):
    """The words our pattern hinges on: its CJK runs, longest-match split
    against the word order when a run isn't itself a word we teach."""
    out = []
    for run in CJK.findall(pattern):
        i = 0
        while i < len(run):
            for j in range(len(run), i, -1):
                if run[i:j] in WORDS or j == i + 1:
                    out.append(run[i:j])
                    i = j
                    break
    seen = set()
    return [t for t in out if not (t in seen or seen.add(t))]


def parse(doc: str, level: str):
    """Walk headlines and table rows in document order, so each point
    lands under the section heading it appeared beneath."""
    events = [(m.start(), "head", strip_tags(m.group(1))) for m in HEADLINE.finditer(doc)]
    events += [(m.start(), "row", m.groups()) for m in ROW.finditer(doc)]
    events.sort(key=lambda e: e[0])

    category, points = None, []
    for _, kind, val in events:
        if kind == "head":
            category = val
            continue
        point_id, name, pattern, example = val
        pattern = strip_tags(pattern)
        points.append(
            {
                "id": point_id,
                "name": strip_tags(name),
                "level": level,
                "category": category,
                "pattern": pattern,
                "example": strip_tags(example),
                "triggers": triggers(pattern),
            }
        )
    return points


GAP = 2


def assign_positions(points):
    """Give every point its insertion index into the word order."""
    order = {w: i for i, w in enumerate(word_order)}
    unlock = {
        p["id"]: max(order[t] for t in p["triggers"]) if p["triggers"] else None
        for p in points
    }

    def med(values):
        values = sorted(values)
        return values[len(values) // 2]

    medians = {
        lvl: med([unlock[p["id"]] for p in points
                  if p["level"] == lvl and unlock[p["id"]] is not None])
        for lvl in LEVELS
    }
    floors = {lvl: medians[lvl] // 2 for lvl in LEVELS}
    span_end = {
        lvl: floors[LEVELS[i + 1]] if i + 1 < len(LEVELS) else medians[lvl]
        for i, lvl in enumerate(LEVELS)
    }

    raw = {}
    for lvl in LEVELS:
        lvl_points = [p for p in points if p["level"] == lvl]
        clamped = [
            p for p in lvl_points
            if unlock[p["id"]] is None or unlock[p["id"]] + 1 < floors[lvl]
        ]
        for k, p in enumerate(clamped):
            raw[p["id"]] = floors[lvl] + round(
                k * (span_end[lvl] - floors[lvl]) / len(clamped)
            )
        for p in lvl_points:
            if p["id"] not in raw:
                raw[p["id"]] = unlock[p["id"]] + 1

    last = -GAP
    for p in sorted(points, key=lambda p: (raw[p["id"]], LEVELS.index(p["level"]))):
        p["position"] = max(raw[p["id"]], last + GAP)
        last = p["position"]


def main():
    points = []
    for level in LEVELS:
        page_points = parse(fetch(level), level)
        print("{}: {} points".format(level, len(page_points)))
        points.extend(page_points)

    ids = [p["id"] for p in points]
    assert len(ids) == len(set(ids)), "duplicate grammar point ids"
    missing = {t for p in points for t in p["triggers"] if t not in WORDS}
    assert not missing, "triggers outside the word order: {}".format(missing)

    assign_positions(points)
    points.sort(key=lambda p: p["position"])

    out = HERE / "grammar.json"
    with open(out, "w") as f:
        json.dump(points, f, ensure_ascii=False, indent=1)
    print("{} points -> {}".format(len(points), out))


if __name__ == "__main__":
    main()
