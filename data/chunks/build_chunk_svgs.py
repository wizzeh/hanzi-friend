"""Build chunk_svgs.json: real stroke outlines for codepoint-less chunks.

The decomposition database names some chunks by number because Unicode
has no character for them (the top of 学 is ⺍ over 冖). CSS-stacking
their parts only approximates the shape, so we carve the true shape out
of a real character instead: Make Me a Hanzi has per-stroke SVG paths
for ~9.5k characters, and a top/left chunk is simply the parent's first
k strokes (bottom/right: last k). We take mmah's own stroke-to-component
matches when both databases split the parent the same way, else carve by
stroke order, validated against the sibling's stroke count and the
chunk's expected position so interleaved or overlapping shapes get
rejected (those fall back to CSS stacking in the app) rather than
miscarved.

Run from anywhere: python data/chunks/build_chunk_svgs.py
Downloads the mmah data files on first run (~33 MB, gitignored).
Output chunk_svgs.json is committed (force-add: .gitignore has *.json).

Stroke data: Make Me a Hanzi (https://github.com/skishore/makemeahanzi),
derived from Arphic fonts under the Arphic Public License.
"""

import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

from chunking import (  # noqa: E402
    NUMBERED_PRODUCTIVITY,
    _composite,
    _contains,
    arrangement_of,
    decomposer,
)

MMAH_BASE = "https://raw.githubusercontent.com/skishore/makemeahanzi/master/"

IDS_BINARY = set("⿰⿱⿴⿵⿶⿷⿸⿹⿺⿻")
IDS_TERNARY = set("⿲⿳")
NUM_RE = re.compile(r"-?\d+\.?\d*")


def load_mmah(name):
    path = HERE / name
    if not path.exists():
        print(f"downloading {name}...")
        urllib.request.urlretrieve(MMAH_BASE + name, path)
    entries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            entries[entry["character"]] = entry
    return entries


dictionary = load_mmah("dictionary.txt")
graphics = load_mmah("graphics.txt")


def top_level_arity(ids):
    """How many operands mmah's top-level split has, or None."""
    if not ids or ids[0] not in IDS_BINARY | IDS_TERNARY:
        return None
    return 3 if ids[0] in IDS_TERNARY else 2


def stroke_count(piece):
    """Stroke count of a glyph or numbered node, or None if unknown."""
    if piece.isdigit():
        entry = decomposer.characters.get(piece)
        if not entry:
            return None
        counts = [stroke_count(c) for c in entry["components"]]
        return None if any(c is None for c in counts) else sum(counts)
    glyph = graphics.get(piece)
    return len(glyph["strokes"]) if glyph else None


def path_bbox(paths):
    xs, ys = [], []
    for path in paths:
        nums = [float(n) for n in NUM_RE.findall(path)]
        xs.extend(nums[0::2])
        ys.extend(900 - n for n in nums[1::2])  # rendered y after the flip
    return min(xs), min(ys), max(xs), max(ys)


def plausible_position(bbox, arrangement, index):
    """A top chunk should sit in the upper half of its parent, etc."""
    x0, y0, x1, y1 = bbox
    if arrangement == "v":
        center = (y0 + y1) / 2
        return center < 500 if index == 0 else center > 400
    center = (x0 + x1) / 2
    return center < 560 if index == 0 else center > 460


def carve(parent, index, expected):
    """Our chunk's strokes within one parent, or None."""
    glyph = graphics.get(parent)
    if not glyph:
        return None
    total = len(glyph["strokes"])
    if not 0 < expected < total:
        return None
    # Preferred: mmah's own stroke->component matches, whenever mmah
    # also splits the parent in two at the top level.
    entry = dictionary.get(parent)
    if entry and entry.get("matches") and not any(m is None for m in entry["matches"]):
        if top_level_arity(entry.get("decomposition", "")) == 2:
            paths = [
                stroke
                for stroke, match in zip(glyph["strokes"], entry["matches"])
                if match[0] == index
            ]
            if len(paths) == expected:
                return paths
    # Fallback: a top/left chunk is written first, a bottom/right chunk
    # last; validate against the sibling's stroke count so interleaved
    # stroke orders get rejected instead of miscarved.
    sibling = decomposer.characters[parent]["components"][1 - index]
    sibling_count = stroke_count(sibling)
    if sibling_count is None or sibling_count + expected != total:
        return None
    return glyph["strokes"][:expected] if index == 0 else glyph["strokes"][-expected:]


def extract(number):
    """Carve our chunk's strokes out of the parent character where it
    draws largest. Returns (svg_dict, parent) or None."""
    entry = decomposer.characters[number]
    arrangement = arrangement_of(entry["decomposition_type"])
    if arrangement is None:
        return None
    parents = [
        (char, e["components"].index(number))
        for char, e in decomposer.characters.items()
        if not char.isdigit()
        and number in e["components"]
        and len(e["components"]) == 2
    ]
    expected = stroke_count(number)
    if not expected:
        # Some parts (𠂉, bare strokes) have no stroke data; derive the
        # count from the parents instead and take the majority verdict.
        counts = Counter()
        for parent, index in parents:
            glyph = graphics.get(parent)
            sibling = decomposer.characters[parent]["components"][1 - index]
            sibling_count = stroke_count(sibling)
            if glyph and sibling_count is not None and len(glyph["strokes"]) > sibling_count:
                counts[len(glyph["strokes"]) - sibling_count] += 1
        if not counts:
            return None
        expected = counts.most_common(1)[0][0]

    best = None
    for parent, index in parents:
        paths = carve(parent, index, expected)
        if not paths:
            continue
        bbox = path_bbox(paths)
        if not plausible_position(bbox, arrangement, index):
            continue
        x0, y0, x1, y1 = bbox
        area = (x1 - x0) * (y1 - y0)
        if best is None or area > best[0]:
            best = (area, parent, paths, bbox)
    if best is None:
        return None
    _, parent, paths, (x0, y0, x1, y1) = best
    # A square crop around the shape, so wide chunks render wide.
    side = max(x1 - x0, y1 - y0) * 1.12
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    viewbox = f"{cx - side / 2:.0f} {cy - side / 2:.0f} {side:.0f} {side:.0f}"
    return {"viewBox": viewbox, "paths": paths, "from": parent}


def main():
    candidates = sorted(
        number
        for number in decomposer.characters
        if number.isdigit()
        and _contains[number] >= NUMBERED_PRODUCTIVITY
        and _composite(number) is not None
        and _composite(number).parts
    )
    results = {}
    for number in candidates:
        svg = extract(number)
        if svg:
            results[number] = svg
    print(f"extracted {len(results)} of {len(candidates)} composite chunks")

    out = HERE / "chunk_svgs.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
