"""Build glyph_svgs.json.gz: stroke outlines for every piece the
decomposition quizzes can show.

Rendering only the codepoint-less chunks as SVG left them in a
different style (mmah's KaiTi outlines) than the font-rendered pieces
around them, so the quizzes draw *every* answer button from stroke
data instead. Full characters come straight from Make Me a Hanzi in
their natural em box. Components mmah lacks (𤴓, ⺙, 𠂉, ...) are
carved out of a parent character exactly like the numbered chunks in
build_chunk_svgs.py — 𤴓 is the last strokes of 是. Pieces neither
source can draw fall back to font text in the app.

Covers both distractor pools plus every chunk the current deck and
lexicon emit; new words outside that set fall back to font text until
a rebuild. Run: python data/chunks/build_glyph_svgs.py

Stroke data: Make Me a Hanzi (https://github.com/skishore/makemeahanzi),
derived from Arphic fonts under the Arphic Public License.
"""

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

import build_chunk_svgs as carving  # noqa: E402  (loads the mmah data)
from chunking import (  # noqa: E402
    character_pool,
    component_pool,
    decomposer,
    word_chunks,
    arrangement_of,
)

# Full characters keep their natural em box so they sit exactly like
# font glyphs; carved components get a tight crop like the chunks.
EM_VIEWBOX = "0 -124 1024 1024"


def needed_pieces():
    pieces = set(component_pool()) | set(character_pool())
    with open(HERE.parent.parent / "db.json", encoding="utf-8") as f:
        data = json.load(f)
    words = {r["word"] for r in data["_default"].values() if r.get("quiz_type")}
    words |= {e["word"] for e in data.get("lexicon", {}).values() if "word" in e}
    for word in words:
        for chunk in word_chunks(word):
            if chunk.text:
                pieces.add(chunk.text)
    return pieces


def carve_glyph(piece):
    """Carve a piece mmah lacks out of a parent character, like the
    numbered chunks; position within each parent comes from that
    parent's own split."""
    parents = [
        (char, entry["components"].index(piece), arrangement_of(entry["decomposition_type"]))
        for char, entry in decomposer.characters.items()
        if not char.isdigit()
        and piece in entry["components"]
        and len(entry["components"]) == 2
    ]
    parents = [p for p in parents if p[2] is not None]
    expected = carving.stroke_count(piece)
    if not expected:
        counts = Counter()
        for parent, index, _ in parents:
            glyph = carving.graphics.get(parent)
            sibling = decomposer.characters[parent]["components"][1 - index]
            sibling_count = carving.stroke_count(sibling)
            if glyph and sibling_count is not None and len(glyph["strokes"]) > sibling_count:
                counts[len(glyph["strokes"]) - sibling_count] += 1
        if not counts:
            return None
        expected = counts.most_common(1)[0][0]
    best = None
    for parent, index, arrangement in parents:
        paths = carving.carve(parent, index, expected)
        if not paths:
            continue
        bbox = carving.path_bbox(paths)
        if not carving.plausible_position(bbox, arrangement, index):
            continue
        x0, y0, x1, y1 = bbox
        area = (x1 - x0) * (y1 - y0)
        if best is None or area > best[0]:
            best = (area, paths, bbox)
    if best is None:
        return None
    _, paths, (x0, y0, x1, y1) = best
    side = max(x1 - x0, y1 - y0) * 1.12
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    viewbox = f"{cx - side / 2:.0f} {cy - side / 2:.0f} {side:.0f} {side:.0f}"
    return {"viewBox": viewbox, "paths": paths}


def main():
    pieces = needed_pieces()
    out = {}
    carved = missing = 0
    for piece in sorted(pieces):
        glyph = carving.graphics.get(piece)
        if glyph:
            out[piece] = {"viewBox": EM_VIEWBOX, "paths": glyph["strokes"]}
            continue
        svg = carve_glyph(piece)
        if svg:
            out[piece] = svg
            carved += 1
        else:
            missing += 1
    print(f"{len(out)} pieces ({carved} carved from parents), {missing} left to font text")

    path = HERE / "glyph_svgs.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
