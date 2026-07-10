"""Salience-based character chunking for the decomposition quizzes.

The decomposition database recurses every character down to Kangxi
radicals, which turns quiz answers into stroke soup (蓝 becomes
艹丨丨丿一丶皿) and blows through the chunks a learner actually
recognizes (每 in 海, 京 in 惊). These quizzes exist to train visual
parsing, not stroke-level etymology, so we stop recursion at any piece
the learner can be expected to recognize: a unit from the Loach-Wang
learning order, a Kangxi radical, or a piece that recurs across enough
characters to be worth knowing on its own.

The database names some chunks by number because they have no Unicode
codepoint (the top of 学 is ⺍ over 冖). Recurring numbered chunks that
stack vertically or sit side by side are kept as composite chunks whose
parts the template renders with CSS to suggest the real shape; the rest
are recursed through so their visual material still shows up as
individual pieces.
"""

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

from hanzipy.decomposer import HanziDecomposer

from loach_word_order import word_order
from radicals import radicals as all_radicals

decomposer = HanziDecomposer()


class Chunk(NamedTuple):
    text: str  # the glyph, or "" for codepoint-less composites
    parts: Tuple[str, ...] = ()
    arrangement: str = ""  # "v" (stacked) or "h" (side by side)
    # The true shape carved from a real character's stroke data (see
    # data/chunks/build_chunk_svgs.py); empty when we could only carve
    # ambiguously and the template should stack the parts with CSS.
    svg_viewbox: str = ""
    svg_paths: Tuple[str, ...] = ()


_svgs_path = Path(__file__).parent / "data" / "chunks" / "chunk_svgs.json"
_chunk_svgs = (
    json.loads(_svgs_path.read_text(encoding="utf-8")) if _svgs_path.exists() else {}
)

# Stroke outlines for whole glyphs, so every quiz answer renders in the
# same style as the carved chunks (data/chunks/build_glyph_svgs.py).
# Loaded on first use: ~15 MB of JSON we don't want at import time.
_glyph_svgs_path = Path(__file__).parent / "data" / "chunks" / "glyph_svgs.json.gz"
_glyph_svgs = None


def glyph_svg(piece: str) -> Optional[dict]:
    """{"viewBox", "paths"} for our piece, or None if only the font
    can draw it."""
    global _glyph_svgs
    if _glyph_svgs is None:
        if _glyph_svgs_path.exists():
            with gzip.open(_glyph_svgs_path, "rt", encoding="utf-8") as f:
                _glyph_svgs = json.load(f)
        else:
            _glyph_svgs = {}
    return _glyph_svgs.get(piece)


# Pieces the Loach-Wang order treats as learnable units in their own
# right (it schedules components like 𤴓 before the characters that
# contain them), so stopping at them leaves the learner on familiar
# ground.
LOACH_CHARS = frozenset(w for w in word_order if len(w) == 1)
RADICALS = frozenset(all_radicals)

# hanzipy's stroke set plus the CJK Strokes block (㇒, ㇗, ...). Never
# worth descending below, and a split that yields nothing else isn't
# helping anyone parse the character.
PLAIN_STROKES = frozenset("一丨丶⺀丿乙⺃乚⺄亅丷乛")

# How many characters must contain a piece before we treat it as a
# recognizable chunk in its own right (glyphs get the easier tests
# above first; numbered chunks only have this).
GLYPH_PRODUCTIVITY = 4
NUMBERED_PRODUCTIVITY = 2

_MAX_DEPTH = 6

# Direct-containment counts over the whole database: how many entries
# list this piece as an immediate component.
_contains = Counter(
    component
    for entry in decomposer.characters.values()
    for component in entry["components"]
)


def _subcomponents(piece: str) -> List[str]:
    entry = decomposer.characters.get(piece)
    if not entry:
        return []
    return [c for c in entry["components"] if c != piece]


def _is_stroke(piece: str) -> bool:
    return len(piece) == 1 and (
        piece in PLAIN_STROKES or 0x31C0 <= ord(piece) <= 0x31EF
    )


def _renderable(piece: str) -> bool:
    # The database maps a few strokes into the Private Use Area, which
    # renders as tofu; those can't be shown to the learner.
    return all(not 0xE000 <= ord(ch) <= 0xF8FF for ch in piece)


def _salient(piece: str) -> bool:
    if _is_stroke(piece):
        return False
    return (
        piece in LOACH_CHARS
        or piece in RADICALS
        or _contains[piece] >= GLYPH_PRODUCTIVITY
    )


def _composite(number: str) -> Chunk:
    """A renderable composite for a numbered chunk, or None when its
    arrangement can't be suggested by stacking glyphs."""
    entry = decomposer.characters.get(number)
    if not entry:
        return None
    arrangement = arrangement_of(entry["decomposition_type"])
    parts = []
    for component in entry["components"]:
        if component.isdigit():
            nested = _composite(component)
            if nested is None or (nested.parts and nested.arrangement != arrangement):
                return None
            parts.extend(nested.parts or [nested.text])
        elif _renderable(component):
            parts.append(component)
        else:
            return None
    if len(parts) == 1:
        # A one-part entry is a stretched or repositioned variant of
        # that part; the part alone is the best rendering we have.
        return Chunk(text=parts[0])
    if arrangement is None:
        return None  # surrounds, locks and overlaps don't stack
    svg = _chunk_svgs.get(number, {})
    return Chunk(
        text="",
        parts=tuple(parts),
        arrangement=arrangement,
        svg_viewbox=svg.get("viewBox", ""),
        svg_paths=tuple(svg.get("paths", ())),
    )


def arrangement_of(kind: str) -> str:
    if kind.startswith("d"):
        return "v"
    if kind.startswith("a"):
        return "h"
    return None  # surrounds, locks and overlaps don't stack


def _expand(piece: str, depth: int) -> List[Tuple[Chunk, bool]]:
    """Expand one piece into (chunk, is_salient) pairs."""
    if piece.isdigit():
        if _contains[piece] >= NUMBERED_PRODUCTIVITY:
            composite = _composite(piece)
            if composite is not None:
                return [(composite, True)]
        return [
            expanded
            for sub in _subcomponents(piece)
            for expanded in _expand(sub, depth + 1)
        ]
    if _is_stroke(piece) or _salient(piece) or depth >= _MAX_DEPTH:
        return [(Chunk(text=piece), _salient(piece))] if _renderable(piece) else []
    subs = _subcomponents(piece)
    if not subs:
        return [(Chunk(text=piece), False)] if _renderable(piece) else []
    expanded = [pair for sub in subs for pair in _expand(sub, depth + 1)]
    # Descending bought us nothing but strokes: the piece itself is the
    # more recognizable unit, even if nothing vouches for it.
    if _renderable(piece) and all(
        chunk.text and _is_stroke(chunk.text) for chunk, _ in expanded
    ):
        return [(Chunk(text=piece), False)]
    return expanded


def word_chunks(word: str) -> List[Chunk]:
    """The recognizable chunks of our word: its characters when there
    are several (decomposing those re-tests what each character's own
    quiz already covers), else one forced split, deduplicated in
    reading order."""
    if len(word) > 1:
        chunks = []
        for char in word:
            chunk = Chunk(text=char)
            if chunk not in chunks:
                chunks.append(chunk)
        return chunks
    chunks = []
    for char in word:
        pieces = _subcomponents(char)
        if not pieces:
            expanded = [(Chunk(text=char), _salient(char))]
        else:
            expanded = [pair for piece in pieces for pair in _expand(piece, 1)]
            # A split in which the learner recognizes nothing (心 into
            # 𠁼 and a hooked stroke) is worse than no split.
            if expanded and not any(salient for _, salient in expanded):
                expanded = [(Chunk(text=char), _salient(char))]
        for chunk, _ in expanded:
            if chunk not in chunks:
                chunks.append(chunk)
    return chunks


def character_pool() -> List[str]:
    """Characters worth offering as distractors for multi-character
    words: everything the learning order treats as a unit."""
    return sorted(c for c in LOACH_CHARS if ord(c) >= 0x2E80)


def component_pool() -> List[str]:
    """Glyphs worth offering as quiz distractors: recognizable pieces
    that actually occur as components."""
    pool = set(RADICALS)
    pool.update(
        piece
        for piece, count in _contains.items()
        if count >= NUMBERED_PRODUCTIVITY
        and not piece.isdigit()
        and not _is_stroke(piece)
        and _renderable(piece)
        and _salient(piece)
    )
    return sorted(pool)
