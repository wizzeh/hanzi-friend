"""Embed hanzi glyphs with DINOv2-small and evaluate against same_stroke.txt."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModel

HERE = Path(__file__).parent
PROJECT = HERE.parent.parent

# Locate a Simplified Chinese font face. Prefer Source Han Sans SC (standard CJK
# reference); fall back to the standalone Resource Han Rounded SC.
def find_font(size: int) -> ImageFont.FreeTypeFont:
    han_ttc = "/nix/store/ndv62zspai2l8paa6lzzznrbq05vmcnh-home-manager-path/share/fonts/opentype/source-han-sans/SourceHanSans.ttc"
    if os.path.exists(han_ttc):
        for idx in range(40):
            try:
                f = ImageFont.truetype(han_ttc, size, index=idx)
                name = " ".join(f.getname())
                if "SC" in name and "HW" not in name and "Regular" in name:
                    print(f"  font: {name}  (ttc index {idx})")
                    return f
            except Exception:
                break
    fallback = "/nix/store/ndv62zspai2l8paa6lzzznrbq05vmcnh-home-manager-path/share/fonts/truetype/ResourceHanRoundedSC-Regular.ttf"
    f = ImageFont.truetype(fallback, size)
    print(f"  font: {' '.join(f.getname())}  (fallback)")
    return f


def render(ch: str, font: ImageFont.FreeTypeFont, canvas: int = 224) -> Image.Image:
    img = Image.new("RGB", (canvas, canvas), "white")
    draw = ImageDraw.Draw(img)
    # Center the glyph in the canvas using its bbox.
    l, t, r, b = draw.textbbox((0, 0), ch, font=font)
    x = (canvas - (r - l)) / 2 - l
    y = (canvas - (b - t)) / 2 - t
    draw.text((x, y), ch, font=font, fill="black")
    return img


def load_char_set() -> list[str]:
    """Use the Loach order list as the primary char set; union same_stroke.txt
    so every ground-truth char is also in the search space. Drop any char the
    font cannot render (typically rare Extension B+ radicals)."""
    sys.path.insert(0, str(PROJECT))
    from loach_word_order import word_order  # type: ignore

    # word_order contains both single chars and multi-char words; keep only
    # single-codepoint hanzi (BMP + supplementary). Multi-char words would
    # otherwise pollute NN lists for their constituent chars.
    chars: set[str] = {w for w in word_order if len(w) == 1}
    with open(HERE / "same_stroke.txt", encoding="utf-8") as f:
        for line in f:
            for c in line.strip().split("\t"):
                if c and len(c) == 1:
                    chars.add(c)

    font = find_font(64)
    keep = []
    for c in sorted(chars):
        if font.getmask(c).getbbox() is not None:
            keep.append(c)
    print(f"  filtered out {len(chars) - len(keep)} unrenderable chars")
    return keep


def embed_all(chars: list[str], batch: int = 32) -> np.ndarray:
    print(f"  loading dinov2-small...")
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  device: {device}")

    font = find_font(180)

    embs = np.empty((len(chars), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(chars), batch):
            chunk = chars[i : i + batch]
            imgs = [render(c, font) for c in chunk]
            x = proc(images=imgs, return_tensors="pt").to(device)
            out = model(**x).last_hidden_state[:, 0]  # CLS
            embs[i : i + batch] = out.cpu().numpy()
            if i % (batch * 10) == 0:
                rate = (i + batch) / max(time.time() - t0, 1e-3)
                eta = (len(chars) - i - batch) / max(rate, 1e-3)
                print(f"    {i+batch}/{len(chars)}  {rate:.1f} chars/s  eta {eta:.0f}s")
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    return embs


def evaluate(chars: list[str], embs: np.ndarray) -> None:
    idx_of = {c: i for i, c in enumerate(chars)}

    # Ground-truth groups from same_stroke.txt (each group is a clique).
    truth: dict[str, set[str]] = {}
    with open(HERE / "same_stroke.txt", encoding="utf-8") as f:
        for line in f:
            grp = [c for c in line.strip().split("\t") if c]
            for c in grp:
                truth.setdefault(c, set()).update(d for d in grp if d != c)

    # Compute NN for chars that have ground truth and are in our embedding set.
    queries = [c for c in chars if c in truth and truth[c]]
    qi = np.array([idx_of[c] for c in queries])
    print(f"  evaluating {len(queries)} chars with ground truth")

    # Cosine = dot product since embs are normalized.
    sims = embs[qi] @ embs.T
    np.fill_diagonal(sims[:, qi[0] : qi[0] + 1], -1)  # we'll mask self per row below
    for i, q in enumerate(qi):
        sims[i, q] = -1.0
    top = np.argsort(-sims, axis=1)

    for k in (1, 5, 10, 20):
        recalls = []
        for row, c in enumerate(queries):
            gt = truth[c] & set(chars)
            if not gt:
                continue
            preds = set(chars[j] for j in top[row, :k])
            recalls.append(len(preds & gt) / len(gt))
        print(f"  recall@{k}: {np.mean(recalls):.3f}  (n={len(recalls)})")

    print("\n  spot checks (top 10 NN, ground-truth confusables in [brackets]):")
    for c in "已己巳日曰未末抹戊戌戍土士千干于口囗":
        if c not in idx_of:
            print(f"    {c}: not in embedding set")
            continue
        i = idx_of[c]
        gt = truth.get(c, set())
        s = embs[i] @ embs.T
        s[i] = -1
        nn = np.argsort(-s)[:10]
        annotated = " ".join(
            f"[{chars[j]}]" if chars[j] in gt else chars[j] for j in nn
        )
        gt_str = "".join(sorted(gt)) if gt else "—"
        print(f"    {c}  gt={gt_str:<10}  →  {annotated}")


def main() -> None:
    chars = load_char_set()
    print(f"char set: {len(chars)} chars")
    cache_path = HERE / "dinov2_embeddings.npz"
    if cache_path.exists() and "--rebuild" not in sys.argv:
        print(f"loading cache: {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        cached_chars = list(d["chars"])
        if cached_chars == chars:
            embs = d["embs"]
        else:
            print("  cache char set differs; rebuilding")
            embs = embed_all(chars)
            np.savez_compressed(cache_path, chars=np.array(chars), embs=embs)
    else:
        embs = embed_all(chars)
        np.savez_compressed(cache_path, chars=np.array(chars), embs=embs)
    print(f"embeddings shape: {embs.shape}")
    evaluate(chars, embs)


if __name__ == "__main__":
    main()
