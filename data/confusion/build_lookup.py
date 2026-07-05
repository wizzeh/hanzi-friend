"""Build confusables.json: top-K visually similar chars with z-scores.

Output schema: { "char": [["neighbor", z_score], ...], ... }
The z-score is per-row standardized cosine after mean-centering — see threshold.py.
The app filters at display time (default cutoff z >= 2.5).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
TOP_K = 10

d = np.load(HERE / "dinov2_embeddings.npz", allow_pickle=True)
chars: list[str] = list(d["chars"])
embs = d["embs"].astype(np.float32)

# Mean-center, re-normalize. This widens the cosine distribution from a
# collapsed [0.95, 0.99] band into a usable range.
embs_c = embs - embs.mean(axis=0, keepdims=True)
embs_c /= np.linalg.norm(embs_c, axis=1, keepdims=True) + 1e-9

sims = embs_c @ embs_c.T

# Per-row z-score: how many std above this char's mean similarity. Compute on
# the full matrix (the diagonal contributes a constant ~1.0 to every row, so
# z-scores are still well-defined), then mask self before ranking.
mu = sims.mean(axis=1, keepdims=True)
sd = sims.std(axis=1, keepdims=True) + 1e-9
z = (sims - mu) / sd
np.fill_diagonal(z, -np.inf)

top_idx = np.argsort(-z, axis=1)[:, :TOP_K]
top_z = np.take_along_axis(z, top_idx, axis=1)

lookup: dict[str, list[list]] = {}
for i, c in enumerate(chars):
    lookup[c] = [
        [chars[top_idx[i, k]], round(float(top_z[i, k]), 2)]
        for k in range(TOP_K)
    ]

out_path = HERE / "confusables.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(lookup, f, ensure_ascii=False, separators=(",", ":"))
print(f"wrote {out_path}  ({out_path.stat().st_size / 1024:.0f} KB, {len(lookup)} chars)")

# Quick stats.
above_25 = sum(1 for v in lookup.values() if v[0][1] >= 2.5)
above_30 = sum(1 for v in lookup.values() if v[0][1] >= 3.0)
print(f"chars with top-1 z >= 2.5: {above_25} ({100*above_25/len(lookup):.1f}%)")
print(f"chars with top-1 z >= 3.0: {above_30} ({100*above_30/len(lookup):.1f}%)")

# Sanity print.
print("\nsample:")
for c in "已戊口千林森":
    if c in lookup:
        nbrs = " ".join(f"{n}({s})" for n, s in lookup[c])
        print(f"  {c}: {nbrs}")
