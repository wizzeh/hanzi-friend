"""Explore similarity-floor thresholds for the DINOv2 NN lookup.

The raw CLS-token cosine is collapsed (everyone is ~0.97 cosine to everyone),
so we test two reshapings: mean-centering the embeddings, and per-row z-scores
of the cosine matrix. Both give a more separable distribution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
d = np.load(HERE / "dinov2_embeddings.npz", allow_pickle=True)
chars = list(d["chars"])
embs_raw = d["embs"].astype(np.float32)
idx_of = {c: i for i, c in enumerate(chars)}
print(f"{len(chars)} chars, dim {embs_raw.shape[1]}")


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


# Three views of the embedding space.
embs_norm = normalize(embs_raw)
embs_centered = normalize(embs_raw - embs_raw.mean(axis=0, keepdims=True))


def report(name: str, embs: np.ndarray) -> np.ndarray:
    sims = embs @ embs.T
    np.fill_diagonal(sims, -1.0)
    nn1 = sims.max(axis=1)
    print(f"\n[{name}]  NN1 cosine: "
          f"p5={np.quantile(nn1,.05):.3f} "
          f"p25={np.quantile(nn1,.25):.3f} "
          f"p50={np.quantile(nn1,.50):.3f} "
          f"p75={np.quantile(nn1,.75):.3f} "
          f"p95={np.quantile(nn1,.95):.3f} "
          f"min={nn1.min():.3f}")
    return sims


sims_raw = report("raw L2-normed", embs_norm)
sims_ctr = report("mean-centered then L2-normed", embs_centered)

# Per-row z-score on the centered sims (most separable).
mu = sims_ctr.mean(axis=1, keepdims=True)
sd = sims_ctr.std(axis=1, keepdims=True) + 1e-9
sims_z = (sims_ctr - mu) / sd
np.fill_diagonal(sims_z, -np.inf)
print(f"\n[z-score on centered]  NN1 z: "
      f"p5={np.quantile(sims_z.max(1),.05):.2f} "
      f"p50={np.quantile(sims_z.max(1),.50):.2f} "
      f"p95={np.quantile(sims_z.max(1),.95):.2f} "
      f"max={sims_z.max():.2f}")


def show_spot(label: str, sims: np.ndarray, fmt: str) -> None:
    print(f"\n=== Spot checks [{label}] ===")
    for c in "已己巳日未末抹戊戌戍土士千干口木林森人入大太六会请":
        if c not in idx_of:
            continue
        i = idx_of[c]
        s = sims[i]
        nn = np.argsort(-s)[:10]
        pretty = " ".join(f"{chars[j]}({s[j]:{fmt}})" for j in nn)
        print(f"  {c}: {pretty}")


show_spot("centered cosine", sims_ctr, ".2f")
show_spot("per-row z-score", sims_z, ".1f")


# Threshold sweep on each view.
def sweep(name: str, sims: np.ndarray, thresholds) -> None:
    print(f"\nThreshold sweep [{name}]:")
    print(f"  {'thr':>6}  {'%chars w/≥1':>12}  {'avg n':>7}  {'med n':>6}  {'p90 n':>6}  {'max n':>6}")
    K = 50
    order = np.argsort(-sims, axis=1)[:, :K]
    top_sims = np.take_along_axis(sims, order, axis=1)
    for thr in thresholds:
        counts = (top_sims >= thr).sum(axis=1)
        has_any = (counts > 0).mean() * 100
        avg_n = counts.mean()
        med_n = np.median(counts)
        p90_n = np.quantile(counts, 0.9)
        max_n = counts.max()
        print(f"  {thr:>6.2f}  {has_any:>11.1f}%  {avg_n:>7.2f}  {med_n:>6.0f}  {p90_n:>6.0f}  {max_n:>6}")


sweep("centered cosine", sims_ctr, (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90))
sweep("per-row z-score", sims_z, (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0))


# Which view best matches same_stroke.txt?
truth: dict[str, set[str]] = {}
with open(HERE / "same_stroke.txt", encoding="utf-8") as f:
    for line in f:
        grp = [c for c in line.strip().split("\t") if c]
        for cc in grp:
            truth.setdefault(cc, set()).update(d for d in grp if d != cc)

qchars = [c for c in chars if truth.get(c) and (truth[c] & set(chars))]
print(f"\nGT eval on {len(qchars)} chars (recall@10, no threshold):")
for name, sims in [("raw", sims_raw), ("centered", sims_ctr), ("z-score", sims_z)]:
    recs = []
    for c in qchars:
        i = idx_of[c]
        nn = np.argsort(-sims[i])[:10]
        gt = truth[c] & set(chars)
        preds = {chars[j] for j in nn}
        recs.append(len(preds & gt) / len(gt))
    print(f"  {name:>10}: recall@10 = {np.mean(recs):.3f}")
