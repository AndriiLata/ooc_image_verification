#!/usr/bin/env python3
"""
03_rank_evidence.py (VERITE_small version)

For each VERITE_small pair (image + caption), rank the external evidence
(text + images) by CLIP similarity and attach top-k evidence features.

Output:
  outputs/verite_small/ranked_verite_small_<tag>.csv
"""
import sys
from pathlib import Path

# Add repo root to PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import os
import yaml
import numpy as np
import pandas as pd

from ooclab.evidence.rank import rank_evidence

# ----------------- repo root & sys.path -----------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# ----------------- config & paths -----------------
cfg = yaml.safe_load(open(REPO_ROOT / "config" / "default.yaml"))
tag = cfg["encoder"]["version"].replace("-", "").replace("/", "")
outs_root = REPO_ROOT / cfg["paths"]["outputs"]

# 1) Load VERITE_small labels (id, image_id, falsified)
labels_csv = outs_root / "verite_small" / "verite_small_labels.csv"
if not labels_csv.exists():
    raise SystemExit(f"Labels CSV not found at: {labels_csv}")

df_verite = pd.read_csv(labels_csv)

df_verite["id"] = df_verite["id"].astype(str)
df_verite["image_id"] = df_verite["image_id"].astype(str)

# 2) Load query (claim) embeddings for VERITE_small
#    These were saved by 01_extract_verite_small.py via save_embeddings(...)
ids = np.load(outs_root / "verite_small" / f"verite_small_item_ids_{tag}.npy")
I   = np.load(outs_root / "verite_small" / f"verite_small_clip_image_embeddings_{tag}.npy")
T   = np.load(outs_root / "verite_small" / f"verite_small_clip_text_embeddings_{tag}.npy")

# I, T shapes are (n_items, dim). We want DataFrames where:
#   - columns are ids
#   - each column is a dim-long vector
ids_str = np.array(ids).astype(str)

img_map = pd.DataFrame(I.T, columns=ids_str)  # shape (dim, n_items)
txt_map = pd.DataFrame(T.T, columns=ids_str)  # shape (dim, n_items)


# 3) Load evidence embeddings (from 02_extract_evidence_features_small.py)
Xti = np.load(outs_root / "evidence" / f"X_text_ids_{tag}.npy", allow_pickle=True)
Xte = np.load(outs_root / "evidence" / f"X_text_embeddings_{tag}.npy")

Xii = np.load(outs_root / "evidence" / f"X_image_ids_{tag}.npy", allow_pickle=True)
Xie = np.load(outs_root / "evidence" / f"X_image_embeddings_{tag}.npy")


# Xte, Xie shapes are (n_evidence, dim). rank_evidence expects them transposed.
# So we pass Xie.T, Xte.T below, and ids as lists.
print(f"[info] Claim embeddings: images {I.shape}, texts {T.shape}")
print(f"[info] Evidence embeddings: images {Xie.shape}, texts {Xte.shape}")

# 4) Rank evidence for all VERITE_small rows
#    df_verite must have at least: id, image_id, falsified
print(f"[info] Ranking evidence for {len(df_verite)} VERITE_small items (topk=1)")

ranked = rank_evidence(
    df_verite,
    img_map,              # query image embeddings (dim x n_items, columns=ids)
    txt_map,              # query text embeddings (dim x n_items, columns=ids)
    Xie, list(Xii),       # evidence image embeddings: (n_evidence, dim)
    Xte, list(Xti),       # evidence text embeddings: (n_evidence, dim)
    topk=1
)


# 5) Merge ranking features back into the label dataframe
out = df_verite.merge(ranked, on=["id", "image_id"], how="left")

out_dir = outs_root / "verite_small"
out_dir.mkdir(parents=True, exist_ok=True)

out_csv = out_dir / f"ranked_verite_small_{tag}.csv"
out.to_csv(out_csv, index=False)

print(f"✓ Saved ranked VERITE_small with evidence features to: {out_csv}")
