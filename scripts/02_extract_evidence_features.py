#!/usr/bin/env python3
"""
02_extract_evidence_features_small.py

Encode evidence captions + images (from collected_evidence_small.csv)
with CLIP and save them as numpy arrays.

Input CSV: data/evidence/collected_evidence_small.csv
Columns:
  - match_index: VERITE id (int)
  - captions: JSON list of strings
  - images_paths: JSON list of relative paths (e.g. "data/evidence/000000/...jpg")
"""

import sys
from pathlib import Path

# Add repo root to PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import os
import ast
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
from PIL import Image

from ooclab.encoders.clip_encoder import ClipEncoder

# ----------------- paths & config -----------------
REPO_ROOT = Path(__file__).resolve().parents[1]

cfg = yaml.safe_load(open(REPO_ROOT / "config" / "default.yaml"))
encv = cfg["encoder"]["version"]
dev = cfg["encoder"]["device"]

# evidence root is the same logic as in 00_collect_evidence.py
evidence_root_cfg = (
    cfg["paths"].get("evidence_root")
    or cfg["paths"].get("evidence_out")
    or "data/evidence"
)
evidence_root = (REPO_ROOT / evidence_root_cfg).resolve()
evi_csv = evidence_root / "collected_evidence_small.csv"

if not evi_csv.exists():
    raise SystemExit(f"Evidence CSV not found at: {evi_csv}")

df = pd.read_csv(evi_csv)

# parse stringified lists safely
# (captions and images_paths are JSON-like arrays as strings)
df["captions"] = df["captions"].fillna("[]").apply(ast.literal_eval)
df["images_paths"] = df["images_paths"].fillna("[]").apply(ast.literal_eval)

# ----------------- explode to caption-level / image-level -----------------
cap_rows, img_rows = [], []
for _, r in df.iterrows():
    idx = str(r["match_index"])
    caps = r["captions"] or []
    imgs = r["images_paths"] or []

    for i, c in enumerate(caps):
        cap_rows.append((f"{idx}_{i}", str(c)))

    for i, p in enumerate(imgs):
        img_rows.append((f"{idx}_{i}", str(p)))

cap_df = pd.DataFrame(cap_rows, columns=["X_id", "caption"])
img_df = pd.DataFrame(img_rows, columns=["X_id", "image_path"])

print(f"Text evidence items: {len(cap_df)}")
print(f"Image evidence items: {len(img_df)}")

enc = ClipEncoder(encv, dev)

# ----------------- encode texts -----------------
text_chunks, ids_t = [], []
if len(cap_df):
    for i in range(0, len(cap_df), 256):
        chunk = cap_df.iloc[i:i+256]
        ids_t.extend(chunk["X_id"].tolist())
        text_chunks.append(enc.encode_texts(chunk["caption"].tolist()).cpu().numpy())
    Xtxt = np.vstack(text_chunks)
else:
    Xtxt = np.zeros((0, enc.dim), dtype="float32")  # in case of empty

# ----------------- encode images -----------------
img_chunks, ids_i = [], []
if len(img_df):
    for i in range(0, len(img_df), 256):
        chunk = img_df.iloc[i:i+256]
        batch_ids = []
        pil_images = []

        for x_id, rel_path in zip(chunk["X_id"].tolist(), chunk["image_path"].tolist()):
            img_path = (REPO_ROOT / rel_path).resolve()
            if not img_path.exists():
                print(f"[warn] image not found, skipping: {img_path}")
                continue
            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[warn] failed to open image {img_path}: {e}")
                continue
            batch_ids.append(x_id)
            pil_images.append(pil)

        if not pil_images:
            continue

        emb = enc.encode_images(pil_images).cpu().numpy()
        ids_i.extend(batch_ids)
        img_chunks.append(emb)

    if img_chunks:
        Ximg = np.vstack(img_chunks)
    else:
        Ximg = np.zeros((0, enc.dim), dtype="float32")
else:
    Ximg = np.zeros((0, enc.dim), dtype="float32")

# ----------------- save outputs -----------------
tag = encv.replace("-", "").replace("/", "")
outdir = REPO_ROOT / "outputs" / "evidence"
outdir.mkdir(parents=True, exist_ok=True)

np.save(outdir / f"X_text_embeddings_{tag}.npy", Xtxt.astype("float32"))
np.save(outdir / f"X_text_ids_{tag}.npy", np.array(ids_t, dtype=object))

np.save(outdir / f"X_image_embeddings_{tag}.npy", Ximg.astype("float32"))
np.save(outdir / f"X_image_ids_{tag}.npy", np.array(ids_i, dtype=object))

print(f"✓ Saved evidence embeddings to {outdir}")
