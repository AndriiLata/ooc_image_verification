# scripts/03_rank_evidence.py
#!/usr/bin/env python3
"""
03_rank_evidence.py (VERITE_small)

Loads:
- outputs/verite_small/verite_small_labels.csv (id, image_id, falsified)
- outputs/verite_small/verite_small_*embeddings*.npy
- outputs/evidence/X_*_<tag>.npy

Ranks evidence by CLIP similarity via ooclab.evidence.rank.rank_evidence
and saves:
- outputs/verite_small/ranked_verite_small_<tag>.csv
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import yaml
import numpy as np
import pandas as pd
from ooclab.evidence.rank import rank_evidence


def main():
    cfg = yaml.safe_load(open(REPO_ROOT / "config" / "default.yaml"))
    tag = cfg["encoder"]["version"].replace("-", "").replace("/", "")
    outs_root = REPO_ROOT / cfg["paths"]["outputs"]

    # 1) labels
    labels_csv = outs_root / "verite_small" / "verite_small_labels.csv"
    if not labels_csv.exists():
        raise SystemExit(f"Labels CSV not found at: {labels_csv}")

    df_verite = pd.read_csv(labels_csv)
    for c in ["id", "image_id", "falsified"]:
        if c not in df_verite.columns:
            raise SystemExit(f"labels file missing column '{c}': {labels_csv}")

    df_verite["id"] = df_verite["id"].astype(str)
    df_verite["image_id"] = df_verite["image_id"].astype(str)

    # 2) claim embeddings
    ids_fp = outs_root / "verite_small" / f"verite_small_item_ids_{tag}.npy"
    I_fp   = outs_root / "verite_small" / f"verite_small_clip_image_embeddings_{tag}.npy"
    T_fp   = outs_root / "verite_small" / f"verite_small_clip_text_embeddings_{tag}.npy"

    if not (ids_fp.exists() and I_fp.exists() and T_fp.exists()):
        raise SystemExit("Claim embedding files not found. Run 01_extract_verite.py first.")

    ids = np.load(ids_fp, allow_pickle=True)
    I   = np.load(I_fp)
    T   = np.load(T_fp)

    ids_str = np.array(ids).astype(str)

    # map format: dim x n_items, columns are ids
    img_map = pd.DataFrame(I.T, columns=ids_str)
    txt_map = pd.DataFrame(T.T, columns=ids_str)

    # 3) evidence embeddings
    Xti_fp = outs_root / "evidence" / f"X_text_ids_{tag}.npy"
    Xte_fp = outs_root / "evidence" / f"X_text_embeddings_{tag}.npy"
    Xii_fp = outs_root / "evidence" / f"X_image_ids_{tag}.npy"
    Xie_fp = outs_root / "evidence" / f"X_image_embeddings_{tag}.npy"

    if not (Xti_fp.exists() and Xte_fp.exists() and Xii_fp.exists() and Xie_fp.exists()):
        raise SystemExit("Evidence embedding files not found. Run 02_extract_evidence_features.py first.")

    Xti = np.load(Xti_fp, allow_pickle=True)
    Xte = np.load(Xte_fp)
    Xii = np.load(Xii_fp, allow_pickle=True)
    Xie = np.load(Xie_fp)

    print(f"[info] Claim embeddings: images {I.shape}, texts {T.shape}")
    print(f"[info] Evidence embeddings: images {Xie.shape}, texts {Xte.shape}")

    # 4) sanity check: do ids match evidence prefixes?
    # Evidence ids look like "<verite_id>_<k>"
    evidence_prefixes = set(str(x).split("_")[0] for x in list(Xti)[:200] + list(Xii)[:200])
    sample_ids = set(df_verite["id"].astype(str).head(50))
    overlap = len(sample_ids.intersection(evidence_prefixes))
    if overlap == 0:
        print("[warn] ZERO overlap between claim ids and evidence prefixes. "
              "Your VERITE_small ids still don't match VERITE_articles_small ids.")
    else:
        print(f"[info] id/evidence overlap check: {overlap}/50 sample ids found in evidence prefixes (good sign).")

    # 5) rank
    ranked = rank_evidence(
        df_verite,
        img_map,
        txt_map,
        Xie, list(Xii),
        Xte, list(Xti),
        topk=1,
    )

    out = df_verite.merge(ranked, on=["id", "image_id"], how="left")

    out_dir = outs_root / "verite_small"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"ranked_verite_small_{tag}.csv"
    out.to_csv(out_csv, index=False)

    print(f"✓ Saved ranked VERITE_small with evidence features to: {out_csv}")


if __name__ == "__main__":
    main()
