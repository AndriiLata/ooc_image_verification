#!/usr/bin/env python3
# scripts/04_build_decision_features_verite_small.py

import sys, ast
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # assumes 1D vectors
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def parse_list_cell(x):
    """CSV cells look like "['0_true_1']" or "[]". Return python list."""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s:
        return []
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def main():
    cfg = yaml.safe_load(open(REPO_ROOT / "config" / "default.yaml"))
    tag = cfg["encoder"]["version"].replace("-", "").replace("/", "")
    outs_root = REPO_ROOT / cfg["paths"]["outputs"]

    # --------- paths ---------
    ranked_fp = outs_root / "verite_small" / f"ranked_verite_small_{tag}.csv"
    if not ranked_fp.exists():
        raise SystemExit(f"Missing ranked file: {ranked_fp} (run 03_rank_evidence.py first)")

    ids_fp = outs_root / "verite_small" / f"verite_small_item_ids_{tag}.npy"
    I_fp   = outs_root / "verite_small" / f"verite_small_clip_image_embeddings_{tag}.npy"
    T_fp   = outs_root / "verite_small" / f"verite_small_clip_text_embeddings_{tag}.npy"
    for p in [ids_fp, I_fp, T_fp]:
        if not p.exists():
            raise SystemExit(f"Missing claim embeddings: {p} (run 01_extract_verite.py first)")

    Xti_fp = outs_root / "evidence" / f"X_text_ids_{tag}.npy"
    Xte_fp = outs_root / "evidence" / f"X_text_embeddings_{tag}.npy"
    Xii_fp = outs_root / "evidence" / f"X_image_ids_{tag}.npy"
    Xie_fp = outs_root / "evidence" / f"X_image_embeddings_{tag}.npy"
    for p in [Xti_fp, Xte_fp, Xii_fp, Xie_fp]:
        if not p.exists():
            raise SystemExit(f"Missing evidence embeddings: {p} (run 02_exract_evidence_features.py first)")

    # --------- load ranked table ---------
    ranked = pd.read_csv(ranked_fp)

    # parse list cells
    ranked["img_ranked_items"] = ranked["img_ranked_items"].apply(parse_list_cell)
    ranked["txt_ranked_items"] = ranked["txt_ranked_items"].apply(parse_list_cell)

    # --------- load claim embeddings ---------
    ids = np.load(ids_fp, allow_pickle=True).astype(str)
    I = np.load(I_fp).astype(np.float32)
    T = np.load(T_fp).astype(np.float32)

    # normalize once
    I = l2norm(I)
    T = l2norm(T)

    # map: sample_id -> vector
    claim_img = {sid: I[i] for i, sid in enumerate(ids)}
    claim_txt = {sid: T[i] for i, sid in enumerate(ids)}

    # --------- load evidence embeddings ---------
    Xti = np.load(Xti_fp, allow_pickle=True).astype(str)
    Xte = np.load(Xte_fp).astype(np.float32)
    Xii = np.load(Xii_fp, allow_pickle=True).astype(str)
    Xie = np.load(Xie_fp).astype(np.float32)

    Xte = l2norm(Xte)
    Xie = l2norm(Xie)

    ev_txt = {eid: Xte[i] for i, eid in enumerate(Xti)}
    ev_img = {eid: Xie[i] for i, eid in enumerate(Xii)}

    # --------- build decision features ---------
    rows = []
    for _, r in ranked.iterrows():
        sid = str(r["id"])

        # claim vectors
        vi = claim_img.get(sid, None)
        vt = claim_txt.get(sid, None)
        if vi is None or vt is None:
            # skip if embedding missing
            continue

        # top evidence ids (top-1)
        top_img_eid = r["img_ranked_items"][0] if len(r["img_ranked_items"]) else None
        top_txt_eid = r["txt_ranked_items"][0] if len(r["txt_ranked_items"]) else None

        # sims that don't need evidence
        sim_img_txt = float(np.dot(vi, vt))  # cosine because normalized

        # sims with evidence (default NaN if missing)
        sim_img_top_txt = np.nan
        sim_txt_top_img = np.nan
        sim_txt_top_txt = np.nan
        sim_img_top_img = np.nan

        if top_txt_eid and top_txt_eid in ev_txt:
            sim_img_top_txt = float(np.dot(vi, ev_txt[top_txt_eid]))
            sim_txt_top_txt = float(np.dot(vt, ev_txt[top_txt_eid]))

        if top_img_eid and top_img_eid in ev_img:
            sim_txt_top_img = float(np.dot(vt, ev_img[top_img_eid]))
            sim_img_top_img = float(np.dot(vi, ev_img[top_img_eid]))

        # gap features (evidence minus claim consistency)
        gap_img_eviTxt = (sim_img_top_txt - sim_img_txt) if not np.isnan(sim_img_top_txt) else np.nan
        gap_txt_eviImg = (sim_txt_top_img - sim_img_txt) if not np.isnan(sim_txt_top_img) else np.nan
        gap_txt_eviTxt = (sim_txt_top_txt - sim_img_txt) if not np.isnan(sim_txt_top_txt) else np.nan
        gap_img_eviImg = (sim_img_top_img - sim_img_txt) if not np.isnan(sim_img_top_img) else np.nan

        rows.append({
            "id": sid,
            "falsified": int(r["falsified"]),
            "base_id": r.get("base_id", ""),
            "label": r.get("label", ""),

            "top_evi_img_id": top_img_eid or "",
            "top_evi_txt_id": top_txt_eid or "",

            # core sims
            "sim_img_txt": sim_img_txt,
            "sim_img_top_txt": sim_img_top_txt,
            "sim_txt_top_img": sim_txt_top_img,
            "sim_txt_top_txt": sim_txt_top_txt,
            "sim_img_top_img": sim_img_top_img,

            # gaps
            "gap_img_eviTxt": gap_img_eviTxt,
            "gap_txt_eviImg": gap_txt_eviImg,
            "gap_txt_eviTxt": gap_txt_eviTxt,
            "gap_img_eviImg": gap_img_eviImg,
        })

    feat = pd.DataFrame(rows)

    out_fp = outs_root / "verite_small" / f"decision_features_verite_small_{tag}.csv"
    feat.to_csv(out_fp, index=False)
    print(f"✓ Wrote decision feature table: {out_fp}")
    print("Columns:", ", ".join(feat.columns))


if __name__ == "__main__":
    main()
