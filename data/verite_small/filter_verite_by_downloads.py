#!/usr/bin/env python3
"""
filter_verite_keep_only_complete_pairs_reindex.py

Reindexing version.

Inputs:
  - VERITE_articles.csv  (must have column 'id')
  - VERITE.csv           (must have columns: caption, image_path, label; first column may be unnamed)
  - images/true_<id>.jpg and images/false_<id>.jpg

Rules:
- Keep base id ONLY if BOTH true_<id>.jpg and false_<id>.jpg exist.
- Drop the entire base id if either one is missing.
- Reindex kept base ids to 0..N-1.
- Update both CSVs + rewrite images/ folder accordingly.
"""

import re
import shutil
from pathlib import Path
import pandas as pd

# ---------------- settings ----------------
IMAGES_DIR = Path("images")

ARTICLES_IN = Path("VERITE_articles.csv")
VERITE_IN   = Path("VERITE.csv")

ARTICLES_OUT = Path("VERITE_articles_filtered.csv")
VERITE_OUT   = Path("VERITE_filtered.csv")

BACKUP_DIR = Path("images_backup_before_filter")

# ---------------- helpers ----------------
TRUE_RE = re.compile(r"^images/true_(\d+)\.jpg$")
FALSE_RE = re.compile(r"^images/false_(\d+)\.jpg$")

def normalize_int(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None

def has_both_images(base_id: int) -> bool:
    return (IMAGES_DIR / f"true_{base_id}.jpg").exists() and (IMAGES_DIR / f"false_{base_id}.jpg").exists()

def base_id_from_image_path(p: str) -> int:
    p = str(p).strip()
    m = TRUE_RE.match(p) or FALSE_RE.match(p)
    if not m:
        raise ValueError(f"Unexpected image_path format: {p}")
    return int(m.group(1))

def is_true_path(p: str) -> bool:
    return str(p).strip().startswith("images/true_")

def is_false_path(p: str) -> bool:
    return str(p).strip().startswith("images/false_")

def remap_path(p: str, new_id: int) -> str:
    if is_true_path(p):
        return f"images/true_{new_id}.jpg"
    if is_false_path(p):
        return f"images/false_{new_id}.jpg"
    raise ValueError(f"Unexpected image_path: {p}")

# ---------------- main ----------------
def main():
    if not IMAGES_DIR.exists():
        raise SystemExit(f"Missing images folder: {IMAGES_DIR.resolve()}")
    if not ARTICLES_IN.exists():
        raise SystemExit(f"Missing: {ARTICLES_IN.resolve()}")
    if not VERITE_IN.exists():
        raise SystemExit(f"Missing: {VERITE_IN.resolve()}")

    # ---------- Load VERITE_articles.csv ----------
    art = pd.read_csv(ARTICLES_IN)

    if "id" not in art.columns:
        raise SystemExit("VERITE_articles.csv must have a column named 'id'")

    art["id_norm"] = art["id"].apply(normalize_int)
    art = art.dropna(subset=["id_norm"]).copy()
    art["id_norm"] = art["id_norm"].astype(int)

    base_ids = sorted(set(art["id_norm"].tolist()))
    valid_ids = [i for i in base_ids if has_both_images(i)]
    valid_set = set(valid_ids)

    print(f"[info] base ids in articles: {len(base_ids)}")
    print(f"[info] valid ids (both images exist): {len(valid_ids)}")
    print(f"[info] dropped ids (missing true or false): {len(base_ids) - len(valid_ids)}")

    # Filter articles
    art_f = art[art["id_norm"].isin(valid_set)].copy()
    art_f["old_id"] = art_f["id_norm"].astype(int)

    # ---------- Load VERITE.csv ----------
    ver = pd.read_csv(VERITE_IN)

    # Rename first col if unnamed
    first_col = ver.columns[0]
    if first_col == "" or str(first_col).startswith("Unnamed"):
        ver = ver.rename(columns={first_col: "row_id"})

    required = {"caption", "image_path", "label"}
    missing = required - set(ver.columns)
    if missing:
        raise SystemExit(f"VERITE.csv missing columns: {sorted(missing)}")

    ver["old_base_id"] = ver["image_path"].apply(base_id_from_image_path)
    ver_f = ver[ver["old_base_id"].isin(valid_set)].copy()

    # ---------- Reindex mapping old -> new ----------
    # Deterministic: sorted old ids -> 0..N-1
    id_map = {old: new for new, old in enumerate(sorted(valid_ids))}

    # Reindex articles ids
    art_f["id"] = art_f["old_id"].map(id_map).astype(int)
    art_f = art_f.drop(columns=["id_norm"])

    # Reindex VERITE base ids + update image paths
    ver_f["base_id"] = ver_f["old_base_id"].map(id_map).astype(int)
    ver_f["image_path"] = [remap_path(p, bid) for p, bid in zip(ver_f["image_path"], ver_f["base_id"])]

    # ---------- Rewrite images folder safely ----------
    tmp_dir = IMAGES_DIR.parent / (IMAGES_DIR.name + "_tmp_reindexed")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Copy only valid ids into tmp with new names
    for old, new in id_map.items():
        src_t = IMAGES_DIR / f"true_{old}.jpg"
        src_f = IMAGES_DIR / f"false_{old}.jpg"
        dst_t = tmp_dir / f"true_{new}.jpg"
        dst_f = tmp_dir / f"false_{new}.jpg"

        # Should exist by definition
        if not src_t.exists() or not src_f.exists():
            raise RuntimeError(f"Internal error: missing files for old id={old} even though it was valid.")

        shutil.copy2(src_t, dst_t)
        shutil.copy2(src_f, dst_f)

    # Backup old images folder
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    IMAGES_DIR.rename(BACKUP_DIR)
    tmp_dir.rename(IMAGES_DIR)
    print(f"[images] Reindexed images/ written. Backup saved at: {BACKUP_DIR}")

    # ---------- Write outputs ----------
    art_f.to_csv(ARTICLES_OUT, index=False)
    ver_f.to_csv(VERITE_OUT, index=False)

    # ---------- Sanity checks ----------
    n = len(valid_ids)
    for i in range(n):
        if not (IMAGES_DIR / f"true_{i}.jpg").exists() or not (IMAGES_DIR / f"false_{i}.jpg").exists():
            raise RuntimeError(f"Sanity fail: missing reindexed images for id={i}")

    # Ensure no old ids remain in image_path
    bad_paths = ver_f[~ver_f["image_path"].str.match(r"^images/(true|false)_\d+\.jpg$")]
    if len(bad_paths):
        raise RuntimeError("Sanity fail: some image_path values have unexpected format.")

    print(f"[ok] wrote: {ARTICLES_OUT} ({len(art_f)} rows)")
    print(f"[ok] wrote: {VERITE_OUT} ({len(ver_f)} rows)")
    print(f"[ok] new id range: 0 .. {n-1}")

if __name__ == "__main__":
    main()
