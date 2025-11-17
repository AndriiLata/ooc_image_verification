
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import os
import yaml
import pandas as pd
from ooclab.features.extract_main import extract_for_dataframe, save_embeddings


# ----------------- paths & config -----------------
REPO_ROOT = Path(__file__).resolve().parents[1]

cfg = yaml.safe_load(open(REPO_ROOT / "config" / "default.yaml"))
encv = cfg["encoder"]["version"]
dev = cfg["encoder"]["device"]

# where to find the small VERITE subset
verite_small_root = REPO_ROOT / "data" / "verite_small"
csv_path = verite_small_root / "VERITE_small.csv"

if not csv_path.exists():
    raise SystemExit(f"VERITE_small.csv not found at: {csv_path}")

outs_root = REPO_ROOT / cfg["paths"]["outputs"]

# ----------------- load dataframe -----------------
df = pd.read_csv(csv_path)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# add ids/labels
df["id"] = df.index.astype(str)  # string ids: "0", "1", ...
df["image_id"] = df["image_path"].astype(str)

# binary label if you want "misuse vs true"
# Option A: only miscaptioned = 1
# df["falsified"] = (df["label"].str.lower() == "miscaptioned").astype(int)
# Option B (often nicer): everything that is not "true" is misuse
df["falsified"] = (df["label"].str.lower() != "true").astype(int)

# keep only rows whose image file exists
def _ok(rel_path: str) -> bool:
    rel_path = rel_path.strip("./")
    return (verite_small_root / rel_path).exists()

mask = df["image_path"].astype(str).apply(_ok)
missing_ct = (~mask).sum()
if missing_ct:
    print(f"Warning: {missing_ct} image files missing in verite_small/images — skipping those rows.")

df = df[mask].reset_index(drop=True)
df["id"] = df.index.astype(str)  # reindex ids _after_ filtering

# ----------------- encode with CLIP via your helper -----------------
ids, I, T = extract_for_dataframe(
    df,
    img_root=verite_small_root,                  # IMPORTANT: root for "images/..."
    id_col="id",
    img_path_col="image_path",
    text_col="caption",
    encoder_version=encv,
    device=dev,
    batch=cfg["run"]["batch_size"],
)

# ----------------- save embeddings + labels -----------------
out_dir = outs_root / "verite_small"
out_dir.mkdir(parents=True, exist_ok=True)

save_embeddings(out_dir, "verite_small", ids, I, T, encv)

df[["id", "image_id", "falsified"]].to_csv(
    out_dir / "verite_small_labels.csv",
    index=False
)

print(f"✓ Saved VERITE_small embeddings to {out_dir}")
