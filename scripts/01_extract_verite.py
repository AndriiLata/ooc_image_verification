# scripts/01_extract_verite.py
import sys
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import yaml
import pandas as pd
from ooclab.features.extract_main import extract_for_dataframe, save_embeddings


def base_id_from_image_path(p: str) -> str:
    """
    Extract the numeric article id from paths like:
      images/true_0.jpg, images/false_12.jpg
    Returns "0", "12", ...
    """
    m = re.search(r"_(\d+)\.(jpg|jpeg|png|webp)$", str(p).strip(), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse base id from image_path: {p}")
    return m.group(1)


def sample_suffix_from_label(label: str) -> str:
    l = str(label).strip().lower()
    if l == "true":
        return "true"
    if l == "miscaptioned":
        return "mis"
    if l in {"out-of-context", "out_of_context", "ooc"}:
        return "ooc"
    # fallback
    return re.sub(r"[^a-z0-9]+", "_", l)[:12] or "unk"


def main():
    cfg = yaml.safe_load(open(REPO_ROOT / "config" / "default.yaml"))
    encv = cfg["encoder"]["version"]
    dev = cfg["encoder"]["device"]

    verite_small_root = REPO_ROOT / "data" / "verite_small"
    csv_path = verite_small_root / "VERITE_small.csv"
    if not csv_path.exists():
        raise SystemExit(f"VERITE_small.csv not found at: {csv_path}")

    outs_root = REPO_ROOT / cfg["paths"]["outputs"]
    out_dir = outs_root / "verite_small"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # required columns
    for col in ["caption", "image_path", "label"]:
        if col not in df.columns:
            raise SystemExit(f"VERITE_small.csv missing required column: {col}")

    # Build stable base_id from image_path, and unique sample id per row
    base_ids = []
    sample_ids = []
    for p, lab in zip(df["image_path"].astype(str).tolist(), df["label"].astype(str).tolist()):
        b = base_id_from_image_path(p)           # "0", "1", ...
        sfx = sample_suffix_from_label(lab)      # "true"/"mis"/"ooc"
        base_ids.append(b)
        sample_ids.append(f"{b}_{sfx}")          # "0_true", "0_mis", ...

    df["base_id"] = base_ids
    df["id"] = sample_ids
    df["image_id"] = df["id"]

    # label: misuse vs true
    df["falsified"] = (df["label"].astype(str).str.lower() != "true").astype(int)

    # keep only rows whose image exists (NO reindexing, keep ids!)
    def _ok(rel_path: str) -> bool:
        rel_path = str(rel_path).strip().lstrip("./")
        return (verite_small_root / rel_path).exists()

    mask = df["image_path"].astype(str).apply(_ok)
    missing_ct = int((~mask).sum())
    if missing_ct:
        print(f"[warn] {missing_ct} image files missing — dropping those rows (ids unchanged).")
    df = df[mask].copy()

    # encode
    ids, I, T = extract_for_dataframe(
        df,
        img_root=verite_small_root,
        id_col="id",                 # <- "0_true", "0_mis", "0_ooc"
        img_path_col="image_path",
        text_col="caption",
        encoder_version=encv,
        device=dev,
        batch=cfg["run"]["batch_size"],
    )

    # save embeddings
    save_embeddings(out_dir, "verite_small", ids, I, T, encv)

    # save labels
    df[["id", "image_id", "falsified", "base_id", "label"]].to_csv(
        out_dir / "verite_small_labels.csv",
        index=False
    )

    print(f"✓ Saved VERITE_small embeddings to {out_dir}")
    print("✓ IDs are now sample-level like: 0_true / 0_mis / 0_ooc")


if __name__ == "__main__":
    main()
