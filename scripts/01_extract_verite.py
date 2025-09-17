import os, yaml, pandas as pd
from ooclab.features.extract_main import extract_for_dataframe, save_embeddings

cfg = yaml.safe_load(open("config/default.yaml"))
encv, dev = cfg["encoder"]["version"], cfg["encoder"]["device"]
ver_root = cfg["paths"]["verite"]
outs = cfg["paths"]["outputs"]

csv = os.path.join(ver_root, "VERITE.csv")
df = pd.read_csv(csv)
if "Unnamed: 0" in df.columns: df = df.drop(columns=["Unnamed: 0"])

# add ids/labels
df["id"] = df.index.astype(str)
df["image_id"] = df["image_path"].astype(str)
df["falsified"] = (df["label"].str.lower() == "miscaptioned").astype(int)

# keep only rows whose image file exists
def _ok(p): return os.path.exists(os.path.join(ver_root, p.strip("./")))
mask = df["image_path"].apply(_ok)
missing_ct = (~mask).sum()
if missing_ct:
    print(f"Warning: {missing_ct} image files missing — skipping those rows.")
df = df[mask].reset_index(drop=True)
df["id"] = df.index.astype(str)  # reindex ids after filtering

ids, I, T = extract_for_dataframe(
    df, img_root=ver_root,
    id_col="id", img_path_col="image_path", text_col="caption",
    encoder_version=encv, device=dev, batch=cfg["run"]["batch_size"]
)

os.makedirs(os.path.join(outs, "verite"), exist_ok=True)
save_embeddings(os.path.join(outs, "verite"), "verite", ids, I, T, encv)
df[["id","image_id","falsified"]].to_csv(os.path.join(outs,"verite","verite_labels.csv"), index=False)
print("✓ Saved embeddings to outputs/verite/")
