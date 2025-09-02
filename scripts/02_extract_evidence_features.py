# expects a CSV with columns: match_index, captions (list), images_paths (list)
import os, yaml, pandas as pd, numpy as np
from PIL import Image
from ooclab.encoders.clip_encoder import ClipEncoder

cfg = yaml.safe_load(open("config/default.yaml"))
encv, dev = cfg["encoder"]["version"], cfg["encoder"]["device"]
evi_csv = "data/visualnews/merged_balanced_train.csv"  # example
df = pd.read_csv(evi_csv)
df["captions"] = df["captions"].fillna("[]").apply(eval)
df["images_paths"] = df["images_paths"].fillna("[]").apply(eval)

# explode to caption-level / image-level items with ids "matchindex_i"
cap_rows, img_rows = [], []
for _, r in df.iterrows():
    idx = str(r["match_index"])
    for i, c in enumerate(r["captions"] or []):
        cap_rows.append((f"{idx}_{i}", str(c)))
    for i, p in enumerate(r["images_paths"] or []):
        img_rows.append((f"{idx}_{i}", p))

cap_df = pd.DataFrame(cap_rows, columns=["X_id","caption"])
img_df = pd.DataFrame(img_rows, columns=["X_id","image_path"])

enc = ClipEncoder(encv, dev)
# texts
text_chunks, ids_t = [], []
for i in range(0, len(cap_df), 256):
    chunk = cap_df.iloc[i:i+256]
    ids_t += chunk["X_id"].tolist()
    text_chunks.append(enc.encode_texts(chunk["caption"].tolist()).cpu().numpy())
Xtxt = np.vstack(text_chunks)
# images
img_chunks, ids_i = [], []
for i in range(0, len(img_df), 256):
    chunk = img_df.iloc[i:i+256]
    ids_i += chunk["X_id"].tolist()
    pil = [Image.open(p) for p in chunk["image_path"].tolist()]
    img_chunks.append(enc.encode_images(pil).cpu().numpy())
Ximg = np.vstack(img_chunks)

tag = encv.replace("-","").replace("/","")
outdir = "outputs/evidence"; os.makedirs(outdir, exist_ok=True)
np.save(f"{outdir}/X_text_embeddings_{tag}.npy", Xtxt.astype("float32"))
np.save(f"{outdir}/X_text_ids_{tag}.npy", np.array(ids_t))
np.save(f"{outdir}/X_image_embeddings_{tag}.npy", Ximg.astype("float32"))
np.save(f"{outdir}/X_image_ids_{tag}.npy", np.array(ids_i))
