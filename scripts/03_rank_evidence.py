import os, yaml, numpy as np, pandas as pd
from ooclab.evidence.rank import rank_evidence

cfg = yaml.safe_load(open("config/default.yaml"))
tag = cfg["encoder"]["version"].replace("-","").replace("/","")
outs = cfg["paths"]["outputs"]

# load sample splits with labels: id, image_id, falsified
train = pd.read_csv("data/visualnews/merged_balanced_train.csv")
valid = pd.read_csv("data/visualnews/merged_balanced_valid.csv")
test  = pd.read_csv("data/visualnews/merged_balanced_test.csv")

# load query embeddings
ids = np.load(f"{outs}/vn/visualnews_item_ids_{tag}.npy")
I   = np.load(f"{outs}/vn/visualnews_clip_image_embeddings_{tag}.npy").T
T   = np.load(f"{outs}/vn/visualnews_clip_text_embeddings_{tag}.npy").T
img_map = pd.DataFrame(I, index=ids).T; img_map.columns = img_map.columns.astype(str)
txt_map = pd.DataFrame(T, index=ids).T; txt_map.columns = txt_map.columns.astype(str)

# load evidence embeddings
Xti = np.load(f"{outs}/evidence/X_text_ids_{tag}.npy")
Xte = np.load(f"{outs}/evidence/X_text_embeddings_{tag}.npy").T
Xii = np.load(f"{outs}/evidence/X_image_ids_{tag}.npy")
Xie = np.load(f"{outs}/evidence/X_image_embeddings_{tag}.npy").T
Xtxt_map = pd.DataFrame(Xte, index=Xti).T; Xtxt_map.columns = Xtxt_map.columns.astype(str)
Ximg_map = pd.DataFrame(Xie, index=Xii).T; Ximg_map.columns = Ximg_map.columns.astype(str)

for name, df in [("train", train), ("valid", valid), ("test", test)]:
    ranked = rank_evidence(df, img_map, txt_map, Xie.T, list(Xii), Xte.T, list(Xti), topk=1)
    out = df.merge(ranked, on=["id", "image_id"], how="left")
    out.to_csv(f"{outs}/ranked_{name}_{tag}.csv", index=False)
