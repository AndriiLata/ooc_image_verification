import os, numpy as np, pandas as pd
from tqdm import tqdm
from PIL import Image
from ooclab.encoders.clip_encoder import ClipEncoder

def extract_for_dataframe(df: pd.DataFrame,
                          img_root: str,
                          id_col: str = "id",
                          img_path_col: str = "image_path",
                          text_col: str = "caption",
                          encoder_version="ViT-L/14",
                          device="cuda:0",
                          batch=256):
    enc = ClipEncoder(encoder_version, device)
    ids = df[id_col].astype(str).tolist()

    # images
    imgs = []
    for p in df[img_path_col]:
        imgs.append(Image.open(os.path.join(img_root, p.strip("./"))))
    img_feats = []
    for i in tqdm(range(0, len(imgs), batch)):
        img_feats.append(enc.encode_images(imgs[i:i+batch]).cpu())
    img_feats = np.vstack([x.numpy() for x in img_feats])

    # texts
    txts = df[text_col].fillna("").astype(str).tolist()
    txt_feats = []
    for i in tqdm(range(0, len(txts), batch)):
        txt_feats.append(enc.encode_texts(txts[i:i+batch]).cpu())
    txt_feats = np.vstack([x.numpy() for x in txt_feats])

    return ids, img_feats, txt_feats

def save_embeddings(out_dir: str, prefix: str, ids, img_feats, txt_feats, encoder_version: str):
    os.makedirs(out_dir, exist_ok=True)
    tag = encoder_version.replace("-", "").replace("/", "")
    np.save(os.path.join(out_dir, f"{prefix}_item_ids_{tag}.npy"), np.array(ids))
    np.save(os.path.join(out_dir, f"{prefix}_clip_image_embeddings_{tag}.npy"), img_feats.astype("float32"))
    np.save(os.path.join(out_dir, f"{prefix}_clip_text_embeddings_{tag}.npy"), txt_feats.astype("float32"))
