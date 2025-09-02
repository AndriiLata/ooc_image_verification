import numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

SIM_NAMES = ["img_txt","img_X_img","txt_X_img","img_X_txt","txt_X_txt","X_img_X_txt"]

def compute_row(img, txt, X_img=None, X_txt=None):
    out = {}
    out["img_txt"] = float(cosine_similarity(img, txt))
    if X_img is not None and X_img.shape[1] > 0:
        out["img_X_img"] = float(cosine_similarity(img, X_img))
        out["txt_X_img"] = float(cosine_similarity(txt, X_img))
    else:
        out["img_X_img"] = 0.0; out["txt_X_img"] = 0.0
    if X_txt is not None and X_txt.shape[1] > 0:
        out["img_X_txt"] = float(cosine_similarity(img, X_txt))
        out["txt_X_txt"] = float(cosine_similarity(txt, X_txt))
        out["X_img_X_txt"] = float(cosine_similarity(X_img, X_txt)) if X_img is not None and X_img.shape[1] > 0 else 0.0
    else:
        out["img_X_txt"]=0.0; out["txt_X_txt"]=0.0; out["X_img_X_txt"]=0.0
    return out

def build_muse_table(labeled_df: pd.DataFrame,
                     img_map, txt_map, Ximg_map, Xtxt_map,
                     use_evidence=1):
    rows = []
    for _, s in labeled_df.iterrows():
        img = img_map[str(s.image_id)].values.reshape(1,-1)
        txt = txt_map[str(s.id)].values.reshape(1,-1)
        Ximg = Ximg_map[s.img_ranked_items[:use_evidence]].T.values.reshape(1,-1) if s.img_ranked_items else np.zeros((1,0))
        Xtxt = Xtxt_map[s.txt_ranked_items[:use_evidence]].T.values.reshape(1,-1) if s.txt_ranked_items else np.zeros((1,0))
        sims = compute_row(img, txt, Ximg, Xtxt)
        sims.update({"id": s.id, "image_id": s.image_id, "falsified": int(s.falsified)})
        rows.append(sims)
    return pd.DataFrame(rows)[["id","image_id","falsified"]+SIM_NAMES]
